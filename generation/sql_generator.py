"""
generation/sql_generator.py
────────────────────────────
LLM inference via llama.cpp Python bindings.

Key features:
  - GBNF grammar-constrained decoding (forces SELECT-first, no markdown)
  - Structured output contract parsing  { sql, tables_used, confidence, explanation }
  - Retry loop integration (correction prompt on validation failure)
  - Lazy model loading (model loads on first request, not at import)

Hardware: Qwen2.5-Coder 3B Q4_K_M ≈ 2.4 GB VRAM on 8 GB GPU.

FIXES IN THIS VERSION
─────────────────────
FIX-L1 — Two logger.warning() calls in _parse_output() passed the event name
          as a positional argument instead of a keyword argument, inconsistent
          with the rest of the codebase and breaking log filtering by event=.
          Fix: converted to event="..." kwarg pattern matching all other callers.
"""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any

from llama_cpp import Llama, LlamaGrammar

from config.settings import settings
from models.schema import GeneratedSQL
from utils.logging_config import get_logger

logger = get_logger(__name__)

# JSON extraction pattern — handles cases where the model wraps JSON in prose
_JSON_RE = re.compile(r'\{[^{}]*"sql"[^{}]*\}', re.DOTALL)


def _unescape_json_string(s: str) -> str:
    """Unescape a raw JSON string fragment extracted via regex.

    When extracting the "sql" field directly via regex (Try 3 / Try 4 in
    _parse_output), we bypass json.loads.  JSON encodes newlines as the
    two-char sequence \\n, tabs as \\t, etc.  Without unescaping, the
    extracted SQL contains literal backslash-n which crashes sqlglot.

    Uses json.loads for correctness — it handles every JSON escape sequence
    (\\n \\r \\t \\b \\f \\/ \\uXXXX \\\\ \\") exactly as RFC 8259 specifies.
    Falls back to manual replacement only if the fragment is too malformed
    for json.loads (e.g. truncated mid-escape).
    """
    try:
        return json.loads(f'"{s}"').strip()
    except (json.JSONDecodeError, ValueError):
        # Fragment too broken for json.loads — best-effort manual unescape.
        return (s.replace('\\n', '\n')
                 .replace('\\r', '\r')
                 .replace('\\t', '\t')
                 .replace('\\"', '"')
                 .replace('\\\\', '\\')
                 .strip())


class SQLGenerator:
    """
    Wraps llama.cpp for SQL generation.

    Usage:
        gen    = SQLGenerator()
        result = gen.generate(prompt)
    """

    def __init__(self) -> None:
        self._llm: Llama | None         = None
        self._grammar: LlamaGrammar | None = None
        # FIX-M1: flag set to True after the first grammar load attempt.
        # The original sentinel was "if self._grammar is None" — but when
        # has_rules was False, self._grammar was never set to a non-None value,
        # so the property re-read and re-parsed the GBNF file on every single
        # generate() call.  This flag short-circuits the file I/O after the
        # first attempt regardless of outcome.
        self._grammar_checked: bool = False

    @property
    def llm(self) -> Llama:
        """Lazy-load the model on first use."""
        if self._llm is None:
            model_path = settings.llm.model_path
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"Download Qwen2.5-Coder-3B-Q4_K_M.gguf and place it at this path."
                )
            logger.info(
                component="sql_generator",
                event="loading_model",
                path=model_path,
                n_ctx=settings.llm.context_size,
                n_gpu_layers=settings.llm.n_gpu_layers,
            )
            self._llm = Llama(
                model_path    = model_path,
                n_ctx         = settings.llm.context_size,
                n_gpu_layers  = settings.llm.n_gpu_layers,
                n_threads     = settings.llm.n_threads,
                verbose       = False,
            )
            logger.info(component="sql_generator", event="model_loaded")
        return self._llm

    @property
    def grammar(self) -> LlamaGrammar | None:
        """
        Load GBNF grammar if the file exists AND contains actual grammar rules.

        FIX #1: The original GBNF grammar was a no-op placeholder
        (`select-body ::= [^\\x00]+` matches everything). It has been replaced
        with a comments-only file documenting the decision to rely on the
        JSON extraction fallback + sqlglot AST validation instead.

        This property skips grammar loading when:
        - The file does not exist
        - The file contains no lines that look like grammar rules (::=)
        - LlamaGrammar.from_string raises (malformed grammar)

        In all skip cases, llama.cpp generates freely and the output is
        handled by the JSON extraction + validation pipeline.
        """
        # FIX-M1: short-circuit after the first load attempt.
        # Without this, when has_rules is False the property returns None
        # without setting self._grammar to any non-None value, causing every
        # subsequent generate() call to re-read and re-parse the file.
        if self._grammar_checked:
            return self._grammar

        if self._grammar is None:
            gbnf_path = Path(settings.llm.grammar_path)
            if gbnf_path.exists():
                gbnf_text = gbnf_path.read_text(encoding="utf-8")
                # FIX M6 — original used "::=" in line which matched "::="
                # anywhere including comment lines like "# format is ::=".
                # Using a stricter regex that requires a valid rule identifier
                # (starting with a letter or underscore) at the line start.
                has_rules = any(
                    re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*::=', line)
                    for line in gbnf_text.splitlines()
                    if not line.strip().startswith("#")
                )
                if has_rules:
                    try:
                        self._grammar = LlamaGrammar.from_string(gbnf_text)
                        logger.info(component="sql_generator", event="grammar_loaded",
                                    path=str(gbnf_path))
                    except Exception as exc:
                        logger.warning(component="sql_generator", event="grammar_load_failed",
                                       path=str(gbnf_path), error=str(exc),
                                       note="Proceeding without grammar constraints")
                else:
                    logger.info(component="sql_generator", event="grammar_skipped",
                                path=str(gbnf_path),
                                note="No grammar rules found — using JSON extraction fallback")
            else:
                logger.warning(component="sql_generator", event="grammar_not_found",
                               path=str(gbnf_path),
                               note="Proceeding without grammar constraints")

        # FIX-M1: mark checked so subsequent calls skip this block entirely
        self._grammar_checked = True
        return self._grammar

    def generate(self, prompt: str) -> GeneratedSQL:
        """
        Run the LLM on the prompt and parse the structured output contract.

        Returns a GeneratedSQL with .sql, .tables_used, .confidence, .explanation.
        On parse failure, returns a GeneratedSQL with empty sql and confidence=0.
        """
        t0 = time.time()

        kwargs: dict[str, Any] = {
            "max_tokens":  settings.llm.max_tokens,
            "temperature": settings.llm.temperature,
            # FIX-CHATML: stop tokens kept as a defensive safety net.
            # With the chat completions API the ChatML template handles
            # <|im_end|> automatically, so the model should stop without
            # needing explicit stop tokens.  These are retained in case
            # the model overshoots — they cause no harm on the chat endpoint.
            "stop": ["</s>", "<|im_end|>", "<|im_start|>"],
        }
        if self.grammar:
            kwargs["grammar"] = self.grammar

        # FIX-CHATML: Wrap the prompt as a ChatML message.
        # Qwen2.5-Coder is an instruct-tuned model trained on ChatML format
        # (<|im_start|>user\n...\n<|im_end|>).  Both the HTTP and in-process
        # paths now send `messages` instead of a raw prompt string so the
        # model receives proper turn boundaries and stops cleanly after
        # generating the JSON output contract.
        messages = [{"role": "user", "content": prompt}]

        prompt_tokens = None
        completion_tokens = None
        
        # Determine whether to fall back to the in-process local execution.
        # This will be set to True if LLM_BASE_URL is not configured OR if
        # a connection to the external server fails.
        use_in_process_fallback = not settings.llm.base_url

        if settings.llm.base_url:
            import httpx
            # FIX-CHATML: Build URL for /v1/chat/completions (not /v1/completions).
            # The raw /v1/completions endpoint sends the prompt as plain text,
            # bypassing the --chat-template chatml flag on llama-server.exe.
            # Without ChatML framing the model doesn't know when to stop and
            # generates until max_tokens (2048), wasting minutes per query.
            # The /v1/chat/completions endpoint applies the ChatML template
            # so the model stops cleanly after the JSON output contract.
            url = settings.llm.base_url.rstrip('/')
            if url.endswith('/v1'):
                url = f"{url}/chat/completions"
            elif url.endswith('/completions') or url.endswith('/completion'):
                # Legacy: user configured a raw completions URL — upgrade to chat
                url = url.rsplit('/completions', 1)[0] + '/chat/completions'
            else:
                url = f"{url}/v1/chat/completions"

            # FIX-CHATML: Send messages array instead of raw prompt string.
            payload = {
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.2),
                "frequency_penalty": settings.llm.frequency_penalty,  # Pass frequency penalty to prevent repetition loops (e.g. repeating same SQL clauses)
                "presence_penalty": settings.llm.presence_penalty,    # Pass presence penalty to encourage vocabulary/concept diversity
            }
            
            # ── Approach A: Decoupled Server Mode (HTTP REST Client) ─────────
            # Connects directly to the external llama-server.exe process.
            # Highly optimized C++ server execution, fast, and stable.
            try:
                # Set a strict 3.0-second limit to establish the socket connection,
                # but allow up to 120.0 seconds to stream back tokens once connected.
                timeout_cfg = httpx.Timeout(120.0, connect=3.0)
                response = httpx.post(url, json=payload, timeout=timeout_cfg)
                response.raise_for_status()
                res_json = response.json()
                # FIX-CHATML: Chat completions response nests content under
                # choices[].message.content (not choices[].text).
                raw_output = res_json["choices"][0]["message"]["content"].strip()
                usage = res_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            except httpx.RequestError as conn_exc:
                # Catch connection refusals, network timeouts, and host unreachable
                # errors to dynamically trigger the local in-process fallback.
                #
                # Display a highly visible, huge warning block in the terminal console
                # for both main.py (CLI) and batch_run.py (evaluations) to alert the user.
                print()
                print("=" * 80)
                print(" WARNING: EXTERNAL LLAMA-SERVER UNREACHABLE ".center(80, "!"))
                print("=" * 80)
                print(f" Could not connect to the C++ server at: {settings.llm.base_url}")
                print(f" Error details: {conn_exc}")
                print()
                print(" Dynamic fallback triggered:")
                print(" -> Running in-process loading (llama-cpp-python) inside Python.")
                print(" -> WARNING: CPU/local inference is significantly slower!")
                print()
                print(" To resolve this and run at full GPU speed:")
                print(" 1. Open a new, separate terminal window.")
                print(" 2. Start the llama-server manually using this command:")
                print("    .\\llama-server.exe `")
                print("        -m ..\\models\\qwen\\qwen2.5-coder-3b-instruct-q4_k_m.gguf `")
                print("        -ngl 28 `")
                print("        -c 16384 `")
                print("        --chat-template chatml")
                print("=" * 80)
                print()
                
                logger.warning(
                    component="sql_generator",
                    event="llama_server_unreachable_fallback",
                    error=str(conn_exc),
                    note="Connection failed; dynamically falling back to in-process inference."
                )
                use_in_process_fallback = True
            except httpx.HTTPStatusError as status_exc:
                # 5xx = server crash/overload — fall back to in-process inference.
                # 4xx = API contract mismatch (bad request shape) — fail hard.
                if status_exc.response.status_code >= 500:
                    logger.warning(
                        component="sql_generator",
                        event="llama_server_5xx_fallback",
                        status_code=status_exc.response.status_code,
                        note="Server error; dynamically falling back to in-process inference.",
                    )
                    use_in_process_fallback = True
                else:
                    logger.error(
                        component="sql_generator",
                        event="llama_server_4xx_error",
                        status_code=status_exc.response.status_code,
                        error=str(status_exc),
                    )
                    return GeneratedSQL(sql="", raw_output="", confidence=0.0)
            except Exception as exc:
                # Non-HTTP errors (JSON decode, unexpected response shape, etc.)
                # indicate an API contract mismatch — fail hard, don't mask.
                logger.exception("external_inference_error")
                logger.error(component="sql_generator", event="external_inference_error", error=str(exc))
                return GeneratedSQL(sql="", raw_output="", confidence=0.0)

        if use_in_process_fallback:
            # ── Approach B: In-Process Inference (Local Bindings) ───────────
            # Loads the GGUF model file directly into Python memory using llama_cpp.Llama.
            # Completely offline and self-contained, but slower on CPU.
            #
            # FIX-CHATML: Use create_chat_completion() instead of raw __call__().
            # This applies the ChatML template (same as --chat-template chatml
            # on llama-server.exe) so the model receives proper <|im_start|> /
            # <|im_end|> framing and stops cleanly after generating the JSON
            # output contract, instead of babbling to max_tokens.
            try:
                response = self.llm.create_chat_completion(
                    messages    = messages,
                    max_tokens  = kwargs.get("max_tokens", settings.llm.max_tokens),
                    temperature = kwargs.get("temperature", settings.llm.temperature),
                    stop        = kwargs.get("stop", []),
                    frequency_penalty = settings.llm.frequency_penalty,  # Pass frequency penalty to prevent repetition loops (e.g. repeating same SQL clauses)
                    presence_penalty = settings.llm.presence_penalty,    # Pass presence penalty to encourage vocabulary/concept diversity
                )
                # FIX-CHATML: Chat completions response nests content under
                # choices[].message.content (not choices[].text).
                raw_output = response["choices"][0]["message"]["content"].strip()
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            except Exception as exc:
                logger.exception("inference_error")
                logger.error(component="sql_generator", event="inference_error", error=str(exc))
                return GeneratedSQL(sql="", raw_output="", confidence=0.0)

        elapsed_ms = round((time.time() - t0) * 1000)
        logger.info(
            component="sql_generator",
            event="inference_complete",
            elapsed_ms=elapsed_ms,
            output_len=len(raw_output),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


        return self._parse_output(raw_output, prompt_tokens, completion_tokens)

    def _parse_output(
        self,
        raw: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> GeneratedSQL:
        """
        Parse the structured JSON output contract from the LLM.

        Expected format:
        {
          "sql": "SELECT ...",
          "tables_used": ["table1", "table2"],
          "confidence": 0.85,
          "explanation": "..."
        }

        Four-layer fallback chain — each layer handles a progressively more
        degraded form of model output:

          Try 1 — clean JSON:        model followed the contract perfectly
          Try 2 — JSON in prose:     model added preamble/postamble around JSON
          Try 3 — truncated JSON:    context limit cut output before closing }
          Try 4 — raw SELECT:        JSON entirely absent; extract SQL directly
        """
        # Strip markdown fences if GBNF grammar was not active.
        # The model sometimes wraps output in ```json ... ``` blocks.
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$",          "", cleaned)

        # ── Try 1: Direct JSON parse ──────────────────────────────────────
        # Happy path — model output is a well-formed JSON object.
        try:
            data = json.loads(cleaned)
            return _dict_to_generated_sql(data, raw, prompt_tokens, completion_tokens)
        except json.JSONDecodeError:
            pass

        # ── Try 2: Extract JSON object from within prose ──────────────────
        # Model added preamble ("Here is the SQL:") or postamble before/after
        # the JSON block.  The regex finds the first {...} that contains a
        # "sql" key.  Nested braces (e.g. JSONB literals) would break _JSON_RE
        # since it uses [^{}]* — acceptable trade-off for a 3B model that
        # rarely uses JSONB in its output contract.
        match = _JSON_RE.search(cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
                return _dict_to_generated_sql(data, raw, prompt_tokens, completion_tokens)
            except json.JSONDecodeError:
                pass

        # ── Try 3: Extract "sql" field from malformed / truncated JSON ─────
        #
        # Root cause this fixes: when the prompt fills 96–99% of the context
        # window, the model runs out of generation budget before it can emit
        # the closing }.  The JSON is syntactically incomplete but the "sql"
        # field value — which appears first in the contract — is still intact.
        #
        # The regex captures everything after the opening " of the sql value,
        # including JSON-escaped characters (\\. handles \n, \t, \", \\, etc.)
        # stopping only at an unescaped closing " or end-of-string.
        #
        # CRITICAL — unescape after extraction:
        # Inside a JSON string, newlines are encoded as the two-character
        # sequence \n (backslash + n).  json.loads handles this automatically
        # on a successful parse.  Here we are bypassing json.loads, so we must
        # unescape manually.  Without this step the extracted SQL contains
        # literal \n sequences which crash the sqlglot tokenizer with:
        #   TokenError: Missing " from 3:678
        #
        # This Try MUST run before Try 4 (raw SELECT) because Try 4 would
        # extract the same SQL but without unescaping, producing broken SQL.
        sql_field_match = re.search(
            r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)',
            cleaned,
            re.DOTALL,
        )
        if sql_field_match:
            sql = _unescape_json_string(sql_field_match.group(1))
            if sql.upper().startswith('SELECT'):
                # FIX-L1: event as kwarg — consistent with rest of codebase
                logger.warning(
                    component="sql_generator",
                    event="json_truncated_sql_extracted",
                    sql_preview=sql[:100],
                    note="JSON truncated by context limit — sql field extracted directly",
                )
                return GeneratedSQL(
                    sql         = sql,
                    raw_output  = raw,
                    confidence  = 0.0,
                    explanation = "SQL extracted from truncated JSON — review carefully",
                    prompt_tokens = prompt_tokens,
                    completion_tokens = completion_tokens,
                )

        # ── Try 4: Last resort — raw SELECT extraction ────────────────────
        #
        # JSON is entirely absent (model ignored the output contract).
        # Extract the first SELECT statement from the raw output.
        #
        # CRITICAL termination fix:
        # The original regex used (?:;|$) as the terminator.  JSON never
        # contains semicolons, so with re.DOTALL the match always extended to
        # end-of-string, pulling in the entire JSON tail as part of the SQL:
        #
        #   SELECT ... FROM ...", "tables_used": ["board"], "confidence": 0.95
        #
        # sqlglot then tried to tokenize the JSON fragment and crashed with:
        #   TokenError: Error tokenizing ', "answer_script", "board"]...'
        #
        # The lookahead (?=\s*"[a-z_]+"\s*:) fires the moment the remaining
        # text looks like a JSON key (e.g. ", "tables_used": ["), cutting the
        # match cleanly before the metadata even if no semicolon is present.
        #
        # Unescape for the same reason as Try 3 — see note above.
        sql_match = re.search(
            r'(SELECT\s+.+?)(?:;|(?=\s*"[a-z_]+"\s*:)|$)',
            cleaned,
            re.DOTALL | re.IGNORECASE,
        )
        if sql_match:
            sql = _unescape_json_string(sql_match.group(1))
            # FIX-L1: event as kwarg — consistent with rest of codebase
            logger.warning(
                component="sql_generator",
                event="json_parse_failed_extracted_sql",
                sql_preview=sql[:100],
                note="JSON contract not honoured — confidence set to 0.0",
            )
            return GeneratedSQL(
                sql         = sql,
                raw_output  = raw,
                confidence  = 0.0,
                explanation = "SQL extracted from non-JSON output — review carefully",
                prompt_tokens = prompt_tokens,
                completion_tokens = completion_tokens,
            )

        # All four layers exhausted — model produced no parseable output.
        # Logged as error so it surfaces clearly in the failure log.
        logger.error(component="sql_generator", event="parse_failed_completely", raw=raw[:200])
        return GeneratedSQL(sql="", raw_output=raw, confidence=0.0, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _dict_to_generated_sql(
    data: dict[str, Any],
    raw: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> GeneratedSQL:
    """Convert a parsed JSON dict to a GeneratedSQL object."""
    sql         = str(data.get("sql", "")).strip()
    tables_used = data.get("tables_used", [])
    confidence  = float(data.get("confidence", 0.5))
    explanation = str(data.get("explanation", ""))

    # Normalise tables_used to list of strings
    if isinstance(tables_used, str):
        tables_used = [t.strip() for t in tables_used.split(",")]

    return GeneratedSQL(
        sql         = sql,
        tables_used = [str(t) for t in tables_used],
        confidence  = min(max(confidence, 0.0), 1.0),   # clamp [0, 1]
        explanation = explanation,
        raw_output  = raw,
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
    )