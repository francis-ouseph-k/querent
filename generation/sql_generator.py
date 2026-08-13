"""
generation/sql_generator.py
────────────────────────────
Structured-output SQL generation.

RESPONSIBILITY (narrowed)
─────────────────────────
This module used to own transport, provider quirks, AND output parsing. The
first two moved to generation/llm/ (see generation/llm/base.py for why). What
remains here is the part that is genuinely provider-independent:

  - build the ChatML message list (optional system role + user turn)
  - ask the active provider for a completion
  - parse the { sql, tables_used, confidence, explanation } output contract,
    including the four-layer fallback chain for degraded model output

Switching LLM_PROVIDER between local Qwen, Mistral, and Gemini therefore
changes nothing in this file, and nothing in the callers
(pipeline/runner.py, validation/core/sql_validator.py) — SQLGenerator.generate()
keeps its exact signature and return type.

Hardware (default provider): Qwen2.5-Coder 3B Q4_K_M ≈ 2.4 GB VRAM on 8 GB GPU.

FIXES IN THIS VERSION
─────────────────────
FIX-L1 — Two logger.warning() calls in _parse_output() passed the event name
          as a positional argument instead of a keyword argument, inconsistent
          with the rest of the codebase and breaking log filtering by event=.
          Fix: converted to event="..." kwarg pattern matching all other callers.
PROVIDER — Inference extracted behind generation.llm.LLMProvider. Hard provider
          failures now surface as LLMProviderError and are converted back into
          the identical empty GeneratedSQL(sql="", confidence=0.0) the previous
          code returned, so the retry loop is unaffected. The module-level
          `from llama_cpp import ...` is gone: llama-cpp-python is only imported
          when the local in-process path actually runs.
"""

from __future__ import annotations
import json
import re
import time
from typing import Any

from config.settings import settings
from generation.llm import (
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    get_provider,
)
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
    Generate SQL through whichever provider LLM_PROVIDER selects.

    Usage (unchanged):
        gen    = SQLGenerator()
        result = gen.generate(prompt)
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        # Injectable for tests; None means "use the process-wide provider that
        # LLM_PROVIDER selects". Resolution is lazy so that importing this
        # module never constructs an API client or loads a 2.4 GB GGUF — the
        # same lazy-load property the previous implementation had for Llama.
        self._provider: LLMProvider | None = provider

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider()
        return self._provider

    def generate(self, prompt: str, system: str | None = None) -> GeneratedSQL:
        """
        Run the LLM on the prompt and parse the structured output contract.

        system — optional system-role content (FIX-F1). When provided, messages
        become [{"role":"system"...},{"role":"user"...}] so the ChatML template
        renders a real system turn — the shape the fine-tuned adapter was
        trained on. When None (default), behaviour is unchanged: one user
        message, matching the base-model "full" profile. Hosted providers get
        the same two-message shape, mapped to SystemMessage/HumanMessage by
        generation/llm/langchain_provider.py.

        Returns a GeneratedSQL with .sql, .tables_used, .confidence, .explanation.
        On parse failure or provider failure, returns a GeneratedSQL with empty
        sql and confidence=0.
        """
        t0 = time.time()

        # FIX-CHATML: Wrap the prompt as ChatML messages. Qwen2.5-Coder is an
        # instruct-tuned model trained on ChatML; the hosted providers expect
        # the same role structure, so one message list serves all providers.
        messages = ([{"role": "system", "content": system}] if system else []) \
                   + [{"role": "user", "content": prompt}]

        # Stop tokens are a defensive safety net for the local ChatML path only;
        # providers that do not need them ignore the argument.
        stop = ["</s>", "<|im_end|>", "<|im_start|>"]

        try:
            response = self.provider.complete(
                messages,
                max_tokens  = settings.llm.max_tokens,
                temperature = settings.llm.temperature,
                stop        = stop,
            )
        except LLMRateLimitError as exc:
            # FIX-P1: transport failure, not a modelling failure. The provider
            # already exhausted its own bounded backoff. Tag the sentinel so the
            # runner reports "provider_error" instead of scoring this as a wrong
            # answer and spending SQL-correction retries on it.
            return GeneratedSQL(
                sql="", raw_output="", confidence=0.0,
                explanation=f"provider_error: {exc}",
            )
        except LLMProviderError:
            # Provider already logged the specific failure (event=
            # external_inference_error / inference_error / llama_server_4xx_error).
            # Same sentinel the pre-provider code returned — the retry loop in
            # pipeline/runner.py treats it as "empty SQL" and moves on.
            return GeneratedSQL(sql="", raw_output="", confidence=0.0)

        raw_output = response.text
        elapsed_ms = round((time.time() - t0) * 1000)
        logger.info(
            component="sql_generator",
            event="inference_complete",
            provider=settings.llm.provider,
            elapsed_ms=elapsed_ms,
            output_len=len(raw_output),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

        return self._parse_output(
            raw_output, response.prompt_tokens, response.completion_tokens
        )

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
                # FIX-P2: this event previously claimed "JSON truncated by
                # context limit". Measured on the 20260812 run, both firings had
                # completion_tokens of 306 and 451 against LLM_MAX_TOKENS=4096,
                # and the whole run's maximum was 1597 — nothing was truncated.
                # The real cause is malformed JSON (unescaped newlines or quotes
                # inside the sql field). The old wording pointed debugging at
                # token limits that were never the problem.
                logger.warning(
                    component="sql_generator",
                    event="json_sql_field_extracted",
                    sql_preview=sql[:100],
                    completion_tokens=completion_tokens,
                    max_tokens=settings.llm.max_tokens,
                    note="JSON object not closeable — sql field extracted directly. "
                         "Compare completion_tokens against max_tokens before "
                         "assuming truncation; malformed escaping is the usual cause.",
                )
                return GeneratedSQL(
                    sql         = sql,
                    raw_output  = raw,
                    confidence  = 0.0,
                    explanation = "SQL extracted from unparseable JSON — review carefully",
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