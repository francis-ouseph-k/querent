"""
generation/llm/local_llama.py
─────────────────────────────
The DEFAULT provider: locally hosted Qwen2.5-Coder-3B via llama.cpp.

This is a lift-and-shift of the inference block that previously lived inline in
generation/sql_generator.py. The logic is intentionally unchanged, including:

  * Approach A — Decoupled Server Mode: httpx POST to llama-server.exe's
    /v1/chat/completions endpoint (LLM_BASE_URL).
  * Approach B — In-Process Fallback: llama_cpp.Llama.create_chat_completion(),
    used when LLM_BASE_URL is empty OR the C++ server is unreachable/5xx.
  * The FIX-CHATML URL normalisation (…/v1 → …/v1/chat/completions, legacy
    /completions → /chat/completions).
  * The large console warning block on connection failure.
  * The FIX-M1 grammar-load short-circuit flag.
  * 4xx → fail hard; 5xx and RequestError → fall back in-process.

Two deliberate differences from the original:

  1. `from llama_cpp import Llama, LlamaGrammar` is now a LAZY import inside the
     in-process path instead of a module-level import. Previously, importing
     sql_generator required llama-cpp-python to be installed even if you never
     ran inference locally — which would have made LLM_PROVIDER=gemini impossible
     on a machine without the CUDA-built bindings. Nothing else changes: the
     import still happens before the first in-process generation.

  2. Hard failures raise LLMProviderError instead of returning a sentinel
     GeneratedSQL. SQLGenerator converts that back into the identical empty
     GeneratedSQL, so the observable behaviour of the pipeline is unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config.settings import settings
from generation.llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    ProviderInfo,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class LocalLlamaProvider(LLMProvider):
    """llama.cpp — HTTP server first, in-process bindings as fallback."""

    def __init__(self, *, display_name: str = "Qwen Coder 3B") -> None:
        self._display_name = display_name
        self._llm: Any | None = None                # llama_cpp.Llama, lazily built
        self._grammar: Any | None = None            # llama_cpp.LlamaGrammar
        self._grammar_checked: bool = False

    # ── provenance ────────────────────────────────────────────────────────
    def info(self) -> ProviderInfo:
        if settings.llm.base_url:
            model_id = f"{Path(settings.llm.model_path).name} @ {settings.llm.base_url}"
            label = "Local (llama.cpp server)"
        else:
            model_id = Path(settings.llm.model_path).name
            label = "Local (llama.cpp in-process)"
        return ProviderInfo(
            display_name   = self._display_name,
            provider_label = label,
            model_id       = model_id,
            is_local       = True,
        )

    # ── lazy model / grammar (unchanged from sql_generator) ───────────────
    @property
    def llm(self):
        """Lazy-load the GGUF on first in-process use."""
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:               # pragma: no cover - env dependent
                raise LLMProviderError(
                    "llama-cpp-python is not installed but the local in-process "
                    "path was reached. Install it, or set LLM_BASE_URL to a "
                    "running llama-server, or switch LLM_PROVIDER."
                ) from exc

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
    def grammar(self):
        """
        Load the GBNF grammar if the file exists AND contains real rules.
        FIX-M1 short-circuit retained: the file is read at most once.
        """
        if self._grammar_checked:
            return self._grammar

        if self._grammar is None:
            gbnf_path = Path(settings.llm.grammar_path)
            if gbnf_path.exists():
                gbnf_text = gbnf_path.read_text(encoding="utf-8")
                has_rules = any(
                    re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*::=', line)
                    for line in gbnf_text.splitlines()
                    if not line.strip().startswith("#")
                )
                if has_rules:
                    try:
                        from llama_cpp import LlamaGrammar
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

        self._grammar_checked = True
        return self._grammar

    # ── inference ─────────────────────────────────────────────────────────
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        raw_output: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        # Fall back in-process when no server URL is configured, or when the
        # HTTP attempt below fails in a recoverable way.
        use_in_process_fallback = not settings.llm.base_url

        if settings.llm.base_url:
            import httpx

            # FIX-CHATML: always target /v1/chat/completions so llama-server's
            # --chat-template chatml is applied and the model stops cleanly.
            url = settings.llm.base_url.rstrip('/')
            if url.endswith('/v1'):
                url = f"{url}/chat/completions"
            elif url.endswith('/completions') or url.endswith('/completion'):
                url = url.rsplit('/completions', 1)[0] + '/chat/completions'
            else:
                url = f"{url}/v1/chat/completions"

            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "frequency_penalty": settings.llm.frequency_penalty,
                "presence_penalty": settings.llm.presence_penalty,
            }

            try:
                # Connect fast-fail stays at 3s so an unreachable server falls
                # back to in-process immediately; the read budget comes from
                # LLM_PRIMARY_TIMEOUT_SECONDS.
                timeout_cfg = httpx.Timeout(
                    settings.llm.primary_timeout_seconds, connect=3.0
                )
                response = httpx.post(url, json=payload, timeout=timeout_cfg)
                response.raise_for_status()
                res_json = response.json()
                raw_output = res_json["choices"][0]["message"]["content"].strip()
                usage = res_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            except httpx.RequestError as conn_exc:
                self._print_unreachable_banner(conn_exc)
                logger.warning(
                    component="sql_generator",
                    event="llama_server_unreachable_fallback",
                    error=str(conn_exc),
                    note="Connection failed; dynamically falling back to in-process inference.",
                )
                use_in_process_fallback = True
            except httpx.HTTPStatusError as status_exc:
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
                    raise LLMProviderError(str(status_exc)) from status_exc
            except Exception as exc:
                logger.exception("external_inference_error")
                logger.error(component="sql_generator",
                             event="external_inference_error", error=str(exc))
                raise LLMProviderError(str(exc)) from exc

        if use_in_process_fallback:
            try:
                kwargs: dict[str, Any] = {}
                if self.grammar:
                    kwargs["grammar"] = self.grammar
                response = self.llm.create_chat_completion(
                    messages          = messages,
                    max_tokens        = max_tokens,
                    temperature       = temperature,
                    stop              = stop or [],
                    frequency_penalty = settings.llm.frequency_penalty,
                    presence_penalty  = settings.llm.presence_penalty,
                    **kwargs,
                )
                raw_output = response["choices"][0]["message"]["content"].strip()
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            except LLMProviderError:
                raise
            except Exception as exc:
                logger.exception("inference_error")
                logger.error(component="sql_generator",
                             event="inference_error", error=str(exc))
                raise LLMProviderError(str(exc)) from exc

        return LLMResponse(
            text              = raw_output or "",
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
        )

    @staticmethod
    def _print_unreachable_banner(conn_exc: Exception) -> None:
        """Highly visible console warning — unchanged text, moved verbatim."""
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
