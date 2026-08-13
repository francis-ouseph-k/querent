"""
generation/llm/base.py
──────────────────────
Provider-agnostic contract for the LLM querying layer.

WHY THIS MODULE EXISTS
──────────────────────
Before this change, generation/sql_generator.py owned three concerns at once:

    1. transport        (httpx POST to llama-server, or llama-cpp-python in-process)
    2. provider quirks  (ChatML framing, GBNF grammar, llama.cpp usage keys)
    3. output contract  (the four-layer JSON/SQL parsing fallback chain)

Adding Mistral and Gemini by extending (1) and (2) in place would have meant
provider-specific `if` branches threaded through the generator and, indirectly,
through the retry loop that calls it. Instead the transport and provider quirks
move behind this one interface, and SQLGenerator keeps ONLY concern (3) — the
output contract parser, which is identical for every provider because every
provider is asked for the same JSON contract.

The interface is deliberately narrow: one `complete()` call in, one
`LLMResponse` out. Anything a provider needs beyond that (API keys, base URLs,
grammars, timeouts) is read from config.settings at construction time by
generation/llm/factory.py, never passed down the call chain.

ERROR SEMANTICS
───────────────
Providers raise LLMProviderError for any unrecoverable inference failure.
SQLGenerator catches it once and returns the same empty GeneratedSQL
(sql="", confidence=0.0) the pre-change code returned, so the retry loop in
pipeline/runner.py and validation/core/sql_validator.py sees no behavioural
difference. Providers must NOT raise for recoverable conditions they can
handle internally (e.g. the local provider's llama-server → in-process
fallback).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMProviderError(RuntimeError):
    """Unrecoverable inference failure. Caller returns an empty GeneratedSQL."""


class LLMRateLimitError(LLMProviderError):
    """
    Provider refused the request for capacity reasons (HTTP 429 / 5xx) after the
    provider's own bounded retries were exhausted.

    Kept distinct from LLMProviderError so callers can separate "the model got
    the SQL wrong" from "the API did not answer". In the 20260812 benchmark this
    conflation cost two scored failures (Q51, Q102 — the latter being
    `bundle.expected_count > 50`, a one-table query) and silently consumed retry
    budget on two more, because a 429 surfaced to the runner as the same empty
    GeneratedSQL an accuracy failure produces.
    """


class LLMConfigurationError(RuntimeError):
    """Provider selected in .env but not usable (missing key / missing package)."""


@dataclass(frozen=True)
class LLMResponse:
    """
    One completion. `text` is the raw assistant content, unparsed — the JSON
    output-contract parsing stays in SQLGenerator._parse_output().

    Token counts are optional: llama.cpp and both hosted providers report them,
    but nothing in the pipeline depends on their presence (they flow into
    retrieval_meta for telemetry only).
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ProviderInfo:
    """
    Startup provenance for the active LLM. Contains NO secrets by construction —
    only the human-readable model family, the provider label, and the model id.
    API keys never enter this object, so `banner()` is always safe to print and
    to write to the structured log.
    """

    display_name: str          # e.g. "Qwen Coder 3B"
    provider_label: str        # e.g. "Local (llama.cpp)"
    model_id: str              # e.g. "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    is_local: bool = True

    def banner(self) -> str:
        return (
            f"NL\u2192SQL LLM: {self.display_name} | "
            f"Provider: {self.provider_label} | "
            f"Model: {self.model_id}"
        )


class LLMProvider(ABC):
    """
    Minimal chat-completion port.

    `messages` uses the OpenAI/ChatML role dict shape already produced by
    SQLGenerator.generate() — [{"role": "system"|"user", "content": str}] —
    so no call site had to change when this layer was introduced.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Run one completion. Raise LLMProviderError on unrecoverable failure."""

    @abstractmethod
    def info(self) -> ProviderInfo:
        """Secret-free provenance for the startup banner and provenance logs."""
