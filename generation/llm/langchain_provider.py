"""
generation/llm/langchain_provider.py
────────────────────────────────────
LangChain-backed provider. One class serves EVERY hosted model.

DESIGN NOTE — why one wrapper instead of one class per vendor
─────────────────────────────────────────────────────────────
LangChain's BaseChatModel already normalises the vendor differences that would
otherwise become conditionals in this codebase: message roles, the request
shape, retry/timeout handling, and — via `usage_metadata` — the token accounting
that Mistral and Gemini report under different keys. So the per-vendor code is
reduced to "which BaseChatModel do I construct, with which kwargs", and that
lives in factory.py as a small builder function. This class holds the *shared*
adaptation: role-dict → LangChain messages, AIMessage → LLMResponse.

Adding a fourth hosted provider therefore means adding one builder function in
factory.py and one registry entry. No change here, and no change in
SQLGenerator or the retry loop.

CONTENT NORMALISATION
─────────────────────
AIMessage.content is `str` for most providers, but the multi-modal content-block
list (`[{"type": "text", "text": ...}, ...]`) is also valid and Gemini can emit
it. `_content_to_text` collapses both to a plain string so the downstream
JSON-contract parser in SQLGenerator never sees a list.
"""

from __future__ import annotations

from typing import Any

import random
import re
import time

from generation.llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    ProviderInfo,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Transient-failure retry
# ─────────────────────────────────────────────────────────────────────────────
# FIX-P1 (batch run 20260812). Mistral returned HTTP 429 four times across a
# 74-minute, 191-question run:
#
#   Error code: 429 - {'message': 'Rate limit exceeded', 'type': 'rate_limited',
#                      'code': '1300', 'raw_status_code': 429}
#
# There was no retry at this layer, so each 429 became an empty GeneratedSQL.
# Two surfaced as scored accuracy failures; two consumed a validation retry that
# should have gone to fixing SQL. BATCH_QUERY_DELAY_MS=500 between questions was
# not enough, and a fixed inter-question delay is the wrong instrument anyway —
# it slows the whole run to pace the worst moment.
#
# Retry here, not in the runner: this is transport, and the runner's retry budget
# exists for SQL correction. Full jitter avoids a thundering herd when several
# workers back off together.
_TRANSIENT_STATUS_RE = re.compile(r"\b(429|500|502|503|504)\b")
_TRANSIENT_MARKERS = (
    "rate limit",
    "rate_limited",
    "overloaded",
    "capacity",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
)
_MAX_TRANSIENT_ATTEMPTS = 4      # 1 initial + 3 retries
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 16.0


def _is_transient(exc: Exception) -> bool:
    """True for provider errors worth retrying (429 / 5xx / explicit capacity)."""
    text = str(exc).lower()
    if _TRANSIENT_STATUS_RE.search(text):
        return True
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _backoff_seconds(attempt: int) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2**attempt))."""
    ceiling = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** attempt))
    return random.uniform(0.0, ceiling)


def _content_to_text(content: Any) -> str:
    """Collapse str | list[content-block] into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # LangChain text blocks use {"type": "text", "text": "..."}
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "".join(parts).strip()
    return str(content).strip()


class LangChainChatProvider(LLMProvider):
    """Wraps any LangChain BaseChatModel behind the LLMProvider port."""

    def __init__(self, chat_model: Any, info: ProviderInfo) -> None:
        self._model = chat_model
        self._info = info

    def info(self) -> ProviderInfo:
        return self._info

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        # max_tokens / temperature are bound onto the chat model at construction
        # time (factory.py) because that is where LangChain expects them; they
        # are accepted here only to satisfy the shared LLMProvider signature.
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages: list[Any] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        ai = None
        last_exc: Exception | None = None
        for attempt in range(_MAX_TRANSIENT_ATTEMPTS):
            try:
                ai = self._model.invoke(lc_messages)
                break
            except Exception as exc:
                last_exc = exc
                transient = _is_transient(exc)
                if transient and attempt < _MAX_TRANSIENT_ATTEMPTS - 1:
                    delay = _backoff_seconds(attempt)
                    logger.warning(
                        component="sql_generator",
                        event="external_inference_retry",
                        provider=self._info.provider_label,
                        model=self._info.model_id,
                        attempt=attempt + 1,
                        max_attempts=_MAX_TRANSIENT_ATTEMPTS,
                        backoff_s=round(delay, 2),
                        error=str(exc)[:200],
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    component="sql_generator",
                    event="external_inference_error",
                    provider=self._info.provider_label,
                    model=self._info.model_id,
                    attempts=attempt + 1,
                    transient=transient,
                    error=str(exc),
                )
                if transient:
                    # Distinct type so the runner can label this a provider
                    # failure rather than scoring it as a wrong answer.
                    raise LLMRateLimitError(str(exc)) from exc
                raise LLMProviderError(str(exc)) from exc

        if ai is None:  # pragma: no cover — loop always breaks or raises
            raise LLMProviderError(str(last_exc) if last_exc else "no response")

        usage = getattr(ai, "usage_metadata", None) or {}
        return LLMResponse(
            text              = _content_to_text(getattr(ai, "content", "")),
            prompt_tokens     = usage.get("input_tokens"),
            completion_tokens = usage.get("output_tokens"),
        )
