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
from .rate_limiter import get_shared_rate_limiter

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
# Transport attempt budget. Previously a module constant edited by hand
# (4 -> 9) with no way to set it per deployment; now driven by
# LLM_TRANSIENT_MAX_ATTEMPTS.
#
# Deliberately NOT wired to VALIDATION_MAX_RETRIES. That budget buys
# SQL-CORRECTION round trips -- each one sends a different prompt and can
# change the answer. This budget buys re-sends of an IDENTICAL request the
# provider shed. Sharing one number between them means a rate-limit storm
# silently consumes the correction budget, or a raised correction budget
# silently multiplies load on an endpoint already returning 429.
#
# Read through a function rather than bound at import time, so this module
# keeps its existing property of not importing config.settings on import.
_DEFAULT_TRANSIENT_ATTEMPTS = 4      # 1 initial + 3 retries


def _max_transient_attempts() -> int:
    """Configured transport attempt budget, falling back to the default."""
    try:
        from config.settings import settings
        value = int(getattr(settings.llm, "transient_max_attempts", 0))
    except Exception:
        return _DEFAULT_TRANSIENT_ATTEMPTS
    return value if value >= 1 else _DEFAULT_TRANSIENT_ATTEMPTS
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 16.0

# FIX-R2 (batch run 20260817_133854). Full jitter -- uniform(0, ceiling) -- is
# the correct family and is retained; what it lacks is a FLOOR. A uniform draw
# can return ~0.05s, and re-issuing a request 50ms after the provider said
# "rate limit exceeded" cannot succeed: it burns one of only four attempts and
# adds load to an endpoint that is already shedding it. The floor keeps the
# de-correlating property of jitter (the spread is still random, so parallel
# callers do not resynchronise) while guaranteeing the wait is long enough to
# be worth making. Expressed as a fraction of the ceiling rather than a fixed
# constant so it scales with the attempt number instead of dominating the
# first retry and vanishing on the last.
_BACKOFF_FLOOR_FRACTION = 0.25

# A provider that tells us exactly how long to wait is more authoritative than
# any backoff curve we can compute. Parsed from the error text because the
# LangChain wrapper does not surface response headers on the exception.
_RETRY_AFTER_RE = re.compile(r"retry[-_ ]?after[\"'\s:=]+([0-9]+(?:\.[0-9]+)?)", re.I)

# A 429 that states no wait at all. Mistral's is the shape logged throughout
# batch run 20260818_085111:
#   {'object': 'error', 'message': 'Rate limit exceeded',
#    'type': 'rate_limited', 'code': '1300', 'raw_status_code': 429}
# No retry-after field appears anywhere in that payload, so _RETRY_AFTER_RE
# has nothing to match and the generic curve takes over -- which produced
# logged waits of 0.26s and 1.6s immediately after the provider said it was
# shedding load, across 284 retries in one run. A sub-second re-send into an
# active rate-limit window cannot succeed: it burns an attempt and adds load
# to an endpoint already refusing it.
#
# Rate limiting is not the same failure as a transient 5xx and does not
# deserve the same opening backoff, so give it a floor of its own. The
# exponential growth and the jitter both still apply on top; this only
# raises the starting point, and only for errors the provider labelled as
# rate limiting.
_RATE_LIMIT_MIN_BACKOFF_SECONDS = 4.0
_RATE_LIMIT_MARKERS = ("rate limit", "rate_limited", "429")


def _is_rate_limit(exc: Exception | None) -> bool:
    """True when the provider is explicitly shedding load, not merely erroring."""
    if exc is None:
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _is_transient(exc: Exception) -> bool:
    """True for provider errors worth retrying (429 / 5xx / explicit capacity)."""
    text = str(exc).lower()
    if _TRANSIENT_STATUS_RE.search(text):
        return True
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Provider-specified wait, when the error carries one. None otherwise."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if not match:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    # Ignore absurd values rather than stalling the run on a malformed header.
    if value <= 0 or value > _BACKOFF_CAP_SECONDS * 4:
        return None
    return value


def _backoff_seconds(attempt: int, exc: Exception | None = None) -> float:
    """
    Jittered exponential backoff with a floor, honouring Retry-After.

    uniform(ceiling * floor_fraction, ceiling), ceiling = min(cap, base*2**n).
    Still full-jitter in character -- the draw is random across a wide band, so
    concurrent callers de-correlate -- but never returns a wait too short to
    let a rate limit window advance. When the provider states a Retry-After,
    that value wins outright: it is ground truth, not an estimate.
    """
    if exc is not None:
        stated = _retry_after_seconds(exc)
        if stated is not None:
            # Small positive jitter so simultaneous callers handed the SAME
            # Retry-After do not all resume on the identical instant.
            return stated + random.uniform(0.0, 0.25 * stated)
    ceiling = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** attempt))
    floor = ceiling * _BACKOFF_FLOOR_FRACTION
    if _is_rate_limit(exc):
        floor = max(floor, _RATE_LIMIT_MIN_BACKOFF_SECONDS)
        ceiling = max(ceiling, floor * 2)
    return random.uniform(floor, ceiling)


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
        limiter = get_shared_rate_limiter()
        max_attempts = _max_transient_attempts()
        for attempt in range(max_attempts):
            try:
                if limiter is not None:
                    # FIX-A3. Proactive throttle: block here, before the
                    # request is sent, rather than only reacting after a 429.
                    # A generous timeout (2x the backoff cap) means a badly
                    # mistuned rate degrades to the OLD reactive-only
                    # behaviour rather than hanging the pipeline.
                    try:
                        limiter.acquire(timeout=_BACKOFF_CAP_SECONDS * 2)
                    except TimeoutError:
                        logger.warning(
                            component="sql_generator",
                            event="rate_limiter_wait_exceeded",
                            note="proceeding without waiting further; "
                                 "reactive backoff will handle a 429",
                        )
                ai = self._model.invoke(lc_messages)
                break
            except Exception as exc:
                last_exc = exc
                transient = _is_transient(exc)
                if transient and attempt < max_attempts - 1:
                    delay = _backoff_seconds(attempt, exc)
                    logger.warning(
                        component="sql_generator",
                        event="external_inference_retry",
                        provider=self._info.provider_label,
                        model=self._info.model_id,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
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
