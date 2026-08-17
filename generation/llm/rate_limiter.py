"""
generation/llm/rate_limiter.py
──────────────────────────────
Proactive, client-side request throttling for LLM calls.

WHY THIS EXISTS ALONGSIDE THE EXISTING BACKOFF

langchain_provider.py already retries a 429 with full-jitter exponential
backoff (base 1s, cap 16s, 4 attempts) -- that mechanism is correct as
written and is NOT what this module replaces. Full jitter is the standard
AWS-recommended algorithm precisely because it deliberately produces small
delays sometimes (a low draw from the uniform distribution is not a bug); a
review of this codebase that called it "sub-second and therefore broken"
was reading individual draws instead of the distribution.

What backoff cannot do is stop the 429 from happening in the first place.
Backoff is REACTIVE: it only engages after a request has already been
rejected, and every rejection is still latency and still a chance to
exhaust the retry budget on a run doing many requests back-to-back (as a
batch benchmark does). BATCH_QUERY_DELAY_MS existed as a first attempt at
being proactive, but a FIXED delay between every request paces the whole
run to the worst moment -- too slow when the provider has headroom, and
`sql_validator.py`'s own retry loop (which also calls the LLM) is not
covered by it at all, since that delay only runs between top-level batch
questions.

A token bucket paces requests to a configured rate regardless of where in
the pipeline they originate -- top-level generation, a validator-driven
retry, anything that goes through LangChainChatProvider.complete() -- and
only slows things down when the configured rate would actually be
exceeded, so it costs nothing when the provider has headroom.

TUNING: the default below (LLM_REQUESTS_PER_MINUTE) is deliberately
conservative and MUST be tuned to the actual provider tier in use; the
mechanism is correct for any rate, but the right number depends on account-
specific information (Mistral plan tier) this codebase has no way to
discover on its own. Set it too low and throughput suffers with no
correctness cost; set it too high and 429s return, in which case the
existing backoff still catches them -- the two mechanisms are complementary,
not either/or.
"""

from __future__ import annotations

import threading
import time

from utils.logging_config import get_logger

logger = get_logger(__name__)


class TokenBucketRateLimiter:
    """
    Classic token bucket: capacity tokens, refilled continuously at
    `rate_per_minute / 60` tokens per second. `acquire()` blocks the calling
    thread until a token is available, then consumes one.

    Thread-safe: a single instance is meant to be shared across every LLM
    call in the process (batch_run.py's questions, and any validator-driven
    retry within pipeline/runner.py), which is why refill and consumption
    are guarded by one lock rather than given a bucket per caller.
    """

    def __init__(self, rate_per_minute: float, burst: float | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._rate_per_sec = rate_per_minute / 60.0
        # Allow a short burst up to the full per-minute rate by default, so a
        # cold start (or a lull followed by a flurry of retries) is not
        # artificially throttled below what the provider actually allows.
        self._capacity = burst if burst is not None else rate_per_minute
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
        self._last_refill = now

    def acquire(self, *, timeout: float | None = None) -> None:
        """
        Block until a token is available, then consume it. `timeout`, if
        given, is a ceiling on total wait; exceeding it raises TimeoutError
        rather than blocking forever, so a badly-mistuned rate cannot hang
        the whole pipeline silently.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate_per_sec
            if deadline is not None and time.monotonic() + wait > deadline:
                raise TimeoutError(
                    f"rate limiter wait ({wait:.1f}s) would exceed timeout"
                )
            time.sleep(min(wait, 0.25))   # re-check periodically, not one long sleep


_shared_limiter: TokenBucketRateLimiter | None = None
_shared_lock = threading.Lock()


def get_shared_rate_limiter() -> TokenBucketRateLimiter | None:
    """
    Lazily construct the process-wide limiter from settings on first use, so
    importing this module never triggers a settings load / has no import-
    order dependency on config.settings. Returns None when disabled
    (rate_per_minute <= 0), so callers can skip acquire() entirely rather
    than pay a lock + no-op refill on every request.
    """
    global _shared_limiter
    if _shared_limiter is not None:
        return _shared_limiter
    with _shared_lock:
        if _shared_limiter is not None:
            return _shared_limiter
        from config.settings import settings
        rate = getattr(settings.llm, "requests_per_minute", 0)
        if not rate or rate <= 0:
            return None
        _shared_limiter = TokenBucketRateLimiter(rate_per_minute=rate)
        logger.info(
            component="sql_generator",
            event="rate_limiter_initialized",
            requests_per_minute=rate,
        )
        return _shared_limiter
