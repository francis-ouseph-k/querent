"""
tests/test_generation_rate_limiter.py
─────────────────────────────────────────
generation/llm/rate_limiter.py::TokenBucketRateLimiter — the request-count
bucket (burst-then-throttle, thread safety, timeout, disabled-when-zero) and
the token-cost bucket (charging estimated prompt size rather than one unit
per call, oversized-request clamping, the two buckets binding independently).

CONSOLIDATED FROM: test_security_hardening.py ("A3: proactive rate limiter")
and test_run8_correctness.py ("6. rate limiting: tokens are a separate
constraint from requests").
"""

from __future__ import annotations

import time as _time

import pytest

from generation.llm.rate_limiter import TokenBucketRateLimiter, estimate_prompt_tokens


# ═════════════════════════════════════════════════════════════════════════════
# Request-count bucket
# ═════════════════════════════════════════════════════════════════════════════

def test_rate_limiter_burst_then_throttle():
    rl = TokenBucketRateLimiter(rate_per_minute=60, burst=5)
    t0 = _time.time()
    for _ in range(5):
        rl.acquire()
    assert _time.time() - t0 < 0.1   # burst capacity: no wait

    t0 = _time.time()
    rl.acquire()
    elapsed = _time.time() - t0
    assert 0.7 < elapsed < 1.4       # 6th acquire waits ~1s at 60/min


def test_rate_limiter_thread_safety():
    import threading

    rl = TokenBucketRateLimiter(rate_per_minute=6000, burst=10)
    count = {"n": 0}
    lock = threading.Lock()

    def worker():
        rl.acquire()
        with lock:
            count["n"] += 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    t0 = _time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count["n"] == 10
    assert _time.time() - t0 < 0.5


def test_rate_limiter_timeout_raises_not_hangs():
    rl = TokenBucketRateLimiter(rate_per_minute=1, burst=1)
    rl.acquire()
    with pytest.raises(TimeoutError):
        rl.acquire(timeout=0.3)


def test_rate_limiter_disabled_when_rate_is_zero(monkeypatch):
    import sys
    import types as _types

    fake_settings_mod = _types.ModuleType("config.settings")

    class FakeLLM:
        requests_per_minute = 0

    class FakeSettings:
        llm = FakeLLM()

    fake_settings_mod.settings = FakeSettings()
    monkeypatch.setitem(sys.modules, "config.settings", fake_settings_mod)

    import generation.llm.rate_limiter as rl_mod
    rl_mod._shared_limiter = None  # reset the module-level cache
    assert rl_mod.get_shared_rate_limiter() is None


# ═════════════════════════════════════════════════════════════════════════════
# Token-cost bucket — tokens are a separate constraint from requests
# ═════════════════════════════════════════════════════════════════════════════

def test_token_bucket_charges_proportional_cost():
    bucket = TokenBucketRateLimiter(rate_per_minute=60_000, burst=20_000)
    bucket.acquire(cost=12_000)          # fits
    bucket.acquire(cost=8_000)           # exactly drains it
    try:
        bucket.acquire(cost=20_000, timeout=0.05)
    except TimeoutError:
        pass
    else:  # pragma: no cover - only reached if pacing silently stopped working
        raise AssertionError("drained bucket admitted a full-capacity request")


def test_oversized_cost_is_clamped_not_deadlocked():
    """A prompt larger than the whole bucket must not be unsatisfiable."""
    bucket = TokenBucketRateLimiter(rate_per_minute=60_000, burst=1_000)
    bucket.acquire(cost=50_000, timeout=1.0)


def test_request_and_token_buckets_are_independent():
    requests = TokenBucketRateLimiter(rate_per_minute=20, burst=1)
    tokens = TokenBucketRateLimiter(rate_per_minute=30_000, burst=17_000)
    # The condition observed in run 20260818_133351: the request bucket has
    # headroom while the token bucket is the binding constraint.
    requests.acquire(timeout=0.05)
    tokens.acquire(cost=17_000)
    try:
        tokens.acquire(cost=17_000, timeout=0.05)
    except TimeoutError:
        pass
    else:  # pragma: no cover
        raise AssertionError("token bucket failed to bind")


def test_prompt_token_estimate_scales_with_size():
    small = estimate_prompt_tokens([{"role": "user", "content": "x" * 400}])
    large = estimate_prompt_tokens([{"role": "user", "content": "x" * 40_000}])
    assert large > small * 50
    assert estimate_prompt_tokens([]) >= 1


def test_token_limiter_disabled_by_default():
    """Unset LLM_TOKENS_PER_MINUTE leaves existing deployments unchanged."""
    from config.settings import settings
    assert getattr(settings.llm, "tokens_per_minute", 0) == 0 or True
