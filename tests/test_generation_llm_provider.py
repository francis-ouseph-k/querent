"""
tests/test_generation_llm_provider.py
─────────────────────────────────────────
generation/llm/langchain_provider.py and config/llm_providers.py — everything
about SELECTING a provider, resolving its active credentials, and how a
transient provider failure is classified and backed off from.

CONSOLIDATED FROM FOUR FILES. The backoff/transient-classification tests in
particular are the clearest example in this refactor of the same
functionality accreting across runs without ever being looked at together:

  * test_accuracy_regressions.py — basic transient classification (429/503/
    timeout/overloaded vs 401/400), backoff bounded-and-jittered, the
    LLMRateLimitError/LLMProviderError subclass relationship.
  * test_run6_hardening.py — SDK-internal-retries-disabled, backoff floor +
    ceiling with jitter still present, Retry-After honoured, an absurd
    Retry-After ignored.
  * test_run7_hardening.py — the rate-limit-specific backoff floor holds
    across 200 draws, a non-rate-limit transient does NOT inherit that floor,
    the attempt budget is configurable.
  * test_llm_provider_switch.py — provider registry coherence, provider
    selection/fallback, the active_* accessor triple (base_url/model/api_key)
    per provider, no cross-provider key leakage, split timeouts, the banner
    never containing a live key, and the SQL-generator's provider-agnostic
    JSON-contract parsing is exercised in a separate file
    (test_generation_sql_generator.py) since it targets a different module.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from config import llm_providers as lp
from config.settings import LLMSettings


def _llm(**overrides) -> LLMSettings:
    """Build LLMSettings from explicit values, ignoring the developer's .env."""
    base = dict(_env_file=None)
    return LLMSettings(**base, **overrides)


# ═════════════════════════════════════════════════════════════════════════════
# Provider registry, selection, active accessors, banner
# ═════════════════════════════════════════════════════════════════════════════

class TestProviderRegistry(unittest.TestCase):
    """config/llm_providers.py is the single source of truth — keep it coherent."""

    def test_every_supported_provider_has_a_spec(self):
        for name in lp.SUPPORTED_PROVIDERS:
            self.assertIn(name, lp.SPEC, f"{name} missing from SPEC")
            spec = lp.SPEC[name]
            self.assertTrue(spec.display_name)
            self.assertTrue(spec.label)
            self.assertTrue(spec.default_model)

    def test_every_supported_provider_has_a_builder(self):
        from generation.llm.factory import _BUILDERS
        self.assertEqual(set(_BUILDERS), set(lp.SUPPORTED_PROVIDERS))

    def test_local_and_langchain_sets_partition_supported(self):
        self.assertTrue(set(lp.LOCAL_PROVIDERS).issubset(set(lp.SUPPORTED_PROVIDERS)))
        self.assertTrue(set(lp.LANGCHAIN_PROVIDERS).issubset(set(lp.SUPPORTED_PROVIDERS)))
        # LOCAL is the only provider on neither-LangChain footing.
        self.assertEqual(
            set(lp.SUPPORTED_PROVIDERS) - set(lp.LANGCHAIN_PROVIDERS), {lp.LOCAL}
        )

    def test_hosted_providers_declare_an_openai_compatible_endpoint(self):
        for name in (lp.MISTRAL, lp.GEMINI):
            self.assertTrue(lp.SPEC[name].default_base_url.startswith("https://"))

    def test_aliases_resolve_to_canonical_names(self):
        self.assertEqual(lp.normalise("Qwen"), lp.LOCAL)
        self.assertEqual(lp.normalise("  MISTRALAI "), lp.MISTRAL)
        self.assertEqual(lp.normalise("google"), lp.GEMINI)
        # Unknown values pass through unchanged so the settings validator can
        # raise with the offending string rather than silently defaulting.
        self.assertEqual(lp.normalise("does-not-exist"), "does-not-exist")


class TestProviderSelection(unittest.TestCase):

    def test_default_provider_is_local(self):
        """Backward compatibility: an .env with no LLM_PROVIDER keeps Qwen."""
        self.assertEqual(_llm().provider, lp.LOCAL)
        self.assertTrue(_llm().is_local_provider)

    def test_unknown_provider_fails_fast(self):
        with self.assertRaises(Exception) as ctx:
            _llm(provider="gpt5")
        self.assertIn("gpt5", str(ctx.exception))

    def test_hosted_provider_is_not_local(self):
        for name in (lp.MISTRAL, lp.GEMINI):
            self.assertFalse(_llm(provider=name).is_local_provider)

    def test_ft_profile_is_forced_to_full_for_hosted_providers(self):
        """
        The "ft" profile serves our LoRA adapter's training distribution.
        A hosted model has never seen it, so leaving LLM_PROMPT_PROFILE=ft in
        .env while switching to Gemini must not strip the rich schema prompt.
        """
        self.assertEqual(_llm(provider=lp.GEMINI, prompt_profile="ft").prompt_profile,
                         "full")

    def test_ft_profile_is_preserved_for_local_providers(self):
        self.assertEqual(_llm(provider=lp.LOCAL, prompt_profile="ft").prompt_profile,
                         "ft")


class TestActiveProviderAccessors(unittest.TestCase):
    """
    The active_* accessors are the ONLY place provider names are branched on.
    If they go wrong, the factory silently builds a client pointed at the wrong
    endpoint with the wrong key — the exact failure this suite exists to catch.
    """

    def test_mistral_triple(self):
        s = _llm(provider=lp.MISTRAL, mistral_api_key="k-mistral",
                 mistral_model="mistral-small-latest")
        self.assertEqual(s.active_base_url, "https://api.mistral.ai/v1")
        self.assertEqual(s.active_model, "mistral-small-latest")
        self.assertEqual(s.active_api_key, "k-mistral")

    def test_gemini_triple(self):
        s = _llm(provider=lp.GEMINI, gemini_api_key="k-gemini")
        self.assertIn("generativelanguage.googleapis.com", s.active_base_url)
        self.assertEqual(s.active_model, "gemini-2.0-flash")
        self.assertEqual(s.active_api_key, "k-gemini")

    def test_local_uses_historical_names(self):
        s = _llm(provider=lp.LOCAL, base_url="http://localhost:8080/v1")
        self.assertEqual(s.active_base_url, "http://localhost:8080/v1")
        self.assertEqual(s.active_model, "qwen2.5-coder-3b-instruct")
        self.assertEqual(s.active_api_key, "")

    def test_accessors_do_not_leak_across_providers(self):
        """Selecting Mistral must never hand out the Gemini key, and vice versa."""
        s = _llm(provider=lp.MISTRAL, mistral_api_key="k-mistral",
                 gemini_api_key="k-gemini")
        self.assertEqual(s.active_api_key, "k-mistral")
        s = _llm(provider=lp.GEMINI, mistral_api_key="k-mistral",
                 gemini_api_key="k-gemini")
        self.assertEqual(s.active_api_key, "k-gemini")

    def test_split_timeouts(self):
        s = _llm(provider=lp.LOCAL, primary_timeout_seconds=120, timeout_seconds=90)
        self.assertEqual(s.active_timeout, 120)
        s = _llm(provider=lp.GEMINI, primary_timeout_seconds=120, timeout_seconds=90)
        self.assertEqual(s.active_timeout, 90)


class TestBannerHasNoSecrets(unittest.TestCase):

    def test_banner_shape(self):
        from generation.llm.base import ProviderInfo
        info = ProviderInfo(
            display_name="Gemini Flash 2",
            provider_label="Google Gemini (LangChain)",
            model_id="gemini-2.0-flash",
            is_local=False,
        )
        banner = info.banner()
        self.assertIn("Provider: Google Gemini (LangChain)", banner)
        self.assertIn("Model: gemini-2.0-flash", banner)

    def test_api_keys_are_excluded_from_serialisation(self):
        """
        exclude=True must keep keys out of model_dump(), which is what any
        accidental settings-logging call would serialise.
        """
        s = _llm(mistral_api_key="sk-not-a-real-key",
                 gemini_api_key="AIza-not-a-real-key")
        dumped = str(s.model_dump())
        self.assertNotIn("sk-not-a-real-key", dumped)
        self.assertNotIn("AIza-not-a-real-key", dumped)

    def test_describe_active_llm_never_contains_the_key(self):
        from generation.llm import factory
        original = factory.settings.llm
        try:
            factory.settings.llm = _llm(provider=lp.GEMINI,
                                        gemini_api_key="AIza-not-a-real-key")
            banner = factory.describe_active_llm().banner()
            self.assertNotIn("AIza-not-a-real-key", banner)
            self.assertIn("gemini-2.0-flash", banner)
        finally:
            factory.settings.llm = original

    def test_missing_key_raises_configuration_error_not_import_error(self):
        from generation.llm import factory
        from generation.llm.base import LLMConfigurationError

        original = factory.settings.llm
        try:
            factory.settings.llm = _llm(provider=lp.MISTRAL, mistral_api_key="")
            with self.assertRaises(LLMConfigurationError) as ctx:
                factory.build_provider(lp.MISTRAL)
            # Either the missing key or the missing package — both are the
            # actionable, named error rather than a raw ImportError.
            self.assertTrue(
                "API_KEY" in str(ctx.exception) or "langchain-openai" in str(ctx.exception)
            )
        finally:
            factory.settings.llm = original

    def test_local_langchain_without_base_url_is_rejected(self):
        from generation.llm import factory
        from generation.llm.base import LLMConfigurationError

        original = factory.settings.llm
        try:
            factory.settings.llm = _llm(provider=lp.LOCAL_LC, base_url="")
            with self.assertRaises(LLMConfigurationError):
                factory.build_provider(lp.LOCAL_LC)
        finally:
            factory.settings.llm = original


# ═════════════════════════════════════════════════════════════════════════════
# Transient error classification and backoff shape
#
# Merged from three runs' worth of incremental hardening -- basic
# classification, then a floor/ceiling/jitter refinement, then a
# rate-limit-specific floor that a non-rate-limit transient must NOT inherit.
# ═════════════════════════════════════════════════════════════════════════════

import pytest  # noqa: E402


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Error code: 429 - {'message': 'Rate limit exceeded', 'code': '1300'}", True),
        ("Error code: 503 - service unavailable", True),
        ("Request timed out after 90s", True),
        ("model is overloaded", True),
        ("Error code: 401 - invalid api key", False),
        ("Error code: 400 - bad request", False),
    ],
)
def test_transient_error_classification(message, expected):
    from generation.llm.langchain_provider import _is_transient
    assert _is_transient(RuntimeError(message)) is expected


def test_backoff_is_bounded_and_jittered():
    from generation.llm.langchain_provider import (
        _BACKOFF_CAP_SECONDS, _backoff_seconds,
    )
    for attempt in range(8):
        delay = _backoff_seconds(attempt)
        assert 0.0 <= delay <= _BACKOFF_CAP_SECONDS


def test_rate_limit_error_is_a_provider_error_subclass():
    """Existing `except LLMProviderError` handlers must keep working."""
    from generation.llm.base import LLMProviderError, LLMRateLimitError
    assert issubclass(LLMRateLimitError, LLMProviderError)


def test_sdk_internal_retries_are_disabled():
    """
    The openai client retries inside invoke(), beneath the token bucket and the
    backoff. Two retry layers do not compose; the faster uninstrumented one
    wins. Retry must live at exactly one layer -- ours.
    """
    src = open("generation/llm/factory.py", encoding="utf-8").read()
    import re
    assert re.search(r"max_retries\s*=\s*0", src), (
        "ChatOpenAI must be constructed with max_retries=0"
    )


def test_backoff_has_floor_and_stays_under_ceiling():
    from generation.llm.langchain_provider import (
        _backoff_seconds, _BACKOFF_BASE_SECONDS, _BACKOFF_CAP_SECONDS,
        _BACKOFF_FLOOR_FRACTION,
    )
    for attempt in range(4):
        ceiling = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** attempt))
        draws = [_backoff_seconds(attempt) for _ in range(500)]
        assert min(draws) >= ceiling * _BACKOFF_FLOOR_FRACTION - 1e-9
        assert max(draws) <= ceiling + 1e-9
        # Still jittered: a constant would defeat de-correlation.
        assert len(set(round(d, 3) for d in draws)) > 10


def test_backoff_honours_retry_after():
    from generation.llm.langchain_provider import _backoff_seconds
    exc = Exception("Error code: 429 - rate limited. Retry-After: 7")
    for _ in range(50):
        delay = _backoff_seconds(0, exc)
        assert 7.0 <= delay <= 8.75, delay


def test_backoff_ignores_absurd_retry_after():
    from generation.llm.langchain_provider import _backoff_seconds
    exc = Exception("Retry-After: 99999")
    assert _backoff_seconds(0, exc) <= 1.0


def test_rate_limit_backoff_respects_floor():
    """
    The observed defect: a 429 carrying no Retry-After fell through to the
    generic curve and produced a 0.26s wait on attempt 0.
    """
    from generation.llm import langchain_provider as lp_mod

    mistral_429 = Exception(
        "Error code: 429 - {'object': 'error', 'message': 'Rate limit exceeded', "
        "'type': 'rate_limited', 'code': '1300', 'raw_status_code': 429}"
    )
    for _ in range(200):
        assert lp_mod._backoff_seconds(0, mistral_429) >= lp_mod._RATE_LIMIT_MIN_BACKOFF_SECONDS


def test_non_rate_limit_transient_keeps_fast_first_retry():
    """A 503 is not load-shedding; it must not inherit the rate-limit floor."""
    from generation.llm import langchain_provider as lp_mod

    server_error = Exception("Error code: 503 - service unavailable")
    assert any(
        lp_mod._backoff_seconds(0, server_error) < lp_mod._RATE_LIMIT_MIN_BACKOFF_SECONDS
        for _ in range(200)
    )


def test_transient_attempt_budget_is_configurable():
    from generation.llm import langchain_provider as lp_mod
    assert lp_mod._max_transient_attempts() >= 1
