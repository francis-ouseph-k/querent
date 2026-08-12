"""
tests/test_llm_provider_switch.py
──────────────────────────────────
Regression tests for the switchable-LLM change and the v10.5 → v10.10 DDL
upgrade.

Deliberately dependency-free: no network, no API keys, no llama-cpp-python, no
LangChain integration packages. Everything here exercises the selection and
wiring logic, which is where a provider switch actually breaks. The vendor
clients themselves are LangChain's responsibility and are not re-tested.

Run with:    pytest tests/test_llm_provider_switch.py -v
Or stand-alone:  python tests/test_llm_provider_switch.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import llm_providers as lp
from config.settings import LLMSettings, Settings


def _llm(**overrides) -> LLMSettings:
    """Build LLMSettings from explicit values, ignoring the developer's .env."""
    base = dict(_env_file=None)
    return LLMSettings(**base, **overrides)


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


class TestSQLGeneratorIsProviderAgnostic(unittest.TestCase):
    """
    The output-contract parser must behave identically no matter which provider
    produced the text — that is the whole point of the extraction.
    """

    def _generator_with(self, text):
        from generation.llm.base import LLMProvider, LLMResponse, ProviderInfo
        from generation.sql_generator import SQLGenerator

        class _Stub(LLMProvider):
            def complete(self, messages, *, max_tokens, temperature, stop=None):
                self.last_messages = messages
                return LLMResponse(text=text, prompt_tokens=11, completion_tokens=7)

            def info(self):
                return ProviderInfo("Stub", "Stub", "stub-1")

        stub = _Stub()
        return SQLGenerator(provider=stub), stub

    def test_clean_json_contract_is_parsed(self):
        gen, _ = self._generator_with(
            '{"sql": "SELECT 1", "tables_used": ["board"], '
            '"confidence": 0.9, "explanation": "ok"}'
        )
        r = gen.generate("question")
        self.assertEqual(r.sql, "SELECT 1")
        self.assertEqual(r.tables_used, ["board"])
        self.assertAlmostEqual(r.confidence, 0.9)
        self.assertEqual(r.prompt_tokens, 11)

    def test_system_role_is_passed_through(self):
        gen, stub = self._generator_with('{"sql": "SELECT 1"}')
        gen.generate("question", system="you are a sql writer")
        self.assertEqual(stub.last_messages[0]["role"], "system")
        self.assertEqual(stub.last_messages[1]["role"], "user")

    def test_provider_failure_returns_empty_generated_sql(self):
        """Same sentinel the pre-provider code returned — retry loop unaffected."""
        from generation.llm.base import LLMProvider, LLMProviderError, ProviderInfo
        from generation.sql_generator import SQLGenerator

        class _Boom(LLMProvider):
            def complete(self, messages, *, max_tokens, temperature, stop=None):
                raise LLMProviderError("network down")

            def info(self):
                return ProviderInfo("Stub", "Stub", "stub-1")

        r = SQLGenerator(provider=_Boom()).generate("question")
        self.assertEqual(r.sql, "")
        self.assertEqual(r.confidence, 0.0)


class TestSchemaVersionUpgrade(unittest.TestCase):

    ROOT = Path(__file__).resolve().parent.parent
    DDL = ROOT / "data/docs/digital_evaluation_schema_v10_10.sql"

    def test_default_ddl_path_points_at_v10_10(self):
        self.assertIn("v10_10", Settings(_env_file=None).ddl_path)

    def test_v10_10_ddl_file_exists(self):
        self.assertTrue(self.DDL.exists(), f"missing {self.DDL}")

    def test_derived_fks_declares_the_new_schema_version(self):
        import yaml
        data = yaml.safe_load((self.ROOT / "config/derived_fks.yaml").read_text(encoding="utf-8"))
        self.assertEqual(str(data["schema_version"]), "10.10")

    def test_derived_fk_edges_still_exist_in_v10_10(self):
        """
        The five derived (comment-inferred) FK edges are not enforced by the DDL,
        so nothing else would catch it if v10.10 had renamed one of their
        columns. Assert every source/target column is still present.
        """
        import yaml
        from ingestion.ddl_parser import DDLParser

        tables = DDLParser().parse_file(self.DDL)
        data = yaml.safe_load((self.ROOT / "config/derived_fks.yaml").read_text(encoding="utf-8"))

        def col_names(table):
            return {c if isinstance(c, str) else c.name for c in tables[table].columns}

        for edge in data["derived_fks"]:
            src, tgt = edge["source_table"], edge["target_table"]
            self.assertIn(src, tables, f"derived FK source table {src} missing from v10.10")
            self.assertIn(tgt, tables, f"derived FK target table {tgt} missing from v10.10")
            for mapping in edge["column_mappings"]:
                self.assertIn(mapping["source_column"], col_names(src),
                              f"{src}.{mapping['source_column']} missing from v10.10")
                self.assertIn(mapping["target_column"], col_names(tgt),
                              f"{tgt}.{mapping['target_column']} missing from v10.10")


if __name__ == "__main__":
    unittest.main(verbosity=2)
