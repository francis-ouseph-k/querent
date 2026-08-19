"""
tests/test_generation_sql_generator.py
──────────────────────────────────────────
generation/sql_generator.py::SQLGenerator — the output-contract parser must
behave identically no matter which provider produced the text, plus the
confidence-calibration formula applied to its result.

CONSOLIDATED FROM: test_llm_provider_switch.py
(TestSQLGeneratorIsProviderAgnostic — filed under "provider switch"
originally, but it targets sql_generator.py, a distinct module, so it gets
its own file here) and test_security_hardening.py ("A6: confidence
calibration").
"""

from __future__ import annotations

import unittest


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


# ═════════════════════════════════════════════════════════════════════════════
# A6: confidence calibration
# ═════════════════════════════════════════════════════════════════════════════

def test_a6_calibration_penalizes_retries_and_autofix():
    calib_rules = {"retries_needed": {"penalty_per_retry": 0.03},
                    "autofix_applied": {"penalty": 0.05}}

    def simulate(raw_conf, retries, original_sql, validated_sql):
        calibrated = raw_conf
        retry_rule = calib_rules.get("retries_needed", {})
        if retries > 0:
            calibrated -= retry_rule.get("penalty_per_retry", 0.03) * retries
        autofix_rule = calib_rules.get("autofix_applied", {})
        if original_sql and validated_sql.strip() != original_sql.strip():
            calibrated -= autofix_rule.get("penalty", 0.05)
        return max(0.0, round(calibrated, 2))

    assert simulate(0.98, 0, "SELECT 1", "SELECT 1") == 0.98
    assert simulate(0.95, 2, "SELECT 1", "SELECT 1") == 0.89
    assert simulate(0.95, 0, "SELECT as.id FROM t as",
                     "SELECT ans_a.id FROM t ans_a") == 0.90
    assert simulate(0.05, 5, "SELECT as.id", "SELECT ans_a.id") == 0.0
