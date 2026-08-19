"""
tests/test_models_schema.py
───────────────────────────────
models/schema.py — dataclass-level contracts that are not specific to any one
validator.

CONSOLIDATED FROM: test_security_hardening.py.
"""

from __future__ import annotations


def test_validation_result_carries_explicit_autofix_flag():
    """Calibration must not infer 'an autofix ran' from a SQL text diff --
    that is also true whenever a tenant filter is injected."""
    from models.schema import ValidationResult

    assert ValidationResult(passed=True).autofix_applied is False
    assert ValidationResult(passed=True, autofix_applied=True).autofix_applied is True
