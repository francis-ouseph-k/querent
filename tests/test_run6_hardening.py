"""
tests/test_run6_hardening.py
────────────────────────────
Regression contract for the fixes made after batch run 20260817_133854.

Every check here is generic: none matches a query by number, none hardcodes a
schema value, and the vocabulary cases read their expected values from the DDL
itself rather than from a literal in this file.
"""

from __future__ import annotations

import random
import re

import pytest
import sqlglot
import sqlglot.expressions as exp


# ── FIX-R2: SDK retry disabled, backoff floor + Retry-After ──────────────────

def test_sdk_internal_retries_are_disabled():
    """
    The openai client retries inside invoke(), beneath the token bucket and the
    backoff. Two retry layers do not compose; the faster uninstrumented one
    wins. Retry must live at exactly one layer -- ours.
    """
    src = open("generation/llm/factory.py", encoding="utf-8").read()
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


# ── FIX: sensitive-column explicit-request matching is plural-aware ──────────

@pytest.mark.parametrize("question,column,expected", [
    # Plural head noun -- the form that silently gutted a real answer.
    ("Show the S3 version IDs for all scan uploads of script 1.", "s3_version_id", True),
    ("Show the s3 version id for script 1", "s3_version_id", True),
    ("List DEKs with their wrapped keys", "wrapped_key", True),
    ("Show the KEK external IDs", "external_id", True),
    # Must NOT fire when the column was never asked for.
    ("List all DEKs that use the AES-256-GCM algorithm.", "iv", False),
    ("List scan history entries where the operator gave a reason.", "s3_version_id", False),
])
def test_explicit_request_matches_singular_and_plural(question, column, expected):
    from validation.security.exposure import _explicitly_requested
    assert _explicitly_requested(question, column) is expected


# ── FIX: satisfiability — always-empty and always-true predicates ────────────

class _Col:
    def __init__(self, is_pk=False):
        self.is_pk = is_pk


class _Idx:
    def __init__(self, columns, unique=True, partial=False):
        self.columns, self.is_unique, self.is_partial = columns, unique, partial


class _Inv:
    def __init__(self, cols, pk="id", uniq=()):
        self.columns = {c: _Col(c == pk) for c in cols}
        self.indexes = [_Idx(list(u)) for u in uniq]


_SAT_SCHEMA = {
    "configuration": _Inv(["id", "config_key", "version"]),
    "scanner_device": _Inv(["id", "device_name", "is_active"]),
    "result_history": _Inv(["id", "result_id"]),
    "enrolment": _Inv(["id", "student_id", "course_id"],
                      uniq=[("student_id", "course_id")]),
}


class _SatCtx:
    def __init__(self, sql):
        self.sql, self.working_sql = sql, None
        self.schema_map = _SAT_SCHEMA
        try:
            self.ast = sqlglot.parse(sql, dialect="postgres")
        except Exception:
            self.ast = None


@pytest.mark.parametrize("name,should_pass,sql", [
    # Unsatisfiable: grouping by a unique key pins COUNT(*) to 1.
    ("pk_group_count_gt_1", False,
     "SELECT c.id FROM configuration c GROUP BY c.id HAVING COUNT(*) > 1"),
    ("composite_unique_group_count_gt_1", False,
     "SELECT e.student_id FROM enrolment e GROUP BY e.student_id, e.course_id "
     "HAVING COUNT(*) > 1"),
    # Always true: the two branches exhaust the value space.
    ("having_eq0_or_gt0", False,
     "SELECT sd.device_name FROM scanner_device sd GROUP BY sd.device_name "
     "HAVING COUNT(DISTINCT sd.id) = 0 OR COUNT(DISTINCT sd.id) > 0"),
    ("where_gte_or_lt_same_bound", False,
     "SELECT c.id FROM configuration c WHERE c.version >= 5 OR c.version < 5"),
    # Must NOT fire.
    ("non_unique_group_is_legitimate", True,
     "SELECT rh.result_id FROM result_history rh GROUP BY rh.result_id HAVING COUNT(*) > 1"),
    ("count_gte_1_is_satisfiable", True,
     "SELECT c.id FROM configuration c GROUP BY c.id HAVING COUNT(*) >= 1"),
    ("count_of_column_not_star", True,
     "SELECT c.id FROM configuration c GROUP BY c.id HAVING COUNT(c.config_key) > 1"),
    ("joined_scope_breaks_the_guarantee", True,
     "SELECT c.id FROM configuration c JOIN result_history rh ON rh.result_id = c.id "
     "GROUP BY c.id HAVING COUNT(*) > 1"),
    ("genuine_or_filter", True,
     "SELECT c.id FROM configuration c WHERE c.version > 5 OR c.version < 2"),
    ("or_across_different_columns", True,
     "SELECT c.id FROM configuration c WHERE c.version = 0 OR c.id > 0"),
    ("unparseable_never_fires", True, 'SELECT list.",'),
])
def test_satisfiability(name, should_pass, sql):
    from validation.ast.satisfiability import SatisfiabilityValidator
    result = SatisfiabilityValidator().run(_SatCtx(sql))
    assert result.passed is should_pass, (name, result.message)


# ── FIX: array vocabulary harvested from the DDL's own seed rows ─────────────

@pytest.fixture(scope="module")
def real_schema():
    from ingestion.ddl_parser import DDLParser
    ddl = open("data/docs/digital_evaluation_schema_v10_10.sql", encoding="utf-8").read()
    return DDLParser().parse(ddl)


def test_seed_vocabulary_is_extracted_for_array_columns(real_schema):
    """
    Postgres cannot express "every element of this array is one of N values" as
    a CHECK, so an array column's vocabulary lives only in the seed rows. The
    expected values are read from the DDL, not hardcoded here.
    """
    col = real_schema["workflow_state_transition"].columns["allowed_roles"]
    assert col.allowed_values is None          # no CHECK exists
    assert col.observed_values                 # but seed vocabulary was found
    # Whatever the seed says, an attempt_type value must not be in a role column.
    assert "PRIMARY" not in col.observed_values


def _run_types(schema, sql, alias_map):
    from validation.schema.types import validate_types
    from validation.core.context import ValidationContext
    ctx = ValidationContext(
        sql=sql, ast=sqlglot.parse(sql, dialect="postgres"), schema_map=schema,
        fk_graph=None, tables_used=list(alias_map.values()),
        user_context={}, original_query="",
    )
    ctx.alias_map = alias_map
    ctx.sql_tables = set(alias_map.values())
    result = validate_types(ctx)
    return True if result is None else result.passed


def test_array_vocabulary_mismatch_is_rejected(real_schema):
    # Every supplied value belongs to a different column's vocabulary.
    sql = ("SELECT COUNT(*) FROM workflow_state_transition wst "
           "WHERE wst.allowed_roles @> ARRAY['PRIMARY','REVIEW','REVAL','THIRD']::VARCHAR[]")
    assert not _run_types(real_schema, sql, {"wst": "workflow_state_transition"})


def test_array_vocabulary_accepts_real_and_partially_new_values(real_schema):
    col = real_schema["workflow_state_transition"].columns["allowed_roles"]
    known = sorted(col.observed_values)[0]

    ok = ("SELECT wst.from_state FROM workflow_state_transition wst "
          f"WHERE wst.allowed_roles @> ARRAY['{known}']::VARCHAR[]")
    assert _run_types(real_schema, ok, {"wst": "workflow_state_transition"})

    # A genuinely new value alongside a known one must not be rejected: seed
    # data is a sample, not a constraint.
    partial = ("SELECT wst.from_state FROM workflow_state_transition wst "
               f"WHERE wst.allowed_roles @> ARRAY['{known}','BRAND_NEW_ROLE']::VARCHAR[]")
    assert _run_types(real_schema, partial, {"wst": "workflow_state_transition"})


def test_non_vocabulary_arrays_are_untouched(real_schema):
    sql = "SELECT r.id FROM result r WHERE r.source_attempt_ids @> ARRAY[1,2]"
    assert _run_types(real_schema, sql, {"r": "result"})


# ── FIX-L7a: output coverage is advisory, not a retry gate ───────────────────

def test_l7_output_misses_do_not_gate_retry():
    from validation.semantic.logical_audit import run_logical_audit
    audit = run_logical_audit(
        nl_query=("show the complete question hierarchy, including section names, "
                  "question codes, ltree paths, max marks, and group labels"),
        sql=("SELECT qs.name AS section_name, q.code AS question_code, q.path "
             "FROM question_section qs JOIN question q ON q.section_id = qs.id"),
        intent="lookup", tables_used=[],
    )
    # Still reported...
    assert any(w.startswith("[L7]") for w in audit.warnings)
    assert audit.advisory_misses
    # ...but never in the field that drives audit_driven_retry.
    assert not any(m.startswith("output:") for m in audit.coverage_misses)


def test_real_ddl_still_parses_cleanly(real_schema):
    """The seed-vocabulary pass must not disturb existing parsing."""
    assert len(real_schema) == 63
