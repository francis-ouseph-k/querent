"""
tests/test_schema_types.py
─────────────────────────────
validation/schema/types.py::validate_types — array-column vocabulary
harvested from the DDL's own seed rows, since PostgreSQL cannot express
"every element of this array is one of N values" as a CHECK constraint.

CONSOLIDATED FROM: test_run6_hardening.py. Uses the shared `real_schema`
fixture (conftest.py) since expected values are read from the DDL itself,
never hardcoded here.
"""

from __future__ import annotations

import sqlglot

from conftest import make_ctx


def _run_types(schema, sql, alias_map):
    from validation.schema.types import validate_types
    ctx = make_ctx(sql, schema, tables_used=list(alias_map.values()))
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
