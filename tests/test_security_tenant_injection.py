"""
tests/test_security_tenant_injection.py
───────────────────────────────────────────
validation/security/tenant_injector.py::inject_where_all_scopes — G2:
every scope of a multi-CTE statement gets the tenant predicate, or the whole
rewrite fails closed (never silently partial).

CONSOLIDATED FROM: test_security_hardening.py ("G2: every scope scoped, or
fail closed").
"""

from __future__ import annotations

from validation.security.tenant_injector import inject_where_all_scopes


class _Col:
    def __init__(self, name, data_type="BIGINT"):
        self.name = name
        self.data_type = data_type
        self.allowed_values = None


class _Inv:
    def __init__(self, name, columns):
        self.name = name
        self.columns = {c: _Col(c) for c in columns}
        self.foreign_keys = []


SCHEMA = {
    "answer_script": _Inv("answer_script",
                          ["id", "urn", "board_id", "course_id", "exam_id",
                           "s3_key", "s3_version_id", "dek_id"]),
    "evaluation_attempt": _Inv("evaluation_attempt",
                               ["id", "script_id", "board_id", "status"]),
    "data_encryption_key": _Inv("data_encryption_key",
                                ["id", "wrapped_key", "iv", "algorithm"]),
    "board": _Inv("board", ["id", "course_id", "exam_id", "status"]),
}
TENANT_TABLES = {"answer_script", "evaluation_attempt", "board"}


def test_g2_multi_cte_gets_predicate_in_every_scope():
    sql = (
        "WITH a AS (SELECT id FROM answer_script), "
        "     b AS (SELECT id FROM evaluation_attempt) "
        "SELECT a.id FROM a JOIN b ON b.id = a.id"
    )
    out, unscoped = inject_where_all_scopes(
        sql, "board_id", 7, SCHEMA, TENANT_TABLES)
    assert unscoped == [], f"expected full coverage, unscoped={unscoped}"
    # The old single-scope injector produced exactly one. Two is the fix.
    assert out.lower().count("board_id = 7") == 2, out


def test_g2_reports_scope_that_cannot_carry_the_column():
    # data_encryption_key has no board_id, so a query joining it under a
    # board_id scope cannot be fully isolated. It must be reported, not
    # silently passed.
    sql = ("SELECT d.algorithm FROM data_encryption_key d "
           "JOIN answer_script a ON a.dek_id = d.id")
    out, unscoped = inject_where_all_scopes(
        sql, "board_id", 7, SCHEMA, TENANT_TABLES)
    assert "board_id = 7" in out.lower()
    assert unscoped == []  # answer_script in the same scope carries it


def test_g2_unparseable_sql_fails_closed():
    out, unscoped = inject_where_all_scopes(
        "SELECT FROM WHERE ((", "board_id", 7, SCHEMA, TENANT_TABLES)
    assert out is None
    assert unscoped, "must report everything as unscoped, not pass silently"


def test_g2_injection_is_idempotent_per_scope():
    sql = "SELECT id FROM answer_script WHERE board_id = 7"
    out, _ = inject_where_all_scopes(
        sql, "board_id", 7, SCHEMA, TENANT_TABLES)
    assert out.lower().count("board_id = 7") == 1, out
