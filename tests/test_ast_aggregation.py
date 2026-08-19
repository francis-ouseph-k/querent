"""
tests/test_ast_aggregation.py
──────────────────────────────
Everything that exercises validation/ast/aggregation.py: the scope-bounded
aggregate/column detection helpers (`_contains_aggregate_in_scope` and
friends), `_identity_columns` (GROUP BY functional-dependency relaxation),
and `GroupByAlignmentValidator` (Step 7b — SELECT/ORDER BY vs GROUP BY
alignment).

CONSOLIDATED FROM: test_accuracy_regressions.py (aggregate-detection scope
boundary, GROUP BY identity from the DDL) and test_phase1_changes.py
(GroupByAlignmentValidator itself, "Change 3"). Both were testing the same
module from two different angles -- one the detection helpers, one the
validator built on top of them -- and belong together.
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")
import sqlglot.expressions as exp  # noqa: E402

from models.schema import ColumnInfo, TableInventory  # noqa: E402
from validation.ast.aggregation import (  # noqa: E402
    GroupByAlignmentValidator,
    _contains_aggregate_in_scope,
    _contains_column_in_scope,
    _identity_columns,
)

from conftest import make_ctx  # noqa: E402


def _select(sql: str) -> exp.Select:
    node = sqlglot.parse_one(sql, dialect="postgres")
    return node if isinstance(node, exp.Select) else node.find(exp.Select)


# ═════════════════════════════════════════════════════════════════════════════
# _contains_aggregate_in_scope / _contains_column_in_scope
#
# Q155 — aggregate detection must stop at subquery boundaries
# ═════════════════════════════════════════════════════════════════════════════

def test_q155_scalar_subquery_does_not_make_outer_select_aggregate():
    """
    "Count academic units by whether they have children."

    The outer SELECT projects plain columns plus a percentage computed from a
    scalar subquery. The SUM belongs to that subquery; the GROUP BY belongs to
    the CTE. Walking the whole subtree saw "aggregate + non-aggregate + no
    GROUP BY" and failed correct SQL.
    """
    select = _select(
        """
        SELECT unit_type, has_children, unit_count,
               unit_count * 100.0 / NULLIF((SELECT SUM(unit_count)
                                            FROM counted_units), 0) AS pct
        FROM counted_units
        ORDER BY unit_type
        """
    )
    assert any(e.find(exp.AggFunc) for e in select.expressions), "precondition"
    assert not any(_contains_aggregate_in_scope(e) for e in select.expressions)


def test_real_aggregate_without_group_by_is_still_detected():
    select = _select("SELECT dept, COUNT(*) AS n FROM faculty_cache")
    assert any(_contains_aggregate_in_scope(e) for e in select.expressions)
    assert any(_contains_column_in_scope(e) for e in select.expressions)


# ═════════════════════════════════════════════════════════════════════════════
# _identity_columns
#
# Q123 / Q175 / Q20 — GROUP BY identity comes from the DDL, not a name list
# ═════════════════════════════════════════════════════════════════════════════

class _Col:
    def __init__(self, is_pk=False, data_type="VARCHAR", allowed_values=None):
        self.is_pk = is_pk
        self.data_type = data_type
        self.allowed_values = allowed_values
        self.nullable = True
        self.comment = ""
        self.has_jsonb = False


class _Idx:
    def __init__(self, columns, is_unique=False):
        self.columns = columns
        self.is_unique = is_unique


class _Inv:
    def __init__(self, columns, indexes=None):
        self.columns = columns
        self.indexes = indexes or []


def test_q123_non_id_primary_key_counts_as_identity():
    """
    "Count the number of active relationships per relationship type."

    relationship_type_config's PRIMARY KEY is `relationship_type`, a VARCHAR.
    The old hardcoded allowlist ("id", "urn", "code", ...) did not contain it,
    so GROUP BY rtc.relationship_type, rtc.display_name was rejected.
    """
    inv = _Inv({"relationship_type": _Col(is_pk=True), "display_name": _Col()})
    assert _identity_columns(inv) == {"relationship_type"}


def test_single_column_unique_index_counts_as_identity():
    inv = _Inv(
        {"id": _Col(is_pk=True), "bundle_code": _Col(), "status": _Col()},
        indexes=[_Idx(["bundle_code"], is_unique=True), _Idx(["status"])],
    )
    assert _identity_columns(inv) == {"id", "bundle_code"}


def test_composite_unique_index_is_not_an_identity_column():
    """A two-column UNIQUE does not identify a row on either column alone."""
    inv = _Inv({"id": _Col(is_pk=True)}, indexes=[_Idx(["course_id", "exam_id"], True)])
    assert _identity_columns(inv) == {"id"}


# ═════════════════════════════════════════════════════════════════════════════
# GroupByAlignmentValidator (Step 7b)
#
# Now validation/ast/aggregation.py::GroupByAlignmentValidator, reporting
# step="schema" on failure.
# ═════════════════════════════════════════════════════════════════════════════

def _mk_tbl(name: str, cols_with_allowed) -> TableInventory:
    inv = TableInventory(table_name=name)
    for col_name, allowed in cols_with_allowed:
        info = ColumnInfo(name=col_name, data_type="varchar")
        if allowed is not None:
            info.allowed_values = set(allowed)
        inv.columns[col_name] = info
    return inv


SAMPLE_SCHEMA = {
    "board": _mk_tbl("board", [
        ("id", None), ("course_id", None), ("exam_id", None), ("qp_id", None),
        ("status", ["OPEN", "CLOSED"]), ("deadline", None),
    ]),
    "answer_script": _mk_tbl("answer_script", [
        ("id", None), ("urn", None),
        ("lifecycle_status", ["ADMITTED", "ELIGIBLE", "ATTEMPTED", "ABSENT"]),
    ]),
    "evaluation_attempt": _mk_tbl("evaluation_attempt", [
        ("id", None), ("script_id", None),
        ("status", ["ASSIGNED", "IN_PROGRESS", "FROZEN"]),
    ]),
    "question": _mk_tbl("question", [("id", None), ("qp_id", None)]),
    "academic_unit": _mk_tbl("academic_unit", [
        ("id", None), ("parent_id", None), ("name", None),
    ]),
}


class TestGroupByAlignment:

    validator = GroupByAlignmentValidator()

    def _ctx(self, sql):
        return make_ctx(sql, SAMPLE_SCHEMA)

    def test_clean_aggregate_with_groupby_passes(self):
        sql = "SELECT b.status, COUNT(*) FROM board b GROUP BY b.status"
        assert self.validator.run(self._ctx(sql)).passed

    def test_q36_pattern_order_by_uncovered(self):
        """Q36: ORDER BY a.id while GROUP BY does not cover a.id."""
        sql = (
            "SELECT a.id, ea.status, COUNT(*) "
            "FROM answer_script a "
            "JOIN evaluation_attempt ea ON ea.script_id = a.id "
            "GROUP BY ea.status "
            "ORDER BY a.id LIMIT 100"
        )
        result = self.validator.run(self._ctx(sql))
        assert not result.passed
        assert result.step == "schema"

    def test_q155_pattern_case_alias_in_groupby_passes(self):
        """
        Former false positive: a CASE projection is aliased and GROUP BY
        references that alias. PostgreSQL allows it, so the check must too.
        """
        sql = (
            "SELECT CASE WHEN EXISTS ("
            "SELECT 1 FROM academic_unit_closure WHERE ancestor_id = au.id) "
            "THEN 'HAS_CHILDREN' ELSE 'NO_CHILDREN' END AS has_children, "
            "COUNT(au.id) AS count "
            "FROM academic_unit au GROUP BY has_children"
        )
        result = self.validator.run(self._ctx(sql))
        assert result.passed, (result.message or "")[:200]

    def test_window_function_passes(self):
        sql = (
            "SELECT b.status, ROW_NUMBER() OVER (PARTITION BY b.course_id) "
            "FROM board b GROUP BY b.status"
        )
        assert self.validator.run(self._ctx(sql)).passed

    def test_no_groupby_clause_means_no_check(self):
        sql = "SELECT b.status, b.deadline FROM board b"
        assert self.validator.run(self._ctx(sql)).passed

    def test_groupby_id_triggers_fd_relaxation(self):
        """PG's functional-dependency rule: GROUP BY id covers same-table cols."""
        sql = "SELECT b.id, b.status, b.deadline, COUNT(*) FROM board b GROUP BY b.id"
        assert self.validator.run(self._ctx(sql)).passed

    def test_subquery_columns_are_not_flagged(self):
        """A column inside an inner SELECT belongs to that scope, not this one."""
        sql = (
            "SELECT b.status, COUNT(*) FROM board b "
            "WHERE b.qp_id IN (SELECT q.qp_id FROM question q) "
            "GROUP BY b.status"
        )
        assert self.validator.run(self._ctx(sql)).passed
