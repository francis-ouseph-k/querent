"""
tests/test_semantic_zero_suppression.py
───────────────────────────────────────────
validation/semantic/zero_suppression.py::ZeroSuppressionValidator — rejects
AVG (or similar) computed over a COUNT that came from an INNER JOIN, since
parents with zero matching children are silently dropped rather than
contributing a zero.

CONSOLIDATED FROM: test_run7_hardening.py.
"""

from __future__ import annotations

from models.schema import ColumnInfo, IndexInfo, TableInventory
from validation.semantic.zero_suppression import ZeroSuppressionValidator

from conftest import make_ctx


def _table(name: str, columns: dict[str, bool], unique: list[str] | None = None):
    inv = TableInventory(table_name=name)
    for col, is_pk in columns.items():
        inv.columns[col] = ColumnInfo(name=col, data_type="BIGINT", is_pk=is_pk)
    for col in unique or []:
        inv.indexes.append(
            IndexInfo(name=f"uq_{name}_{col}", table_name=name,
                      columns=[col], is_unique=True)
        )
    return inv


SCHEMA = {
    "result": _table("result", {"id": True, "script_id": False, "published_at": False}),
    "result_history": _table("result_history", {"id": True, "result_id": False}),
}


def _ctx(sql: str):
    return make_ctx(sql, SCHEMA, working_sql=sql)


def test_avg_over_inner_joined_count_is_rejected():
    """The Q12 shape: results with zero corrections never reach the average."""
    sql = """
        SELECT AVG(sub.correction_count) AS avg_corrections
        FROM (
          SELECT r.id, COUNT(rh.id) AS correction_count
          FROM   result r
          JOIN   result_history rh ON rh.result_id = r.id
          WHERE  r.published_at IS NOT NULL
          GROUP  BY r.id
        ) sub
    """
    result = ZeroSuppressionValidator().run(_ctx(sql))
    assert not result.passed
    assert "result_history" in result.message
    assert "LEFT JOIN" in result.message


def test_avg_over_left_joined_count_is_accepted():
    """The correct shape — zero-count parents survive with a count of 0."""
    sql = """
        SELECT AVG(sub.correction_count) AS avg_corrections
        FROM (
          SELECT r.id, COUNT(rh.id) AS correction_count
          FROM   result r
          LEFT   JOIN result_history rh ON rh.result_id = r.id
          GROUP  BY r.id
        ) sub
    """
    assert ZeroSuppressionValidator().run(_ctx(sql)).passed


def test_sum_over_inner_joined_count_is_accepted():
    """Absent zero-groups do not change a SUM; only AVG/MIN are affected."""
    sql = """
        SELECT SUM(sub.correction_count)
        FROM (
          SELECT r.id, COUNT(rh.id) AS correction_count
          FROM   result r
          JOIN   result_history rh ON rh.result_id = r.id
          GROUP  BY r.id
        ) sub
    """
    assert ZeroSuppressionValidator().run(_ctx(sql)).passed


def test_non_unique_grouping_key_is_accepted():
    """Without a row-unique key, 'one group per parent' is not established."""
    sql = """
        SELECT AVG(sub.n)
        FROM (
          SELECT r.script_id, COUNT(rh.id) AS n
          FROM   result r
          JOIN   result_history rh ON rh.result_id = r.id
          GROUP  BY r.script_id
        ) sub
    """
    assert ZeroSuppressionValidator().run(_ctx(sql)).passed


def test_cte_form_is_rejected_too():
    """Same defect expressed as a CTE rather than a derived table."""
    sql = """
        WITH counts AS (
          SELECT r.id, COUNT(rh.id) AS n
          FROM   result r
          JOIN   result_history rh ON rh.result_id = r.id
          GROUP  BY r.id
        )
        SELECT AVG(counts.n) FROM counts
    """
    assert not ZeroSuppressionValidator().run(_ctx(sql)).passed
