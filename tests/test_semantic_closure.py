"""
tests/test_semantic_closure.py
──────────────────────────────────
validation/semantic/closure_semantics.py::ClosureSemanticsValidator —
rejects a closure-table join used to count/detect children without
constraining `depth`, since every node is its own ancestor at depth 0.

CONSOLIDATED FROM: test_run8_correctness.py ("3. closure semantics").
"""

from __future__ import annotations

from models.schema import ColumnInfo, ForeignKey, TableInventory
from validation.semantic.closure_semantics import ClosureSemanticsValidator

from conftest import make_ctx


def _col(name, data_type="bigint", *, pk=False, nullable=True):
    return ColumnInfo(name=name, data_type=data_type, nullable=nullable, is_pk=pk)


def _table(name, columns, fks=()):
    return TableInventory(
        table_name=name, columns={c.name: c for c in columns}, foreign_keys=list(fks),
    )


def schema():
    return {
        "academic_unit": _table("academic_unit", [
            _col("id", "bigint", pk=True, nullable=False),
            _col("code", "varchar"),
        ]),
        "academic_unit_closure": _table(
            "academic_unit_closure",
            [
                _col("ancestor_id", "bigint", pk=True, nullable=False),
                _col("descendant_id", "bigint", pk=True, nullable=False),
                _col("depth", "integer", nullable=False),
            ],
            fks=[
                ForeignKey("academic_unit_closure", "ancestor_id", "academic_unit", "id"),
                ForeignKey("academic_unit_closure", "descendant_id", "academic_unit", "id"),
            ],
        ),
    }


def ctx_for(sql):
    return make_ctx(sql, schema())


def test_closure_join_without_depth_is_rejected():
    """Q155: every node is its own ancestor, so child_count is never 0."""
    sql = """
        SELECT au.id, COUNT(auc.descendant_id) AS child_count
        FROM academic_unit au
        LEFT JOIN academic_unit_closure auc ON auc.ancestor_id = au.id
        GROUP BY au.id
    """
    result = ClosureSemanticsValidator().run(ctx_for(sql))
    assert not result.passed
    assert "depth" in result.message


def test_closure_join_with_depth_filter_passes():
    sql = """
        SELECT au.id, COUNT(auc.descendant_id) AS child_count
        FROM academic_unit au
        LEFT JOIN academic_unit_closure auc
               ON auc.ancestor_id = au.id AND auc.depth > 0
        GROUP BY au.id
    """
    assert ClosureSemanticsValidator().run(ctx_for(sql)).passed


def test_closure_join_projecting_depth_passes():
    """Q189 asks about the closure itself; the self-row is wanted."""
    sql = """
        SELECT auc.ancestor_id, auc.descendant_id, auc.depth, au.code
        FROM academic_unit_closure auc
        JOIN academic_unit au ON au.id = auc.descendant_id
    """
    assert ClosureSemanticsValidator().run(ctx_for(sql)).passed


def test_closure_scan_without_join_passes():
    """Q4: MAX(depth) over the closure is a question about the closure."""
    sql = "SELECT MAX(ac.depth) AS max_depth FROM academic_unit_closure ac"
    assert ClosureSemanticsValidator().run(ctx_for(sql)).passed
