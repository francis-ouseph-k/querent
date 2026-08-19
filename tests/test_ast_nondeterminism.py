"""
tests/test_ast_nondeterminism.py
──────────────────────────────────
validation/ast/nondeterminism.py::NondeterministicSelectionValidator —
rejects `LIMIT 1` with no `ORDER BY` used to pin an arbitrary row into a
predicate (directly, or laundered through a CTE), while leaving the ordinary
"latest/highest" idiom, EXISTS, and a plain presentation cap alone.

CONSOLIDATED FROM: test_run7_hardening.py.
"""

from __future__ import annotations

from models.schema import ColumnInfo, IndexInfo, TableInventory
from validation.ast.nondeterminism import NondeterministicSelectionValidator

from conftest import make_ctx


def _table(name: str, columns: dict[str, bool], unique: list[str] | None = None):
    """columns: {column_name: is_pk}. unique: single-column UNIQUE indexes."""
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
    "board": _table("board", {"id": True, "status": False, "exam_id": False}),
    "question_paper": _table("question_paper", {"id": True, "title": False}),
}


def _ctx(sql: str, question: str = ""):
    return make_ctx(sql, SCHEMA, original_query=question, working_sql=sql)


def test_scalar_subquery_limit_without_order_is_rejected():
    """The Q67 shape: an arbitrary board pinned into a join predicate."""
    sql = """
        SELECT r.id
        FROM   result r
        WHERE  r.script_id = (SELECT b.id FROM board b LIMIT 1)
    """
    result = NondeterministicSelectionValidator().run(_ctx(sql))
    assert not result.passed
    assert "arbitrary row" in result.message


def test_cte_limit_without_order_is_rejected():
    """The Q29 shape: the arbitrary pick laundered through a CTE."""
    sql = """
        WITH one_paper AS (
            SELECT qp.id FROM question_paper qp
            WHERE qp.title ILIKE '%Data Structures%' LIMIT 1
        )
        SELECT r.id FROM result r JOIN one_paper p ON p.id = r.script_id
    """
    result = NondeterministicSelectionValidator().run(_ctx(sql))
    assert not result.passed
    assert "one_paper" in result.message


def test_scalar_subquery_with_order_by_is_accepted():
    """`ORDER BY ... LIMIT 1` is the ordinary latest/highest idiom."""
    sql = """
        SELECT r.id
        FROM   result r
        WHERE  r.script_id = (
                 SELECT b.id FROM board b ORDER BY b.id DESC LIMIT 1
               )
    """
    assert NondeterministicSelectionValidator().run(_ctx(sql)).passed


def test_exists_with_limit_is_accepted():
    """EXISTS is a boolean over the whole set; which row satisfies it is moot."""
    sql = """
        SELECT r.id FROM result r
        WHERE EXISTS (SELECT 1 FROM board b WHERE b.id = r.script_id LIMIT 1)
    """
    assert NondeterministicSelectionValidator().run(_ctx(sql)).passed


def test_top_level_limit_is_accepted():
    """A presentation cap on the final result set is not a value choice."""
    sql = "SELECT r.id FROM result r LIMIT 200"
    assert NondeterministicSelectionValidator().run(_ctx(sql)).passed
