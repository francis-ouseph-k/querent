"""
tests/test_schema_columns.py
───────────────────────────────
validation/schema/columns.py — two related concerns:

  1. `_local_scope`: resolving what an alias in a FROM/JOIN clause refers to,
     including DERIVED scopes (`unnest(...) AS sa(attempt_id)`, `LATERAL`)
     that are not a real table or a CTE.
  2. `validate_columns`: whether a `alias.column` reference is legal, INCLUDING
     into a CTE's own projection -- a reference into a CTE used to be skipped
     outright, so it went straight to EXPLAIN, where PostgreSQL names neither
     the CTE nor what it actually projects.

CONSOLIDATED FROM: test_accuracy_regressions.py (derived-scope resolution,
Q38) and test_run8_correctness.py (CTE output-column validation, item 8).
Both exercise the same alias-resolution machinery in schema/columns.py.
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")
import sqlglot.expressions as exp  # noqa: E402

from validation.schema.columns import _local_scope  # noqa: E402

from conftest import make_ctx  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# _local_scope — Q38: set-returning-function aliases are declared, not
# hallucinated
# ═════════════════════════════════════════════════════════════════════════════

def test_q38_unnest_alias_is_registered_as_a_derived_scope():
    """
    "For each published result, show ... the list of source attempt types ..."

    FROM unnest(rd.source_attempt_ids) AS sa(attempt_id) declares `sa`. The
    scope builder only understood exp.Table and exp.Subquery, so every
    reference through `sa` was reported as an undeclared alias.
    """
    stmt = sqlglot.parse_one(
        """
        SELECT (SELECT jsonb_agg(ei.attempt_type)
                FROM unnest(rd.source_attempt_ids) AS sa(attempt_id)
                JOIN evaluator_info ei ON ei.attempt_id = sa.attempt_id) AS types
        FROM result_data rd
        """,
        dialect="postgres",
    )
    derived_aliases = set()
    for select in stmt.find_all(exp.Select):
        _alias_map, _tables, derived = _local_scope(select)
        derived_aliases |= derived
    assert "sa" in derived_aliases


def test_lateral_alias_is_also_registered():
    stmt = sqlglot.parse_one(
        "SELECT x.n FROM board b, LATERAL (SELECT COUNT(*) AS n FROM result) x",
        dialect="postgres",
    )
    derived = set()
    for select in stmt.find_all(exp.Select):
        derived |= _local_scope(select)[2]
    assert "x" in derived


def test_plain_table_aliases_still_resolve_to_their_table():
    node = sqlglot.parse_one(
        "SELECT b.id FROM board b JOIN academic_unit au ON au.id = b.course_id",
        dialect="postgres",
    )
    select = node if isinstance(node, exp.Select) else node.find(exp.Select)
    alias_map, tables, derived = _local_scope(select)
    assert alias_map["b"] == "board"
    assert alias_map["au"] == "academic_unit"
    assert tables == {"board", "academic_unit"}
    assert not derived


# ═════════════════════════════════════════════════════════════════════════════
# validate_columns — CTE output columns are checked, not skipped
#
# Q54 and Q177 of batch run 20260819 both went through this hole to EXPLAIN,
# where PostgreSQL names neither the CTE nor its actual columns -- the
# correction loop had nothing concrete to act on. Schema-step recovery in that
# run was 40%, against 86% for semantic rejections.
# ═════════════════════════════════════════════════════════════════════════════

def _schema():
    from models.schema import ColumnInfo, ForeignKey, TableInventory

    def col(name, data_type="bigint", *, pk=False, nullable=True):
        return ColumnInfo(name=name, data_type=data_type, nullable=nullable, is_pk=pk)

    def table(name, columns, fks=()):
        return TableInventory(
            table_name=name, columns={c.name: c for c in columns}, foreign_keys=list(fks),
        )

    return {
        "board": table("board", [
            col("id", "bigint", pk=True, nullable=False),
            col("course_id", "bigint", nullable=False),
            col("status", "varchar"),
            col("created_at", "timestamptz"),
        ]),
        "academic_unit_relationship": table(
            "academic_unit_relationship",
            [
                col("id", "bigint", pk=True, nullable=False),
                col("from_unit_id", "bigint", nullable=False),
                col("to_unit_id", "bigint", nullable=False),
            ],
        ),
        "question": table("question", [
            col("id", "bigint", pk=True, nullable=False),
            col("max_marks", "integer"),
        ]),
    }


def ctx_for(sql):
    return make_ctx(sql, _schema())


def _columns_result(sql: str):
    from validation.schema.columns import validate_columns
    from validation.schema.tables import validate_tables
    ctx = ctx_for(sql)
    validate_tables(ctx)
    return validate_columns(ctx)


def _is_cte_column_error(result) -> bool:
    return result is not None and not result.passed and \
        "is a CTE and does not project" in (result.message or "")


def test_column_absent_from_cte_projection_is_rejected():
    """Q54: the CTE projects id and created_at; the outer query wants more."""
    sql = """
        WITH last_month_boards AS (
            SELECT id, created_at FROM board WHERE created_at >= CURRENT_DATE
        )
        SELECT b.id, b.course_id
        FROM last_month_boards b
    """
    result = _columns_result(sql)
    assert _is_cte_column_error(result)
    # The message has to be actionable where PostgreSQL's is not: it names the
    # CTE and lists what the CTE really projects.
    assert "last_month_boards" in result.message
    assert "created_at" in result.message and "id" in result.message


def test_renamed_cte_column_referenced_by_original_name_is_rejected():
    """Q177: the CTE aliased from_unit_id to something else."""
    sql = """
        WITH prerequisite_courses AS (
            SELECT aur.from_unit_id AS prerequisite_course_id
            FROM academic_unit_relationship aur
        )
        SELECT aur.from_unit_id FROM prerequisite_courses aur
    """
    result = _columns_result(sql)
    assert _is_cte_column_error(result)
    assert "prerequisite_course_id" in result.message


def test_valid_cte_reference_passes():
    sql = """
        WITH c AS (SELECT b.id AS bid, b.status FROM board b)
        SELECT c.bid, c.status FROM c
    """
    assert not _is_cte_column_error(_columns_result(sql))


def test_aliased_cte_reference_resolves_to_the_cte():
    """`FROM c z` — the alias must be translated back to the CTE's own name."""
    sql = "WITH c AS (SELECT b.id AS bid FROM board b) SELECT z.bid FROM c z"
    assert not _is_cte_column_error(_columns_result(sql))


def test_select_star_cte_is_not_enumerable_and_stays_silent():
    """Expanding a star needs the full FROM closure; guessing would misfire."""
    sql = "WITH c AS (SELECT * FROM board) SELECT c.course_id FROM c"
    assert not _is_cte_column_error(_columns_result(sql))


def test_unaliased_expression_cte_stays_silent():
    """PostgreSQL derives the output name; this code must not predict it."""
    sql = "WITH c AS (SELECT COUNT(*) FROM board) SELECT c.count FROM c"
    assert not _is_cte_column_error(_columns_result(sql))


def test_explicit_cte_column_list_is_honoured():
    """`WITH c(x, y) AS ...` states the output exactly."""
    good = "WITH c(x, y) AS (SELECT id, status FROM board) SELECT c.x FROM c"
    assert not _is_cte_column_error(_columns_result(good))
    bad = "WITH c(x, y) AS (SELECT id, status FROM board) SELECT c.id FROM c"
    assert _is_cte_column_error(_columns_result(bad))


def test_set_operation_cte_uses_leftmost_branch():
    sql = (
        "WITH c AS (SELECT id FROM board UNION SELECT id FROM question) "
        "SELECT c.id FROM c"
    )
    assert not _is_cte_column_error(_columns_result(sql))


def test_derived_table_alias_is_not_treated_as_a_cte():
    """A subquery in FROM validates its own columns in its own scope."""
    sql = (
        "WITH b AS (SELECT id FROM board) "
        "SELECT x.status FROM (SELECT status FROM board) x"
    )
    assert not _is_cte_column_error(_columns_result(sql))
