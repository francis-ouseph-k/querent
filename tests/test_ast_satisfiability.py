"""
tests/test_ast_satisfiability.py
───────────────────────────────────
validation/ast/satisfiability.py::SatisfiabilityValidator — every rule,
merged into one file.

CONSOLIDATED FROM THREE FILES, the clearest case in this refactor of the same
functionality living across different runs:
  * test_run6_hardening.py    — rules 1-2: unique-key GROUP BY HAVING COUNT(*)
                                 pinned to 1; always-true OR of exhaustive
                                 branches.
  * test_run7_hardening.py    — rule 3: contradictory equality conjuncts on
                                 the same column.
  * test_run8_correctness.py  — rules 4-6: constant-true JOIN ON against a
                                 multi-row relation; outer-join nullability
                                 contradiction; a ratio of an expression to
                                 itself.

Each rule set arrived with its own schema fixture, sized to exactly what that
rule reads (unique indexes for 1-2, plain FK/PK shape for 3, the polymorphic
academic_unit hierarchy for 4-6). They are kept as separate local fixtures
rather than forced into one shared schema -- the tables and column shapes are
genuinely different and merging them would risk a collision that changes what
a test is actually asserting.

The "SQL that never parses must never crash a validator" case
(`test_tokenizer_error_is_a_failure_not_a_crash`) lives in
test_semantic_logical_audit.py rather than being duplicated here -- both
files' "unparseable input" tests depend on the same TokenError/ParseError
distinction, but the guard itself only needs asserting once.
"""

from __future__ import annotations

import pytest
import sqlglot

from validation.ast.satisfiability import SatisfiabilityValidator

from conftest import make_ctx


# ═════════════════════════════════════════════════════════════════════════════
# Rules 1-2 (test_run6_hardening.py): unique-key HAVING COUNT(*) > 1; always-
# true OR of exhaustive branches.
# ═════════════════════════════════════════════════════════════════════════════

class _R12Col:
    def __init__(self, is_pk=False):
        self.is_pk = is_pk


class _R12Idx:
    def __init__(self, columns, unique=True, partial=False):
        self.columns, self.is_unique, self.is_partial = columns, unique, partial


class _R12Inv:
    def __init__(self, cols, pk="id", uniq=()):
        self.columns = {c: _R12Col(c == pk) for c in cols}
        self.indexes = [_R12Idx(list(u)) for u in uniq]


_RULES_1_2_SCHEMA = {
    "configuration": _R12Inv(["id", "config_key", "version"]),
    "scanner_device": _R12Inv(["id", "device_name", "is_active"]),
    "result_history": _R12Inv(["id", "result_id"]),
    "enrolment": _R12Inv(["id", "student_id", "course_id"],
                         uniq=[("student_id", "course_id")]),
}


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
def test_rules_1_2(name, should_pass, sql):
    result = SatisfiabilityValidator().run(make_ctx(sql, _RULES_1_2_SCHEMA))
    assert result.passed is should_pass, (name, result.message)


# ═════════════════════════════════════════════════════════════════════════════
# Rule 3 (test_run7_hardening.py): contradictory equality conjuncts.
# ═════════════════════════════════════════════════════════════════════════════

def _r3_table(name: str, columns: dict[str, bool], unique: list[str] | None = None):
    from models.schema import ColumnInfo, IndexInfo, TableInventory
    inv = TableInventory(table_name=name)
    for col, is_pk in columns.items():
        inv.columns[col] = ColumnInfo(name=col, data_type="BIGINT", is_pk=is_pk)
    for col in unique or []:
        inv.indexes.append(
            IndexInfo(name=f"uq_{name}_{col}", table_name=name,
                      columns=[col], is_unique=True)
        )
    return inv


_RULE_3_SCHEMA = {
    "board": _r3_table("board", {"id": True, "status": False, "exam_id": False}),
    "bundle": _r3_table("bundle", {"id": True, "status": False, "bundle_code": False},
                        unique=["bundle_code"]),
}


def _r3_ctx(sql):
    return make_ctx(sql, _RULE_3_SCHEMA, working_sql=sql)


def test_contradictory_equalities_are_rejected():
    sql = "SELECT b.id FROM board b WHERE b.status = 'OPEN' AND b.status = 'CLOSED'"
    result = SatisfiabilityValidator().run(_r3_ctx(sql))
    assert not result.passed
    assert "cannot" in result.message


def test_equality_excluded_by_in_list_is_rejected():
    sql = """
        SELECT b.id FROM board b
        WHERE b.status = 'OPEN' AND b.status IN ('CLOSED', 'SUBMITTED')
    """
    result = SatisfiabilityValidator().run(_r3_ctx(sql))
    assert not result.passed


def test_equality_confirmed_by_in_list_is_accepted():
    """Redundant but satisfiable — not this validator's business to reject."""
    sql = """
        SELECT b.id FROM board b
        WHERE b.status = 'OPEN' AND b.status IN ('OPEN', 'CLOSED')
    """
    assert SatisfiabilityValidator().run(_r3_ctx(sql)).passed


def test_equality_excluded_by_not_in_is_rejected():
    sql = """
        SELECT b.id FROM board b
        WHERE b.status = 'OPEN' AND b.status NOT IN ('OPEN', 'CLOSED')
    """
    assert not SatisfiabilityValidator().run(_r3_ctx(sql)).passed


def test_different_values_under_or_are_accepted():
    """A disjunct may legitimately name a different value."""
    sql = "SELECT b.id FROM board b WHERE b.status = 'OPEN' OR b.status = 'CLOSED'"
    assert SatisfiabilityValidator().run(_r3_ctx(sql)).passed


def test_same_value_on_different_columns_is_accepted():
    sql = """
        SELECT b.id FROM board b JOIN bundle u ON u.id = b.exam_id
        WHERE b.status = 'OPEN' AND u.status = 'COMPLETE'
    """
    assert SatisfiabilityValidator().run(_r3_ctx(sql)).passed


# ═════════════════════════════════════════════════════════════════════════════
# Rules 4-6 (test_run8_correctness.py): constant-true JOIN ON against a
# multi-row relation; outer-join nullability contradiction; self-identical
# ratio.
# ═════════════════════════════════════════════════════════════════════════════

def _r456_col(name, data_type="bigint", *, pk=False, nullable=True):
    from models.schema import ColumnInfo
    return ColumnInfo(name=name, data_type=data_type, nullable=nullable, is_pk=pk)


def _r456_table(name, columns, fks=()):
    from models.schema import TableInventory
    return TableInventory(
        table_name=name, columns={c.name: c for c in columns}, foreign_keys=list(fks),
    )


def _rules_4_6_schema():
    from models.schema import ForeignKey
    return {
        "academic_unit": _r456_table("academic_unit", [
            _r456_col("id", "bigint", pk=True, nullable=False),
            _r456_col("unit_type", "varchar", nullable=False),
        ]),
        "academic_unit_relationship": _r456_table(
            "academic_unit_relationship",
            [
                _r456_col("id", "bigint", pk=True, nullable=False),
                _r456_col("from_unit_id", "bigint", nullable=False),
                _r456_col("to_unit_id", "bigint", nullable=False),
                _r456_col("relationship_type", "varchar", nullable=False),
            ],
            fks=[
                ForeignKey("academic_unit_relationship", "from_unit_id",
                           "academic_unit", "id"),
                ForeignKey("academic_unit_relationship", "to_unit_id",
                           "academic_unit", "id"),
            ],
        ),
        "result_history": _r456_table("result_history", [
            _r456_col("id", "bigint", pk=True, nullable=False),
            _r456_col("result_id", "bigint", nullable=False),
        ]),
        "question": _r456_table("question", [
            _r456_col("id", "bigint", pk=True, nullable=False),
            _r456_col("max_marks", "integer"),
        ]),
    }


def _r456_ctx(sql):
    return make_ctx(sql, _rules_4_6_schema())


def test_join_on_true_is_rejected():
    """Q62: a cross product wearing an ON clause."""
    sql = """
        WITH corrected AS (SELECT DISTINCT rh.result_id FROM result_history rh)
        SELECT q.id
        FROM question q
        JOIN corrected c ON TRUE
    """
    result = SatisfiabilityValidator().run(_r456_ctx(sql))
    assert not result.passed
    assert "cross product" in result.message.lower()


def test_join_on_true_against_single_row_cte_passes():
    """Q168 broadcasts a scalar; that is what ON TRUE is legitimately for."""
    sql = """
        WITH current_year AS (SELECT MAX(q.max_marks) AS m FROM question q)
        SELECT q.id
        FROM question q
        JOIN current_year cy ON TRUE
        WHERE q.max_marks = cy.m
    """
    assert SatisfiabilityValidator().run(_r456_ctx(sql)).passed


def test_outer_join_nullability_contradiction_is_rejected():
    """Q98: the WHERE forces the left side present and absent at once."""
    sql = """
        SELECT au_prog.id
        FROM academic_unit au_prog
        JOIN academic_unit_relationship aur ON aur.from_unit_id = au_prog.id
        RIGHT JOIN academic_unit au_course ON au_course.id = aur.to_unit_id
        WHERE au_prog.unit_type = 'PROGRAM'
          AND aur.id IS NULL
    """
    result = SatisfiabilityValidator().run(_r456_ctx(sql))
    assert not result.passed
    assert "IS NULL" in result.message


def test_plain_anti_join_passes():
    """Q99/Q182: LEFT JOIN + IS NULL is the correct anti-join idiom."""
    sql = """
        SELECT au.id
        FROM academic_unit au
        LEFT JOIN academic_unit_relationship aur ON aur.from_unit_id = au.id
        WHERE au.unit_type = 'COURSE'
          AND aur.id IS NULL
    """
    assert SatisfiabilityValidator().run(_r456_ctx(sql)).passed


def test_self_identical_ratio_is_rejected():
    """Q125: COUNT(*) * 100.0 / NULLIF(COUNT(*), 0) is always 100."""
    sql = """
        SELECT COUNT(*) * 100.0 / NULLIF(COUNT(*), 0) AS leaf_pct
        FROM question q
    """
    result = SatisfiabilityValidator().run(_r456_ctx(sql))
    assert not result.passed
    assert "itself" in result.message


def test_filtered_ratio_passes():
    sql = """
        SELECT COUNT(*) FILTER (WHERE q.max_marks IS NOT NULL) * 100.0
               / NULLIF(COUNT(*), 0) AS leaf_pct
        FROM question q
    """
    assert SatisfiabilityValidator().run(_r456_ctx(sql)).passed
