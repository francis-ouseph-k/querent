"""
tests/test_ast_date_arithmetic.py
───────────────────────────────────
validation/ast/date_arithmetic.py::DateArithmeticValidator — rejects
`EXTRACT(EPOCH FROM <date> - <date>)`, a PostgreSQL type error (DATE - DATE
is an integer day count, not an INTERVAL), while leaving the identical
pattern over TIMESTAMP/TIMESTAMPTZ operands alone.

CONSOLIDATED FROM: test_run8_correctness.py ("5. date arithmetic").
"""

from __future__ import annotations

from models.schema import ColumnInfo, TableInventory
from validation.ast.date_arithmetic import DateArithmeticValidator

from conftest import make_ctx


def _col(name, data_type="bigint", *, pk=False, nullable=True):
    return ColumnInfo(name=name, data_type=data_type, nullable=nullable, is_pk=pk)


def _table(name, columns):
    return TableInventory(table_name=name, columns={c.name: c for c in columns})


SCHEMA = {
    "script_hold": _table("script_hold", [
        _col("id", "bigint", pk=True, nullable=False),
        _col("hold_start_date", "date", nullable=False),
        _col("hold_end_date", "date"),
        _col("is_active", "boolean", nullable=False),
    ]),
    "evaluation_attempt": _table("evaluation_attempt", [
        _col("id", "bigint", pk=True, nullable=False),
        _col("started_at", "timestamptz"),
        _col("frozen_at", "timestamptz"),
        _col("status", "varchar"),
    ]),
}


def _ctx(sql):
    return make_ctx(sql, SCHEMA)


def test_epoch_over_date_difference_is_rejected():
    """Q17: DATE - DATE is an integer, and EXTRACT has no integer signature."""
    sql = """
        SELECT AVG(EXTRACT(EPOCH FROM (
                   COALESCE(sh.hold_end_date, CURRENT_DATE) - sh.hold_start_date
               )) / 86400.0) AS avg_days
        FROM script_hold sh
    """
    result = DateArithmeticValidator().run(_ctx(sql))
    assert not result.passed
    assert "86400" in result.message


def test_epoch_over_timestamp_difference_passes():
    """Q58 and friends: TIMESTAMPTZ - TIMESTAMPTZ really is an INTERVAL."""
    sql = """
        SELECT AVG(EXTRACT(EPOCH FROM (ea.frozen_at - ea.started_at)) / 3600) AS hrs
        FROM evaluation_attempt ea
    """
    assert DateArithmeticValidator().run(_ctx(sql)).passed


def test_plain_date_difference_passes():
    sql = """
        SELECT AVG(COALESCE(sh.hold_end_date, CURRENT_DATE) - sh.hold_start_date)
        FROM script_hold sh
    """
    assert DateArithmeticValidator().run(_ctx(sql)).passed


def test_unknown_operand_type_is_silent():
    sql = "SELECT EXTRACT(EPOCH FROM (x.a - x.b)) FROM unknown_table x"
    assert DateArithmeticValidator().run(_ctx(sql)).passed


# ── subtraction_operand_kind: the public helper Check 8 consults ─────────────
#
# Q17 of run 20260819 deadlocked because validation/semantic/semantic_checks.py
# demanded EXTRACT(EPOCH FROM ...) for any "average duration" question while
# this module rejected that construction over DATE columns. The semantic rule
# now asks this helper rather than keeping a second copy of the type logic.

def test_operand_kind_date():
    from validation.ast.date_arithmetic import subtraction_operand_kind
    sql = ("SELECT AVG(COALESCE(sh.hold_end_date, CURRENT_DATE) - sh.hold_start_date) "
           "FROM script_hold sh")
    assert subtraction_operand_kind(sql, SCHEMA) == "date"


def test_operand_kind_timestamp():
    from validation.ast.date_arithmetic import subtraction_operand_kind
    sql = "SELECT AVG(ea.frozen_at - ea.started_at) FROM evaluation_attempt ea"
    assert subtraction_operand_kind(sql, SCHEMA) == "timestamp"


def test_operand_kind_unknown_for_untypeable_operands():
    from validation.ast.date_arithmetic import subtraction_operand_kind
    assert subtraction_operand_kind("SELECT x.a - x.b FROM unknown_table x", SCHEMA) \
        == "unknown"


def test_operand_kind_unknown_without_schema():
    from validation.ast.date_arithmetic import subtraction_operand_kind
    sql = "SELECT sh.hold_end_date - sh.hold_start_date FROM script_hold sh"
    assert subtraction_operand_kind(sql, {}) == "unknown"


def test_operand_kind_never_raises_on_unparseable_sql():
    from validation.ast.date_arithmetic import subtraction_operand_kind
    assert subtraction_operand_kind('SELECT list.",', SCHEMA) == "unknown"


def test_mixed_subtractions_report_timestamp():
    """One timestamp operand is enough to make EXTRACT(EPOCH ...) correct."""
    from validation.ast.date_arithmetic import subtraction_operand_kind
    sql = ("SELECT sh.hold_end_date - sh.hold_start_date, "
           "       ea.frozen_at - ea.started_at "
           "FROM script_hold sh, evaluation_attempt ea")
    assert subtraction_operand_kind(sql, SCHEMA) == "timestamp"
