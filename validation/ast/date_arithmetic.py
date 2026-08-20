"""
validation/ast/date_arithmetic.py
─────────────────────────────────
DateArithmeticValidator — rejects `EXTRACT(EPOCH FROM <date> - <date>)`.

In PostgreSQL, subtracting two DATE values yields an INTEGER number of days,
not an INTERVAL. `EXTRACT(EPOCH FROM ...)` has no signature accepting an
integer, so the statement dies at plan time:

    function pg_catalog.extract(unknown, integer) does not exist

EXPLAIN already catches it, which is why Q17 failed rather than returning
something wrong. The reason this deserves its own step is the CORRECTION path,
not the detection path: Postgres names no column and suggests no fix, so the
error the retry loop feeds back to the model is "add explicit type casts",
which the model satisfies by adding a cast that does not help. Q17 burned its
whole retry budget and never recovered.

This fires from the DDL's own column types, names the offending expression,
and states the rewrite — `AVG(a - b)`, with the `/ 86400` dropped, because the
subtraction is already in days.

Deliberately silent on `timestamp - timestamp`, which yields a genuine
INTERVAL and for which `EXTRACT(EPOCH FROM ...)` is correct and common in this
corpus, and on any operand whose type cannot be resolved. COALESCE, NULLIF,
parentheses and casts are resolved through, since the observed defect was
wrapped in a COALESCE.
"""

from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp

from ..core.base import BaseValidationStep
from ..core.context import ValidationContext
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

# CURRENT_DATE is DATE; CURRENT_TIMESTAMP / NOW() are not.
_DATE_FUNCTIONS = {"current_date"}


def _alias_map(stmt: exp.Expression) -> dict[str, str]:
    cte_names = {(c.alias or "").lower() for c in stmt.find_all(exp.CTE)}
    out: dict[str, str] = {}
    for tbl in stmt.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if not name or name in cte_names:
            continue
        out[(tbl.alias or name).lower()] = name
        out.setdefault(name, name)
    return out


def _is_date_type(data_type: str) -> bool:
    text = (data_type or "").strip().lower()
    # 'date' exactly — 'timestamp', 'timestamptz' and 'timestamp with time
    # zone' all contain no bare 'date' token but are excluded explicitly for
    # readability of intent.
    return text == "date"


def _operand_is_date(
    node: exp.Expression, alias_map: dict[str, str], schema_map: dict, _depth: int = 0,
) -> bool | None:
    """
    True / False / None (unknown) for "is this expression of type DATE".

    None is returned for anything this module cannot type with certainty, and
    the caller then stays silent.
    """
    if _depth > 6 or node is None:
        return None

    if isinstance(node, exp.Paren):
        return _operand_is_date(node.this, alias_map, schema_map, _depth + 1)

    if isinstance(node, exp.Cast):
        return _is_date_type(node.to.sql(dialect="postgres"))

    if isinstance(node, (exp.Coalesce, exp.Nullif)):
        arguments = [node.this] + list(node.expressions or [])
        verdicts = [
            _operand_is_date(arg, alias_map, schema_map, _depth + 1) for arg in arguments
        ]
        verdicts = [v for v in verdicts if v is not None]
        if not verdicts:
            return None
        # COALESCE resolves to one type; if every typed branch agrees, that is
        # the type. Disagreement means the expression would not have compiled.
        return all(verdicts) if any(verdicts) else False

    if isinstance(node, exp.CurrentDate):
        return True

    if isinstance(node, exp.Anonymous) and (node.name or "").lower() in _DATE_FUNCTIONS:
        return True

    if isinstance(node, exp.Column):
        table = alias_map.get((node.table or "").lower())
        inventory = schema_map.get(table) if table else None
        if inventory is None:
            return None
        column = (getattr(inventory, "columns", {}) or {}).get((node.name or "").lower())
        if column is None:
            return None
        return _is_date_type(getattr(column, "data_type", ""))

    return None


def subtraction_operand_kind(sql: str, schema_map: dict) -> str:
    """
    Classify the date/time subtractions in `sql` by the type of their operands.

    Returns:
      "date"      every subtraction this module can type has DATE on both
                  sides, so `a - b` already yields an INTEGER number of days
      "timestamp" at least one subtraction has a TIMESTAMP/TIMESTAMPTZ operand,
                  so it yields an INTERVAL and EXTRACT(EPOCH ...) is correct
      "unknown"   nothing could be typed with certainty

    WHY THIS IS PUBLIC

    Check 8 in validation/semantic/semantic_checks.py demands
    `EXTRACT(EPOCH FROM (end - start)) / 86400` whenever a question asks for an
    "average duration". That instruction is right for TIMESTAMP columns and a
    type error for DATE columns -- and this module rejects exactly that error.
    Q17 of run 20260819 sat in the resulting deadlock: the semantic rule
    demanded EXTRACT(EPOCH ...) over `script_hold.hold_start_date` /
    `hold_end_date`, both DATE, while DateArithmeticValidator correctly
    refused it. No SQL could satisfy both, so the question was unanswerable by
    construction and burned its whole retry budget.

    Rather than duplicate the type resolution (and let the two copies drift),
    the semantic rule asks this module the same question this module already
    answers for itself.
    """
    if not sql or not schema_map:
        return "unknown"
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return "unknown"

    saw_date = False
    for stmt in statements:
        if stmt is None:
            continue
        alias_map = _alias_map(stmt)
        for sub in stmt.find_all(exp.Sub):
            left = _operand_is_date(sub.this, alias_map, schema_map)
            right = _operand_is_date(sub.expression, alias_map, schema_map)
            if left is None or right is None:
                continue
            if left and right:
                saw_date = True
            else:
                # A typed operand that is not DATE is a timestamp-like value;
                # one such subtraction is enough to make EXTRACT(EPOCH ...)
                # the correct instruction for this statement.
                return "timestamp"
    return "date" if saw_date else "unknown"


class DateArithmeticValidator(BaseValidationStep):
    """Rejects EXTRACT(EPOCH FROM ...) applied to a DATE difference."""

    name = "DateArithmeticValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        sql = ctx.working_sql or ctx.sql
        if not ctx.ast or not ctx.schema_map:
            return ValidationResult(passed=True, step="schema", sql=sql)

        try:
            for stmt in ctx.ast:
                if stmt is None:
                    continue
                alias_map = _alias_map(stmt)

                for extract in stmt.find_all(exp.Extract):
                    unit = (extract.this.name if extract.this is not None else "") or ""
                    if unit.lower() not in {"epoch"}:
                        continue

                    target = extract.expression
                    while isinstance(target, exp.Paren):
                        target = target.this
                    if not isinstance(target, exp.Sub):
                        continue

                    left = _operand_is_date(target.this, alias_map, ctx.schema_map)
                    right = _operand_is_date(target.expression, alias_map, ctx.schema_map)
                    if not (left and right):
                        continue

                    expression_sql = target.sql(dialect="postgres")
                    logger.warning(
                        component="sql_validator",
                        event="epoch_over_date_difference",
                        expression=expression_sql[:160],
                        sql_preview=sql[:120],
                    )
                    return ValidationResult(
                        passed=False, step="schema",
                        message=(
                            f"`EXTRACT(EPOCH FROM {expression_sql})` is a type "
                            f"error. Both operands are DATE, and DATE - DATE "
                            f"yields an INTEGER number of days in PostgreSQL, "
                            f"not an INTERVAL — EXTRACT has no signature for an "
                            f"integer, so the statement fails at plan time. The "
                            f"subtraction is ALREADY in days: write "
                            f"`{expression_sql}` on its own and drop both the "
                            f"EXTRACT and the `/ 86400`. Only use "
                            f"EXTRACT(EPOCH FROM ...) when both operands are "
                            f"TIMESTAMP or TIMESTAMPTZ."
                        ),
                        sql=sql,
                    )
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="date_arithmetic_check_error",
                error=f"{type(exc).__name__}: {exc}",
                note="check skipped due to AST error",
            )

        return ValidationResult(passed=True, step="schema", sql=sql)
