"""
validation/ast/aggregation.py
─────────────────────────────
Aggregate-shape validators.

    GroupByAlignmentValidator  (pipeline step 8, reports `groupby`)
        Every non-aggregated column in the SELECT list must appear in GROUP BY.
        Catches the classic "column must appear in the GROUP BY clause" error
        before it reaches PostgreSQL.

    AggregationValidator       (pipeline step 12, reports `aggregation`)
        Rejects illegal aggregate structure — chiefly nested aggregates such as
        AVG(COUNT(*)) — while correctly allowing an aggregate inside a window
        function. Helpers _node_contains_aggregate / _node_inside_window do the
        AST distinction so a legitimate `COUNT(*) OVER (...)` is not flagged.
"""

import sqlglot.expressions as exp
from typing import Any

from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

class AggregationValidator(BaseValidationStep):
    name = "AggregationValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """Step 1.5: Reject nested aggregate functions and missing GROUP BY."""
        sql = ctx.working_sql or ctx.sql
        if ctx.ast is None:
            return ValidationResult(passed=True, step="aggregation", sql=sql)
            
        for stmt in ctx.ast:
            # Check 1: nested aggregates (existing) — e.g. AVG(COUNT(*))
            for agg in stmt.find_all(exp.AggFunc):
                for child in agg.expressions:
                    if child.find(exp.AggFunc):
                        return ValidationResult(
                            passed=False, step="aggregation",
                            message="Nested aggregate functions are not allowed in PostgreSQL (e.g., AVG(COUNT(*))). Use a subquery or adjust your GROUP BY.",
                            sql=sql
                        )

            # Check 2: SELECT mixes aggregate and non-aggregate columns without GROUP BY.
            select_node = stmt if isinstance(stmt, exp.Select) else stmt.find(exp.Select)
            if select_node and select_node.expressions:
                expressions = select_node.expressions
                has_agg = any(
                    expr.find(exp.AggFunc) for expr in expressions
                )
                if has_agg:
                    # Check if there are non-aggregate columns
                    has_non_agg = False
                    for expr in expressions:
                        # Unwrap aliases: SELECT COUNT(*) AS c is an Alias wrapping AggFunc
                        inner = expr.this if isinstance(expr, exp.Alias) else expr
                        if inner.find(exp.AggFunc):
                            continue  # this expression is aggregate — skip
                        # exp.Column or exp.Alias wrapping a Column → non-aggregate
                        if isinstance(inner, exp.Column) or (
                            isinstance(expr, exp.Alias) and isinstance(inner, exp.Column)
                        ):
                            has_non_agg = True
                            break
                        # Catch expressions that contain a Column but not inside an AggFunc
                        if inner.find(exp.Column) and not inner.find(exp.AggFunc):
                            has_non_agg = True
                            break

                    group_by = select_node.args.get("group")
                    if has_non_agg and not group_by:
                        return ValidationResult(
                            passed=False, step="aggregation",
                            message=(
                                "SELECT mixes aggregate and non-aggregate columns "
                                "without GROUP BY. Add GROUP BY for every "
                                "non-aggregated column."
                            ),
                            sql=sql,
                        )
                        
                    # Check 3: Enforce grouping by entity IDs (per-table alias)
                    if group_by:
                        grouped_cols = list(group_by.find_all(exp.Column))
                        grouped_by_table = {}
                        
                        # 1. Group all columns found in the GROUP BY clause by their table alias.
                        for c in grouped_cols:
                            if c.name:
                                tbl = (c.table or "").lower()
                                if tbl not in grouped_by_table:
                                    grouped_by_table[tbl] = []
                                grouped_by_table[tbl].append(c.name.lower())
                                
                        # 2. Check each table independently.
                        for tbl, cols in grouped_by_table.items():
                            has_id = any(c in ("id", "board_id", "script_id", "course_id", "exam_id", "qp_id", "department_id", "urn", "email", "code") for c in cols)
                            has_desc = any(c in ("name", "title", "display_name", "description") for c in cols)
                            
                            if has_desc and not has_id:
                                return ValidationResult(
                                    passed=False, step="aggregation",
                                    message=(
                                        f"GROUP BY uses a descriptive column from table/alias '{tbl}' "
                                        f"without including a primary key or unique identifier (like id, urn, code) for that table. "
                                        f"Names may not be unique. Always include the entity's unique ID column in the GROUP BY clause."
                                    ),
                                    sql=sql,
                                )

        return ValidationResult(passed=True, step="aggregation", sql=sql)


_AGGREGATE_FUNC_NAMES = frozenset({
    "count", "sum", "avg", "min", "max",
    "string_agg", "array_agg", "json_agg", "jsonb_agg",
    "bool_and", "bool_or", "every",
    "variance", "var_pop", "var_samp",
    "stddev", "stddev_pop", "stddev_samp",
    "covar_pop", "covar_samp",
    "corr", "regr_slope", "regr_intercept",
    "percentile_cont", "percentile_disc", "mode",
})

def _node_contains_aggregate(node: exp.Expression) -> bool:
    """True if node contains an aggregate function call (recursively)."""
    if node is None:
        return False
    for sub in node.walk():
        if isinstance(sub, exp.AggFunc):
            return True
        if isinstance(sub, exp.Anonymous):
            fn = ""
            if isinstance(sub.this, str):
                fn = sub.this.lower()
            elif hasattr(sub.this, "name"):
                fn = (sub.this.name or "").lower()
            if fn in _AGGREGATE_FUNC_NAMES:
                return True
    return False

def _node_inside_window(node: exp.Expression) -> bool:
    """True if node is nested inside an OVER (...) window expression."""
    p = node.parent
    while p is not None:
        if isinstance(p, exp.Window):
            return True
        p = p.parent
    return False


class GroupByAlignmentValidator(BaseValidationStep):
    name = "GroupByAlignmentValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 7b: Verify SELECT non-aggregate columns appear in GROUP BY.
        """
        sql = ctx.working_sql or ctx.sql
        if ctx.ast is None:
            return ValidationResult(passed=True, step="groupby", sql=sql)
            
        for ast in ctx.ast:
            outer = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
            if outer is None:
                continue

            group = outer.args.get("group")
            if group is None:
                continue

            gb_exprs = group.expressions or []

            # Functional-dependency relaxation: skip when any GROUP BY key is "id".
            has_id = False
            for g in gb_exprs:
                for col in g.find_all(exp.Column):
                    if (col.name or "").lower() == "id":
                        has_id = True
                        break
            if has_id:
                continue

            gb_canonical: set[str] = set()
            for g in gb_exprs:
                try:
                    gb_canonical.add(g.sql(dialect="postgres").lower())
                except Exception:
                    continue

            # SELECT-projection aliases that GROUP BY references.
            aliased_gb_targets: set[str] = set()
            for sel in outer.expressions:
                if isinstance(sel, exp.Alias) and sel.alias:
                    alias_lo = sel.alias.lower()
                    if alias_lo in gb_canonical:
                        aliased_gb_targets.add(alias_lo)

            projection_aliases: set[str] = {
                sel.alias.lower() for sel in outer.expressions
                if isinstance(sel, exp.Alias) and sel.alias
            }

            inner_select_node_ids: set[int] = set()
            for inner_sel in outer.find_all(exp.Select):
                if inner_sel is outer:
                    continue
                for n in inner_sel.walk():
                    inner_select_node_ids.add(id(n))

            def _covered(col: exp.Column) -> bool:
                """True if col is legally referenced under PG's GROUP BY rules."""
                if id(col) in inner_select_node_ids:
                    return True
                if _node_inside_window(col):
                    return True
                p = col.parent
                while p is not None and p is not outer:
                    if isinstance(p, exp.AggFunc):
                        return True
                    p = p.parent
                try:
                    col_sql = col.sql(dialect="postgres").lower()
                except Exception:
                    return True
                if col_sql in gb_canonical:
                    return True
                cn = (col.name or "").lower()
                if not (col.table or "") and cn in projection_aliases:
                    return True
                if cn in {k.split('.')[-1].strip() for k in gb_canonical}:
                    return True
                return False

            bad_projections: list[str] = []
            for sel in outer.expressions:
                if isinstance(sel, exp.Alias) and sel.alias and sel.alias.lower() in aliased_gb_targets:
                    continue

                real = sel.this if isinstance(sel, exp.Alias) else sel
                try:
                    real_sql = real.sql(dialect="postgres").lower()
                except Exception:
                    continue

                if real_sql in gb_canonical:
                    continue

                if _node_contains_aggregate(real):
                    continue

                if not real.find(exp.Column):
                    continue

                uncovered: list[str] = []
                for col in real.find_all(exp.Column):
                    if _covered(col):
                        continue
                    try:
                        col_sql = col.sql(dialect="postgres").lower()
                    except Exception:
                        continue
                    uncovered.append(col_sql)

                if uncovered:
                    bad_projections.append(
                        f"`{real.sql(dialect='postgres')[:60]}` "
                        f"(uncovered column(s): {', '.join(sorted(set(uncovered))[:3])})"
                    )

            order_clause = outer.args.get("order")
            if order_clause is not None:
                for col in order_clause.find_all(exp.Column):
                    if _covered(col):
                        continue
                    try:
                        col_sql = col.sql(dialect="postgres").lower()
                    except Exception:
                        continue
                    bad_projections.append(
                        f"ORDER BY `{col_sql}` "
                        f"(uncovered: not in GROUP BY, not in an aggregate)"
                    )
                    break 

            if bad_projections:
                msg = (
                    "SQL has a GROUP BY clause but the following SELECT/ORDER BY "
                    "expression(s) are non-aggregate AND not in GROUP BY: "
                    + "; ".join(bad_projections[:3])
                    + ". Either add them to GROUP BY, or wrap them in an "
                      "aggregate function (e.g. MAX, MIN, STRING_AGG)."
                )
                logger.warning(
                    component="sql_validator",
                    event="groupby_misalignment",
                    bad=bad_projections[:3],
                )
                return ValidationResult(
                    passed=False,
                    step="schema",
                    message=msg,
                    sql=sql,
                )

        return ValidationResult(passed=True, step="groupby", sql=sql)
