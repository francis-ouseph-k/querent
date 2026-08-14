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
                # FIX-W1 (false positive, Q51 of batch run 20260813).
                # sqlglot models LAG/LEAD/ROW_NUMBER/RANK as exp.AggFunc
                # subclasses wrapped in exp.Window, so the old test read a pure
                # window query as "has aggregate" and then demanded a GROUP BY
                # that PostgreSQL does not require. Only a BARE aggregate (one
                # not inside an OVER (...) clause) forces a GROUP BY.
                has_agg = any(
                    _contains_bare_aggregate_in_scope(expr) for expr in expressions
                )
                if has_agg:
                    # Check if there are non-aggregate columns
                    has_non_agg = False
                    for expr in expressions:
                        # Unwrap aliases: SELECT COUNT(*) AS c is an Alias wrapping AggFunc
                        inner = expr.this if isinstance(expr, exp.Alias) else expr
                        # FIX-W1: a window expression is neither an aggregate
                        # that satisfies GROUP BY nor a bare column that
                        # requires it — PostgreSQL evaluates it after grouping.
                        # Skip it in both directions.
                        if _contains_window_in_scope(inner):
                            continue
                        if _contains_bare_aggregate_in_scope(inner):
                            continue  # this expression is aggregate — skip
                        # exp.Column or exp.Alias wrapping a Column → non-aggregate
                        if isinstance(inner, exp.Column) or (
                            isinstance(expr, exp.Alias) and isinstance(inner, exp.Column)
                        ):
                            has_non_agg = True
                            break
                        # Catch expressions that contain a Column but not inside an AggFunc
                        if (_contains_column_in_scope(inner)
                                and not _contains_aggregate_in_scope(inner)):
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
                        res = _check_group_by_identity(group_by, ctx, sql)
                        if res:
                            return res

        return ValidationResult(passed=True, step="aggregation", sql=sql)


# ─────────────────────────────────────────────────────────────────────────────
# Scope-bounded AST walkers
# ─────────────────────────────────────────────────────────────────────────────
# FIX-A1 (false positive, Q155 of batch run 20260812).
#
# `expr.find(exp.AggFunc)` walks the WHOLE subtree, including any nested
# subquery. A scalar subquery in the SELECT list therefore made the OUTER
# projection look aggregate:
#
#     SELECT unit_type, has_children, unit_count,
#            unit_count * 100.0 / NULLIF((SELECT SUM(unit_count)
#                                         FROM counted_units), 0) AS pct
#     FROM counted_units
#     ORDER BY unit_type, has_children DESC
#
# The SUM belongs to the inner SELECT and has nothing to do with the outer
# one, but Check 2 saw "aggregate + non-aggregate + no GROUP BY" and failed a
# query whose GROUP BY lived (correctly) inside the CTE. Aggregate and column
# detection must stop at subquery boundaries, exactly as SQL scoping does.

_SCOPE_BOUNDARIES = (exp.Subquery, exp.Select)


def _walk_in_scope(node: exp.Expression):
    """Yield nodes under `node`, not descending into a nested SELECT/Subquery."""
    if node is None:
        return
    yield node
    for child in node.args.values():
        children = child if isinstance(child, list) else [child]
        for c in children:
            if not isinstance(c, exp.Expression):
                continue
            if isinstance(c, _SCOPE_BOUNDARIES):
                continue  # separate scope — not this SELECT's business
            yield from _walk_in_scope(c)


def _contains_aggregate_in_scope(node: exp.Expression) -> bool:
    """True if an aggregate call appears in THIS scope (not a nested SELECT)."""
    return any(isinstance(n, exp.AggFunc) for n in _walk_in_scope(node))


def _contains_column_in_scope(node: exp.Expression) -> bool:
    """True if a column reference appears in THIS scope (not a nested SELECT)."""
    return any(isinstance(n, exp.Column) for n in _walk_in_scope(node))


def _contains_window_in_scope(node: exp.Expression) -> bool:
    """True if an OVER (...) window expression appears in THIS scope."""
    return any(isinstance(n, exp.Window) for n in _walk_in_scope(node))


def _contains_bare_aggregate_in_scope(node: exp.Expression) -> bool:
    """
    True if an aggregate call appears in THIS scope OUTSIDE any window frame.

    FIX-W1. `_contains_aggregate_in_scope` answers "is there an AggFunc node
    here", which is True for LAG(x) OVER (...) because sqlglot derives the
    window functions from exp.AggFunc. Only a bare aggregate — SUM(x), not
    SUM(x) OVER (...) — collapses rows and therefore forces a GROUP BY.
    """
    for n in _walk_in_scope(node):
        if isinstance(n, exp.AggFunc) and not _node_inside_window(n):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# GROUP BY identity check — DDL-derived
# ─────────────────────────────────────────────────────────────────────────────
# FIX-A2 (false positives, Q20 / Q123 / Q175 of batch run 20260812).
#
# The previous implementation asked "does this alias's GROUP BY column list
# contain one of these ten hardcoded names?" That is not a schema question, it
# is a naming-convention guess, and it was wrong three ways:
#
#   Q123  GROUP BY rtc.relationship_type, rtc.display_name
#         `relationship_type` IS the PRIMARY KEY of relationship_type_config
#         (DDL v10.10 L832) — it just isn't spelled "id".
#
#   Q175  GROUP BY uar.user_id, au.display_name, uar.auxiliary_role
#         Valid PostgreSQL: au.display_name is itself grouped. The check
#         evaluated alias `au` in isolation, ignoring that the descriptive
#         column appears in the GROUP BY list.
#
#   Q20   GROUP BY b.id, au.name  — same shape, and this one is the expensive
#         case: attempt 1 was CORRECT (all LEFT JOINs, board id grouped). The
#         rejection drove four retries, and the final attempt had dropped
#         au.name AND converted every LEFT JOIN to INNER JOIN — a wrong query
#         produced by "fixing" a right one.
#
# Rules now applied, in order:
#   1. A descriptive column that is itself in the GROUP BY is legal. This is
#      what PostgreSQL enforces, and it is the end of the matter for
#      executability. The remaining concern is purely "could two rows share a
#      name and be wrongly merged", so this is now ADVISORY.
#   2. Identity is resolved from the DDL: the table's real PRIMARY KEY, or any
#      single-column UNIQUE index. No name guessing.
#   3. If the table cannot be resolved (derived table, CTE, unknown alias) the
#      check does not fire. Unknown is not the same as wrong.

_DESCRIPTIVE_COLUMNS = frozenset({"name", "title", "display_name", "description"})


def _identity_columns(inv) -> set[str]:
    """PK columns plus single-column UNIQUE index columns, from the DDL."""
    identity: set[str] = set()
    cols = getattr(inv, "columns", None) or {}
    for col_name, col in cols.items():
        if getattr(col, "is_pk", False):
            identity.add(col_name.lower())
    for idx in (getattr(inv, "indexes", None) or []):
        idx_cols = getattr(idx, "columns", None) or []
        if getattr(idx, "is_unique", False) and len(idx_cols) == 1:
            identity.add(str(idx_cols[0]).lower())
    return identity


def _check_group_by_identity(group_by, ctx: ValidationContext, sql: str):
    """
    Advisory-by-default identity check on GROUP BY (see FIX-A2 above).

    Returns a failing ValidationResult only when a descriptive column is
    grouped for a table whose real identity column is absent AND the
    descriptive column is not itself in the GROUP BY — which, given rule 1,
    cannot currently happen. Kept as a function so the rule has one home and
    can be re-armed deliberately rather than by accident.
    """
    grouped_by_table: dict[str, list[str]] = {}
    for c in group_by.find_all(exp.Column):
        if not c.name:
            continue
        tbl = (c.table or "").lower()
        grouped_by_table.setdefault(tbl, []).append(c.name.lower())

    grouped_everywhere = {n for names in grouped_by_table.values() for n in names}

    for tbl, cols in grouped_by_table.items():
        desc = [c for c in cols if c in _DESCRIPTIVE_COLUMNS]
        if not desc:
            continue

        # Rule 3: resolve alias → real table via the DDL inventory. Bail out
        # quietly on derived tables, CTEs and anything unresolvable.
        real_table = (ctx.alias_map or {}).get(tbl, tbl)
        if tbl in (ctx.cte_names or set()) or real_table in (ctx.cte_names or set()):
            continue
        inv = (ctx.schema_map or {}).get(real_table)
        if inv is None:
            continue

        identity = _identity_columns(inv)
        if not identity:
            continue
        if identity & set(cols):
            continue  # Rule 2: a real PK/UNIQUE column is grouped

        # Rule 1: the descriptive column is grouped, so the SQL is legal and
        # executable. Log the merge risk; do not fail the query.
        if all(d in grouped_everywhere for d in desc):
            logger.info(
                component="sql_validator",
                event="aggregation_group_by_identity_advisory",
                table=real_table,
                alias=tbl,
                descriptive=desc,
                identity=sorted(identity),
                note="descriptive column is itself in GROUP BY — legal SQL; "
                     "flagged only as a potential name-collision merge risk",
            )
            continue

        return ValidationResult(
            passed=False, step="aggregation",
            message=(
                f"GROUP BY uses a descriptive column from table/alias '{tbl}' "
                f"({', '.join(desc)}) without including an identifying column "
                f"for {real_table}. Add one of: {', '.join(sorted(identity))}."
            ),
            sql=sql,
        )
    return None


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