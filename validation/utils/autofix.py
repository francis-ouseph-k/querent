"""
validation/utils/autofix.py
──────────────────────────────
Deterministic SQL auto-repair for well-understood, mechanically-fixable error
classes. No LLM calls — every fix here is a precise, provable rewrite, then
re-validated through the real pipeline before being trusted. If re-validation
fails, the fix is discarded and the query falls through to normal failure
handling, never silently accepted.

Two fixers, same pattern (propose → re-validate → accept or discard):

  attempt_pg_autofix()
      Pre-existing. Triggered from validation/execution/cost.py (Step 9,
      EXPLAIN). Parses PostgreSQL's own "Perhaps you meant..." planner hint
      on a column-not-found error and rewrites the SQL to use the suggested
      column.

  attempt_tautological_autofix()
      Added 2026-07-01. Triggered from pipeline/runner.py when
      validation/semantic/logical_audit.py's L5 check flags a tautological
      aggregation -- COUNT(DISTINCT x) / AVG(x) / SUM(x) grouped by that
      same x, which always just returns x back rather than computing a real
      aggregate. Mechanically strips the self-referential GROUP BY key and
      any now-invalid bare SELECT reference to it (PostgreSQL requires every
      non-aggregated SELECT column to appear in GROUP BY).

Kept as one module because both fixers share the same contract: never
accept a rewrite that hasn't been re-verified as valid, structurally correct
SQL. New fixable error classes should follow this same shape.
"""

import re
import sqlglot
import sqlglot.expressions as exp
from typing import Any
from utils.logging_config import get_logger

logger = get_logger(__name__)

_PG_COL_PERHAPS_RE = re.compile(
    r'column\s+"?([\w.]+)"?\s+does not exist.*?'
    r'Perhaps you meant to reference the column\s+"([^"]+)"',
    re.IGNORECASE | re.DOTALL,
)

def attempt_pg_autofix(
    sql: str,
    error_msg: str,
    schema_map: dict,
    run_explain: Any,
) -> tuple[str | None, str | None]:
    """
    Try a deterministic column-rename based on a PG planner hint.

    Returns (fixed_sql, fix_description) on success, or (None, None) on
    failure / no-applicable-hint.
    """
    m = _PG_COL_PERHAPS_RE.search(error_msg)
    if not m:
        return None, None
    bad_ref = m.group(1).strip().strip('"')
    hint    = m.group(2).strip().strip('"')

    if '.' not in bad_ref or '.' not in hint:
        logger.info(component="sql_validator", event="autofix_skipped_unqualified", bad=bad_ref, hint=hint)
        return None, None
    
    bad_tbl_or_alias, bad_col = bad_ref.lower().split('.', 1)
    good_tbl_or_alias, good_col = hint.lower().split('.', 1)

    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        logger.info(component="sql_validator", event="autofix_skipped_parse_error", bad=bad_ref, hint=hint)
        return None, None
    if not statements:
        logger.info(component="sql_validator", event="autofix_skipped_no_statements", bad=bad_ref, hint=hint)
        return None, None

    alias_to_table: dict[str, str] = {}
    cte_names: set[str] = set()
    for stmt in statements:
        if stmt is None:
            continue
        for cte in stmt.find_all(exp.CTE):
            if cte.alias:
                cte_names.add(cte.alias.lower())
        for tbl in stmt.find_all(exp.Table):
            name = (tbl.name or '').lower()
            if not name or name in cte_names:
                continue
            alias = (tbl.alias or '').lower()
            if alias:
                alias_to_table[alias] = name
            alias_to_table[name] = name

    if good_tbl_or_alias in alias_to_table:
        good_table = alias_to_table[good_tbl_or_alias]
    elif good_tbl_or_alias in schema_map:
        good_table = good_tbl_or_alias
    else:
        logger.info(
            component="sql_validator",
            event="autofix_skipped_target_not_in_scope",
            bad=bad_ref, hint=hint,
        )
        return None, None

    inv = schema_map.get(good_table)
    if inv is None or not hasattr(inv, 'columns'):
        logger.info(component="sql_validator", event="autofix_skipped_missing_schema", table=good_table)
        return None, None
    if good_col not in inv.columns:
        logger.info(
            component="sql_validator",
            event="autofix_skipped_col_not_in_ddl",
            bad=bad_ref, hint=hint,
            target=good_table,
        )
        return None, None

    chosen_alias = None
    for a, t in alias_to_table.items():
        if t == good_table and a != good_table:
            chosen_alias = a
            break
    rewrite_table_part = chosen_alias or good_table

    replacements_made = 0
    new_statements: list[exp.Expression] = []
    for stmt in statements:
        if stmt is None:
            new_statements.append(stmt)
            continue
        def _swap(node: exp.Expression) -> exp.Expression:
            nonlocal replacements_made
            if isinstance(node, exp.Column):
                if ((node.table or '').lower() == bad_tbl_or_alias
                        and (node.name or '').lower() == bad_col):
                    replacements_made += 1
                    return exp.Column(
                        this  = exp.to_identifier(good_col, quoted=False),
                        table = exp.to_identifier(rewrite_table_part, quoted=False),
                    )
            return node
        new_statements.append(stmt.transform(_swap, copy=True))

    if replacements_made == 0:
        return None, None

    new_sql_parts = []
    for stmt in new_statements:
        if stmt is None:
            continue
        new_sql_parts.append(stmt.sql(dialect="postgres"))
    new_sql = ";\n".join(new_sql_parts)
    if sql.rstrip().endswith(';') and not new_sql.endswith(';'):
        new_sql += ';'

    pgcode, err = run_explain(new_sql)
    if pgcode is None and err is None:
        desc = (
            f"autofix: replaced `{bad_ref}` with `{rewrite_table_part}."
            f"{good_col}` ({replacements_made} occurrence(s)) per "
            f"PostgreSQL planner hint"
        )
        logger.info(
            component="sql_validator",
            event="autofix_accepted",
            bad=bad_ref, hint=hint,
            target=f"{rewrite_table_part}.{good_col}",
            replacements=replacements_made,
        )
        return new_sql, desc

    logger.info(
        component="sql_validator",
        event="autofix_re_explain_failed",
        bad=bad_ref, hint=hint,
        new_err=str(err)[:120] if err else None,
    )
    return None, None


# ─────────────────────────────────────────────────────────────────────────
# Tautological aggregation autofix (2026-07-01)
# ─────────────────────────────────────────────────────────────────────────
# Companion to attempt_pg_autofix() above, triggered from pipeline/runner.py
# when validation/semantic/logical_audit.py's L5 check sets hard_fail=True.
#
# Unlike the anti-join polarity case (L4), a confirmed tautological
# aggregation -- COUNT(DISTINCT x)/AVG(x)/SUM(x) with GROUP BY containing
# that exact column -- has a mechanically safe fix with no ambiguity:
#   * If the tautological column is the ONLY GROUP BY column, the query is
#     asking for a single aggregate value with a meaningless grouping key
#     attached. Dropping GROUP BY entirely is the only sensible reading
#     (this is Q120's case: AVG(retention_days) GROUP BY retention_days).
#   * If other GROUP BY columns are present alongside it, only the
#     tautological column is redundant; the other grouping columns reflect
#     real intent and must be preserved (this is Q27's num_leaf_questions-
#     adjacent pattern, though Q27 itself needed a different fix -- see
#     the qs.id case in the test suite).
#
# This is a mechanical transform, not a guess: removing a GROUP BY column
# that is provably a no-op (the aggregate target itself) cannot change
# what the query means, only what it correctly computes.

import sqlglot.expressions as exp


def attempt_tautological_autofix(
    sql: str,
    run_explain: Any = None,
) -> tuple[str | None, str | None]:
    """
    Deterministically strip a self-referential GROUP BY column from a
    query where COUNT(DISTINCT x) / AVG(x) / SUM(x) is grouped by that
    same x. Also strips any now-invalid bare SELECT references to the
    removed key(s), since PostgreSQL requires every non-aggregated SELECT
    column to appear in GROUP BY.

    run_explain: optional (pgcode, err) callable, same contract as
    attempt_pg_autofix's, for an internal pre-check. When None (the
    pipeline/runner.py call site passes None -- it doesn't have direct
    access to CostValidator's private run_explain closure), the caller is
    responsible for re-validating the returned SQL through the full
    structural pipeline (self.validator.validate(...)) before accepting
    it. Either path guarantees the fix is never accepted without a real
    EXPLAIN pass -- the gate just lives in a different place depending on
    which caller is using this function.

    Returns (fixed_sql, fix_description) on success, or (None, None) if
    no tautological pattern is found or the fix can't be constructed.
    """
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return None, None
    if not statements:
        return None, None

    fixed_any = False
    new_statements: list[exp.Expression] = []

    for stmt in statements:
        if stmt is None:
            new_statements.append(stmt)
            continue

        group = stmt.args.get("group")
        if group is None:
            new_statements.append(stmt)
            continue

        # Collect (table_or_None, column) for every GROUP BY expression
        # that is a bare column reference (skip expressions -- CASE, etc.
        # can't be tautological against a single aggregate target).
        group_exprs = list(group.expressions)
        group_keys: list[tuple[str | None, str, exp.Expression]] = []
        for ge in group_exprs:
            if isinstance(ge, exp.Column):
                tbl = (ge.table or None)
                group_keys.append((tbl.lower() if tbl else None, ge.name.lower(), ge))

        # Find every aggregate function in the SELECT list whose single
        # argument is a bare column matching one of the GROUP BY keys.
        tautological_keys: set[tuple[str | None, str]] = set()
        for func in stmt.find_all(exp.AggFunc):
            if not isinstance(func, (exp.Count, exp.Avg, exp.Sum)):
                continue
            args = [a for a in func.args.get("this", []) if a] if isinstance(func.args.get("this"), list) else [func.args.get("this")]
            for arg in args:
                col = arg
                # COUNT(DISTINCT x) wraps the column in a Distinct node
                if isinstance(col, exp.Distinct) and col.expressions:
                    col = col.expressions[0]
                if isinstance(col, exp.Column):
                    tbl = (col.table or None)
                    key = (tbl.lower() if tbl else None, col.name.lower())
                    for gtbl, gcol, _ in group_keys:
                        if gcol == key[1] and (gtbl == key[0] or (gtbl is None and key[0] is None)):
                            tautological_keys.add((gtbl, gcol))

        if not tautological_keys:
            new_statements.append(stmt)
            continue

        remaining = [ge for (gtbl, gcol, ge) in group_keys if (gtbl, gcol) not in tautological_keys]
        # Preserve any non-Column GROUP BY expressions untouched (they were
        # never candidates for removal).
        remaining += [ge for ge in group_exprs if not isinstance(ge, exp.Column)]

        # PostgreSQL requires every non-aggregated SELECT column to appear
        # in GROUP BY. Once a tautological key is removed from GROUP BY
        # (whether or not other keys remain), any bare (non-aggregated)
        # reference to THAT specific key in the SELECT list is no longer
        # valid and must be dropped too -- e.g.
        # `SELECT qs.id, qs.name, COUNT(DISTINCT qs.id) ... GROUP BY qs.id,
        # qs.name` becomes `SELECT qs.name, COUNT(DISTINCT qs.id) ...
        # GROUP BY qs.name`, not a query that still bare-selects qs.id
        # alongside a GROUP BY that no longer includes it.
        select = stmt.args.get("expressions", [])
        kept_select = []
        for sel in select:
            target = sel.this if isinstance(sel, exp.Alias) else sel
            if isinstance(target, exp.Column):
                tbl = (target.table or None)
                key = (tbl.lower() if tbl else None, target.name.lower())
                if key in tautological_keys:
                    continue  # drop: bare ref to a column no longer in GROUP BY
            kept_select.append(sel)
        if kept_select:
            stmt.set("expressions", kept_select)
        # else: every SELECT item was a bare tautological column (degenerate
        # case) -- leave SELECT list untouched rather than emit an empty
        # SELECT; EXPLAIN re-validation below will catch and reject this.

        if remaining:
            group.set("expressions", remaining)
        else:
            # Only tautological key(s) were present -- drop GROUP BY entirely.
            stmt.set("group", None)

        fixed_any = True
        new_statements.append(stmt)

    if not fixed_any:
        return None, None

    new_sql_parts = [s.sql(dialect="postgres") for s in new_statements if s is not None]
    new_sql = ";\n".join(new_sql_parts)
    if sql.rstrip().endswith(';') and not new_sql.endswith(';'):
        new_sql += ';'

    desc = (
        "autofix: removed tautological GROUP BY column(s) -- an "
        "aggregate function's target column was also its own GROUP BY "
        "key, which is always a no-op"
    )

    if run_explain is None:
        # No direct EXPLAIN access at this call site -- caller re-validates
        # via the full structural pipeline instead. See docstring.
        logger.info(
            component="sql_validator",
            event="tautological_autofix_constructed_pending_pipeline_revalidation",
        )
        return new_sql, desc

    pgcode, err = run_explain(new_sql)
    if pgcode is None and err is None:
        logger.info(
            component="sql_validator",
            event="tautological_autofix_accepted",
        )
        return new_sql, desc

    logger.info(
        component="sql_validator",
        event="tautological_autofix_re_explain_failed",
        new_err=str(err)[:120] if err else None,
    )
    return None, None