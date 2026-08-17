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

# ─────────────────────────────────────────────────────────────────────────────
# attempt_near_miss_column_autofix()
# ─────────────────────────────────────────────────────────────────────────────
# FIX-N1 (batch run 20260812, Q69).
#
# attempt_pg_autofix() only fires on PostgreSQL's "Perhaps you meant..." hint,
# which is produced at EXPLAIN time (Step 9). The schema validator runs earlier
# (Step 4) and rejects hallucinated columns before EXPLAIN is ever reached, so
# there is no planner hint to parse and no fixer runs at all:
#
#   Validation failed (schema): Hallucinated column(s):
#     revaluation_extension_request.revalidation_request_id
#
# The real column is `revaluation_request_id` — Damerau-Levenshtein distance 2
# from what the model wrote, and the ONLY column on that table within that
# distance. An LLM retry was spent on a two-character transcription slip.
#
# Conditions are deliberately strict, because a wrong rename is worse than a
# clean failure:
#   * the bad name must not exist on the table (obviously),
#   * exactly ONE candidate may lie within the distance threshold — ties are
#     ambiguous and are refused,
#   * the threshold scales with name length (short names are not fuzzy-matched:
#     `id` vs `qp` is distance 2 but means something entirely different),
#   * the rewrite is re-validated by the caller before it is trusted, exactly
#     like the other two fixers in this module.

_SCHEMA_HALLUCINATED_COL_RE = re.compile(
    r"Hallucinated column\(s\):\s*([^.]+)\.(\w+)",
    re.IGNORECASE,
)


def _damerau_levenshtein(a: str, b: str) -> int:
    """Optimal string alignment distance (handles adjacent transpositions)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[la][lb]


def _max_distance_for(name: str) -> int:
    """Edit budget by identifier length. Short names get no fuzzy matching."""
    n = len(name)
    if n < 6:
        return 0
    if n < 12:
        return 1
    return 2


def attempt_near_miss_column_autofix(
    sql: str,
    error_msg: str,
    schema_map: dict,
) -> tuple[str | None, str | None]:
    """
    Repair a single near-miss column name reported by the schema validator.

    Returns (fixed_sql, fix_description), or (None, None) when no unambiguous
    single-candidate correction exists. The caller MUST re-validate the result.
    """
    m = _SCHEMA_HALLUCINATED_COL_RE.search(error_msg or "")
    if not m:
        return None, None

    table = m.group(1).strip().strip('"').lower()
    bad_col = m.group(2).strip().strip('"').lower()

    inv = schema_map.get(table)
    if inv is None or not hasattr(inv, "columns"):
        return None, None

    real_cols = {c.lower() for c in inv.columns}
    if bad_col in real_cols:
        return None, None

    budget = _max_distance_for(bad_col)
    if budget == 0:
        return None, None

    candidates = [c for c in real_cols if _damerau_levenshtein(bad_col, c) <= budget]
    if len(candidates) != 1:
        logger.info(
            component="sql_validator",
            event="near_miss_autofix_skipped",
            table=table,
            bad_column=bad_col,
            candidate_count=len(candidates),
            note="no unambiguous single candidate within edit budget",
        )
        return None, None

    good_col = candidates[0]

    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return None, None
    if not statements:
        return None, None

    # Rename only columns that resolve to the offending table, so an identically
    # named column on another table in the same query is left alone.
    alias_to_table: dict[str, str] = {}
    for stmt in statements:
        if stmt is None:
            continue
        cte_names = {c.alias.lower() for c in stmt.find_all(exp.CTE) if c.alias}
        for tbl in stmt.find_all(exp.Table):
            name = (tbl.name or "").lower()
            if not name or name in cte_names:
                continue
            alias = (tbl.alias or "").lower()
            if alias:
                alias_to_table[alias] = name
            alias_to_table[name] = name

    changed = 0
    for stmt in statements:
        if stmt is None:
            continue
        for col in stmt.find_all(exp.Column):
            if (col.name or "").lower() != bad_col:
                continue
            qualifier = (col.table or "").lower()
            if qualifier and alias_to_table.get(qualifier) != table:
                continue
            if not qualifier and len(alias_to_table) > 1:
                continue  # unqualified and ambiguous — leave it for the LLM
            col.set("this", exp.to_identifier(good_col))
            changed += 1

    if not changed:
        return None, None

    fixed_sql = ";\n".join(s.sql(dialect="postgres") for s in statements if s)
    desc = (
        f"renamed {table}.{bad_col} -> {table}.{good_col} "
        f"(edit distance {_damerau_levenshtein(bad_col, good_col)}, "
        f"{changed} reference(s))"
    )
    logger.info(
        component="sql_validator",
        event="near_miss_autofix_applied",
        table=table,
        bad_column=bad_col,
        good_column=good_col,
        references=changed,
    )
    return fixed_sql, desc


# ─────────────────────────────────────────────────────────────────────────
# Reserved-word alias autofix (2026-08-14)
# ─────────────────────────────────────────────────────────────────────────
# SyntaxValidator (validation/ast/syntax.py) has detected `as` (or another
# SQL reserved word) used as a table alias since before this hardening pass,
# but only to PRODUCE an error message -- the retry loop then burns a full
# model round-trip re-generating the entire query from scratch to fix one
# token. That has cost 5+ occurrences across every benchmark run to date,
# ~15-50s of wall time each, and occasionally does not even get fixed (the
# model repeats the same mistake on retry).
#
# The bug is single-token and its shape is always the same: the reserved
# word is unparseable as an alias BECAUSE it collides with the keyword's own
# grammar, but the fix -- give it a different name -- is unambiguous once
# the declaration site is located. No model call is needed to rename a
# variable.

_RESERVED_ALIAS_TARGET_RE = re.compile(
    r'''(?ix)
        \b(from|join)\s+
        (?:(?P<schema>[a-z_][a-z0-9_]*)\.)?(?P<table>[a-z_][a-z0-9_]*)\s+
        (?:as\s+)?(?P<alias>as|in|group|order|table|select|where|having|
                            limit|offset|union|with|all|and|any|case|
                            check|exists|using)\b
    '''
)


def _mask_strings_and_comments(sql: str) -> str:
    """
    Blank out string/comment CONTENTS while preserving length, so a regex
    match position found in the masked text maps 1:1 onto the original SQL.
    (The existing detection regex in syntax.py replaces string literals with
    a fixed `''`, which shifts every later offset -- fine for a yes/no
    detection, not safe for a position-based rewrite.)
    """
    out = list(sql)

    def _blank(m: "re.Match") -> str:
        s, e = m.span()
        for i in range(s, e):
            if out[i] not in ("'", '"'):
                out[i] = " "
        return sql[s:e]

    for pat in (r"--[^\n]*", r"/\*.*?\*/", r"'(?:[^']|'')*'"):
        for m in re.finditer(pat, sql, flags=re.DOTALL):
            _blank(m)
    return "".join(out)


def _sub_alias_refs_outside_literals(
    sql: str, masked: str, bad_alias: str, candidate: str,
    start: int = 0, end: int | None = None,
) -> str:
    """
    Replace `<bad_alias>.` with `<candidate>.` ONLY where the match falls
    outside a string literal or comment, within [start, end).

    A plain `re.sub` on the raw SQL cannot tell a table qualifier from the
    same characters inside a literal, and silently rewrites the DATA:

        WHERE as.note = 'see as.txt for detail'
          ->  WHERE ans_a.note = 'see ans_a.txt for detail'   <- corrupted

    `masked` has literal/comment CONTENTS blanked but the SAME LENGTH as
    `sql`, so a match found in `masked` maps 1:1 onto `sql` by offset.
    Applying replacements right-to-left keeps earlier offsets valid.
    """
    if end is None:
        end = len(sql)
    pattern = re.compile(rf"\b{re.escape(bad_alias)}\.", re.IGNORECASE)
    spans = [m.span() for m in pattern.finditer(masked, start, end)]
    out = sql
    for m_start, m_end in reversed(spans):
        out = out[:m_start] + candidate + "." + out[m_end:]
    return out


def attempt_reserved_alias_autofix(sql: str) -> tuple[str | None, str | None]:
    """
    Rename a reserved-word table alias (`FROM answer_script AS as`, or the
    implicit form `FROM answer_script as`) to a safe, unique identifier, and
    fix up every `<alias>.<column>` reference to match.

    Returns (fixed_sql, fix_description) on success, or (None, None) if no
    reserved-word alias declaration was found, or if the rewrite still fails
    to parse (never trust a text rewrite without re-verifying it).
    """
    masked = _mask_strings_and_comments(sql)
    m = _RESERVED_ALIAS_TARGET_RE.search(masked)
    if not m:
        return None, None

    bad_alias = m.group("alias").lower()
    table     = m.group("table")

    # A short, non-reserved, deterministic replacement: first 3 letters of
    # the table name + "_a" reads naturally (answer_script -> ans_a) and is
    # exceedingly unlikely to collide with an existing alias. If it does
    # collide, append a counter until it doesn't.
    base = re.sub(r"[^a-z0-9_]", "", table.lower())[:3] or "t"
    candidate = f"{base}_a"
    existing_aliases = {
        a.lower() for a in re.findall(r"\b(?:as\s+)?([a-z_][a-z0-9_]*)\s*(?=[,\n)]|$)",
                                       masked, flags=re.IGNORECASE)
    }
    n = 2
    while candidate in existing_aliases:
        candidate = f"{base}_a{n}"
        n += 1

    # 1) The declaration site: replace exactly the matched alias TOKEN (not
    #    the table name, not the AS keyword before it) using the position
    #    found in the masked text, which is 1:1 with the original.
    decl_start, decl_end = m.span("alias")
    rewritten = sql[:decl_start] + candidate + sql[decl_end:]

    # 2) `<alias>.` used as a table qualifier. Substituted position-wise
    #    against the MASKED text so a string literal that happens to contain
    #    `as.` (e.g. `WHERE note = 'see as.txt'`) is not rewritten as if it
    #    were SQL. The declaration rename above shifted offsets, so re-mask
    #    the rewritten text rather than reusing the original mask.
    rewritten = _sub_alias_refs_outside_literals(
        rewritten, _mask_strings_and_comments(rewritten), bad_alias, candidate,
    )

    try:
        reparsed = sqlglot.parse(rewritten, dialect="postgres")
    except Exception as exc:
        logger.info(
            component="sql_validator",
            event="autofix_reserved_alias_still_unparseable",
            bad_alias=bad_alias, candidate=candidate,
            error=str(exc)[:120],
        )
        return None, None
    if not reparsed or any(stmt is None for stmt in reparsed):
        logger.info(
            component="sql_validator",
            event="autofix_reserved_alias_still_unparseable",
            bad_alias=bad_alias, candidate=candidate,
        )
        return None, None

    desc = (
        f"autofix: renamed reserved-word alias `{bad_alias}` (on table "
        f"`{table}`) to `{candidate}`"
    )
    logger.info(
        component="sql_validator",
        event="autofix_reserved_alias_accepted",
        bad_alias=bad_alias, table=table, candidate=candidate,
    )
    return rewritten, desc


# ─────────────────────────────────────────────────────────────────────────
# Duplicate table alias autofix (2026-08-14)
# ─────────────────────────────────────────────────────────────────────────
# `... JOIN app_user cb ON cb.id = x.dept_head ... JOIN department cb ON
# cb.id = x.dept_id` -- the same alias bound to two different tables in one
# scope. PostgreSQL rejects this outright ("table name specified more than
# once"); the second declaration is what must be renamed, and every
# reference to it that follows its own declaration (up to the next
# redeclaration, if any) must move with it. Purely mechanical: which
# occurrence of a column reference belongs to which declaration is fully
# determined by clause order, never a guess about intent.

def attempt_duplicate_alias_autofix(sql: str) -> tuple[str | None, str | None]:
    """
    Find an alias bound to more than one table within the same statement and
    rename every occurrence AFTER the first declaration -- both the second
    `FROM`/`JOIN ... <table> <alias>` clause and every `<alias>.<column>`
    reference that follows it -- to a unique name.

    IMPORTANT ambiguity note: PostgreSQL rejects the duplicate outright, so
    (unlike the reserved-word case) there is no "correct" scope resolution
    to defer to -- a reference to the shared alias is genuinely ambiguous in
    the source SQL itself, not just to this tool. "Rename the second
    declaration and every `<alias>.` reference AFTER it" is a deliberate,
    disclosed convention (most-recently-declared-name wins going forward),
    not a claim about the model's original intent, which is unrecoverable.
    Any reference BEFORE the second declaration (most commonly the SELECT
    list, which is textually first) is left untouched and will bind to the
    FIRST (surviving, unrenamed) declaration once the rewrite makes the SQL
    valid -- the only unambiguous outcome available.
    """
    masked = _mask_strings_and_comments(sql)

    # table + explicit or implicit alias, in FROM/JOIN position.
    decl_re = re.compile(
        r'''(?ix)
            \b(from|join)\s+
            (?:[a-z_][a-z0-9_]*\.)?(?P<table>[a-z_][a-z0-9_]*)\s+
            (?:as\s+)?(?P<alias>[a-z_][a-z0-9_]*)\b
            (?!\s*\()
        '''
    )
    decls = [m for m in decl_re.finditer(masked)
             if m.group("alias").lower() not in
             {"on", "where", "group", "order", "having", "limit", "join",
              "left", "right", "inner", "outer", "cross", "full", "union"}]

    seen: dict[str, list["re.Match"]] = {}
    for m in decls:
        seen.setdefault(m.group("alias").lower(), []).append(m)

    dupes = {alias: ms for alias, ms in seen.items() if len(ms) > 1}
    if not dupes:
        return None, None

    # Take the first duplicated alias with more than one distinct TABLE bound
    # to it (re-declaring the same alias on the same table is a redundant but
    # harmless no-op and must not be "fixed" into something else).
    target_alias, matches = None, None
    for alias, ms in dupes.items():
        tables = {m.group("table").lower() for m in ms}
        if len(tables) > 1:
            target_alias, matches = alias, ms
            break
    if target_alias is None:
        return None, None

    second = matches[1]
    table = second.group("table")
    base = re.sub(r"[^a-z0-9_]", "", table.lower())[:3] or "t"
    existing = {m.group("alias").lower() for m in decls}
    candidate = f"{base}_2"
    n = 3
    while candidate in existing:
        candidate = f"{base}_{n}"
        n += 1

    # Rename the SECOND declaration's alias token, then every bare
    # `<alias>.` reference from that declaration's END to the end of the
    # statement (or up to the NEXT re-declaration of the same alias, if any
    # -- rare, but a query could redeclare it a third time).
    #
    # CRITICAL: the substitution below must run ONLY on the slice from the
    # second declaration onward. An earlier revision ran it on the full
    # rewritten string, which also renamed `cb.` inside the FIRST
    # declaration's own ON clause (textually before the second declaration,
    # but still matched by a global regex) -- corrupting a reference that
    # was correct and had nothing to do with the duplicate. Splitting into
    # an untouched `prefix` and a substituted `middle` is what keeps the fix
    # scoped to only what actually needs to change.
    cut_end = len(sql)
    if len(matches) > 2:
        cut_end = matches[2].start("alias")

    alias_start, alias_end = second.span("alias")
    # Rename the declaration token first, then re-mask so offsets line up,
    # then substitute references only in [after-the-rename, cut_end) -- outside
    # string literals, and never touching the prefix, which belongs to the
    # FIRST (surviving) declaration.
    renamed = sql[:alias_start] + candidate + sql[alias_end:]
    shift = len(candidate) - (alias_end - alias_start)
    sub_end = (cut_end + shift) if cut_end < len(sql) else len(renamed)
    rewritten = _sub_alias_refs_outside_literals(
        renamed, _mask_strings_and_comments(renamed), target_alias, candidate,
        start=alias_start + len(candidate),
        end=sub_end,
    )

    try:
        reparsed = sqlglot.parse(rewritten, dialect="postgres")
    except Exception as exc:
        logger.info(
            component="sql_validator",
            event="autofix_duplicate_alias_still_unparseable",
            alias=target_alias, error=str(exc)[:120],
        )
        return None, None
    if not reparsed or any(stmt is None for stmt in reparsed):
        return None, None

    desc = (
        f"autofix: alias `{target_alias}` was bound to two different tables; "
        f"renamed the second occurrence (`{table}`) to `{candidate}`"
    )
    logger.info(
        component="sql_validator",
        event="autofix_duplicate_alias_accepted",
        alias=target_alias, table=table, candidate=candidate,
    )
    return rewritten, desc


# ─────────────────────────────────────────────────────────────────────────
# string_agg / array_agg DISTINCT + ORDER BY autofix (2026-08-14)
# ─────────────────────────────────────────────────────────────────────────
# PostgreSQL rule: `agg(DISTINCT expr [, ...] ORDER BY sort_expr)` requires
# every ORDER BY expression to be one of the DISTINCT'd arguments. Sorting a
# de-duplicated list by a column that was never part of what got de-
# duplicated is genuinely ambiguous to Postgres (which of the collapsed
# duplicate rows' sort_expr would even apply?), so it refuses outright:
# "in an aggregate with DISTINCT, ORDER BY expressions must appear in
# argument list". Confirmed across FOUR separate benchmark runs on
# string_agg(DISTINCT ...) calls.
#
# The mechanical fix is to point ORDER BY at the DISTINCT'd expression
# itself. This is not a guess about intent -- it is the only sort key
# PostgreSQL considers well-defined for a DISTINCT aggregate. It does
# change the sort order actually produced (alphabetical-by-content, rather
# than by whatever the model wanted to sort on), which is a real semantic
# change; that is the trade-off between a query that runs and one that
# doesn't, and it is disclosed in the returned description so it shows up
# in the audit trail and confidence adjustment.

_STRING_AGG_DISTINCT_ORDER_ERR_RE = re.compile(
    r"in an aggregate with distinct.*?order by expressions must appear",
    re.IGNORECASE | re.DOTALL,
)


def attempt_distinct_order_by_autofix(
    sql: str, error_msg: str,
) -> tuple[str | None, str | None]:
    """
    Realign `agg(DISTINCT expr ORDER BY other_expr)` to
    `agg(DISTINCT expr ORDER BY expr)` for every DISTINCT aggregate call in
    the statement whose ORDER BY does not already match its argument.

    sqlglot models `agg(DISTINCT x ORDER BY y)` as
    `Func(this=Order(this=Distinct(expressions=[x]), expressions=[Ordered(this=y)]))`
    -- DISTINCT and ORDER BY are nested INSIDE the function's single `this`
    argument, not as separate top-level args on the function node. Both must
    be found by walking `this`, not by reading `agg.args["distinct"]` /
    `agg.args["order"]`, which do not exist at that level.
    """
    if not _STRING_AGG_DISTINCT_ORDER_ERR_RE.search(error_msg or ""):
        return None, None

    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return None, None
    if not statements or any(s is None for s in statements):
        return None, None

    changed = []
    for stmt in statements:
        if stmt is None:
            continue
        for order_node in stmt.find_all(exp.Order):
            distinct_node = order_node.args.get("this")
            if not isinstance(distinct_node, exp.Distinct):
                continue
            distinct_exprs = distinct_node.expressions
            if len(distinct_exprs) != 1:
                continue   # multi-arg DISTINCT is a different, ambiguous case
            distinct_expr = distinct_exprs[0]

            for ordered in order_node.expressions:
                if not isinstance(ordered, exp.Ordered):
                    continue
                sort_expr = ordered.this
                if sort_expr is None:
                    continue
                if sort_expr.sql(dialect="postgres") == distinct_expr.sql(dialect="postgres"):
                    continue
                old_expr = sort_expr.sql(dialect="postgres")
                ordered.set("this", distinct_expr.copy())
                changed.append((old_expr, distinct_expr.sql(dialect="postgres")))

    if not changed:
        return None, None

    new_sql = statements[0].sql(dialect="postgres") if len(statements) == 1 else \
        ";\n".join(s.sql(dialect="postgres") for s in statements if s is not None)

    try:
        reparsed = sqlglot.parse(new_sql, dialect="postgres")
    except Exception:
        return None, None
    if not reparsed or any(s is None for s in reparsed):
        return None, None

    old, new = changed[0]
    desc = (
        f"autofix: an aggregate's DISTINCT ORDER BY referenced `{old}`, which is "
        f"not part of the DISTINCT'd expression -- PostgreSQL requires the "
        f"ORDER BY key to be one of the DISTINCT arguments. Changed ORDER BY "
        f"to `{new}` (the aggregated expression itself); this sorts the "
        f"de-duplicated values by their own content rather than by "
        f"`{old}`, which is the only ordering PostgreSQL considers "
        f"well-defined here."
    )
    logger.info(
        component="sql_validator",
        event="autofix_distinct_order_by_accepted",
        changes=len(changed), first_change=f"{old} -> {new}",
    )
    return new_sql, desc


# ─────────────────────────────────────────────────────────────────────────
# Missing GROUP BY column autofix (2026-08-14)
# ─────────────────────────────────────────────────────────────────────────
# Mirrors attempt_pg_autofix()'s pattern exactly: PostgreSQL names the exact
# column in its own error text ("column \"x.y\" must appear in the GROUP BY
# clause or be used in an aggregate function"), so the fix is to add that
# literal column to the GROUP BY of the scope where it is missing. No
# inference is needed about which column PostgreSQL means -- it says so.

_MISSING_GROUP_BY_RE = re.compile(
    r'column\s+"([\w.]+)"\s+must appear in the group by clause',
    re.IGNORECASE,
)


def attempt_missing_group_by_autofix(
    sql: str, error_msg: str,
) -> tuple[str | None, str | None]:
    """Add PostgreSQL's own named column to the nearest enclosing GROUP BY."""
    m = _MISSING_GROUP_BY_RE.search(error_msg or "")
    if not m:
        return None, None
    missing_col = m.group(1)

    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return None, None
    if not statements or any(s is None for s in statements):
        return None, None

    target_col = None
    if "." in missing_col:
        tbl, col = missing_col.split(".", 1)
        target_col = exp.column(col, table=tbl)
    else:
        target_col = exp.column(missing_col)

    applied = False
    for stmt in statements:
        if stmt is None:
            continue
        for select_node in stmt.find_all(exp.Select):
            group = select_node.args.get("group")
            if group is None:
                continue
            # Only touch a scope that actually projects the named column and
            # already has a GROUP BY (i.e. is doing grouped aggregation) --
            # never invent a GROUP BY where none existed, and never guess
            # which of several scopes PostgreSQL meant beyond "the one that
            # references this column."
            refs_column = any(
                (c.table or "").lower() == (target_col.table or "").lower()
                and (c.name or "").lower() == (target_col.name or "").lower()
                for c in select_node.find_all(exp.Column)
            )
            if not refs_column:
                continue
            already_grouped = any(
                (g.table or "").lower() == (target_col.table or "").lower()
                and (g.name or "").lower() == (target_col.name or "").lower()
                for g in group.expressions if isinstance(g, exp.Column)
            )
            if already_grouped:
                continue
            group.append("expressions", target_col.copy())
            applied = True

    if not applied:
        return None, None

    new_sql = statements[0].sql(dialect="postgres") if len(statements) == 1 else \
        ";\n".join(s.sql(dialect="postgres") for s in statements if s is not None)

    try:
        reparsed = sqlglot.parse(new_sql, dialect="postgres")
    except Exception:
        return None, None
    if not reparsed or any(s is None for s in reparsed):
        return None, None

    desc = (
        f"autofix: added `{missing_col}` to GROUP BY (PostgreSQL's own error "
        f"named this exact column as missing)"
    )
    logger.info(
        component="sql_validator",
        event="autofix_missing_group_by_accepted",
        column=missing_col,
    )
    return new_sql, desc
