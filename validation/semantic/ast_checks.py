"""
validation/semantic/ast_checks.py
────────────────────────────────
Scope-aware, AST-based implementations of the two logical-audit checks that
carry hard_fail authority: L5 (tautological aggregation) and L8 (filter buried
in a LEFT JOIN ON clause).

WHY THIS MODULE EXISTS

Both checks were originally written against lowercased SQL TEXT with regular
expressions. Text has no notion of scope, so every question the check needs to
answer — "is this column in THIS scope's GROUP BY", "is this alias projected in
THIS scope's SELECT list", "is this aggregate a window function" — had to be
approximated by pattern matching, and every approximation eventually met SQL it
misread. The failure history is the argument:

  * L5 terminator set was `order|having|limit|$`, so a GROUP BY inside a CTE
    ran past the closing paren and swallowed the outer SELECT (run 132132,
    Q138). Widening the terminator set is a bigger regex, not a fix.
  * L5 scanned only the FIRST GROUP BY, making the result depend on CTE
    ordering. Scanning ALL of them and unioning made it worse — an aggregate in
    one CTE matched a GROUP BY in an unrelated one (run 155341, Q43).
  * L5 could not see `SUM(x) OVER (PARTITION BY y)` as a window function
    without a lookahead for the literal token `over`.
  * L8 needed one regex exemption for the anti-join idiom, a second for the
    chained anti-join, and a third for optional attachment — each added after a
    specific query was wrongly blocked (runs 102207, 132132).

That is six corrections to two checks, every one of them reactive. The pattern
is not bad luck; it is what happens when a check with the authority to REJECT a
query reasons about a structured language as though it were a string. sqlglot
already builds the scope tree these checks need, so they should ask it.

WHAT CHANGES SEMANTICALLY

Nothing is loosened. Both checks answer the same questions, from the AST:

  L5  A column is "in the GROUP BY" only when it is in the GROUP BY OF THE SAME
      SELECT SCOPE as the aggregate. Window functions are excluded because they
      have an exp.Window ancestor, not because the token `over` follows.

  L8  A filter is "silently dropped" only when the LEFT-joined alias is used
      nowhere that would make the ON placement deliberate. The three exemptions
      that were bolted on as regexes become one structural question with three
      answers, all read off the same tree.

Both functions keep the (sql, result) signature and the AuditResult contract, so
run_logical_audit does not change shape and neither does anything downstream.
When the SQL will not parse, both return silently: the syntax step owns that
failure and a hard_fail check must never fire on a tree it could not build.
"""

from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp
from sqlglot.optimizer.scope import build_scope


# Aggregates whose value is changed by duplicate rows within a group.
_AGG_NODES = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)


def _parse(sql: str):
    try:
        return sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return None


def _scopes(sql: str):
    """Every SELECT scope in the statement, or [] when the SQL will not parse."""
    tree = _parse(sql)
    if tree is None:
        return []
    try:
        root = build_scope(tree)
    except Exception:
        return []
    if root is None:
        return []
    try:
        return list(root.traverse())
    except Exception:
        return []


def _is_windowed(node: exp.Expression) -> bool:
    """
    A window aggregate computes across its PARTITION, not across a GROUP BY
    group, so no GROUP BY column can make it tautological. Structural test:
    does an exp.Window sit above it?
    """
    return node.find_ancestor(exp.Window) is not None


def _group_by_columns(select_node: exp.Select) -> set[tuple[str | None, str]]:
    """(alias, column) pairs in THIS scope's GROUP BY. Bare names keep alias None."""
    group = select_node.args.get("group")
    if group is None:
        return set()
    out: set[tuple[str | None, str]] = set()
    for e in group.expressions:
        if isinstance(e, exp.Column):
            out.add(((e.table or None) and e.table.lower(), (e.name or "").lower()))
    return out


def _matches_group_column(
    grouped: set[tuple[str | None, str]], alias: str | None, column: str,
) -> bool:
    """
    Exact qualified match, or unqualified on BOTH sides. A qualified GROUP BY
    column never matches a bare name from a different table — that was the Q27
    false-positive class and the rule is preserved verbatim.
    """
    for g_alias, g_col in grouped:
        if g_col != column:
            continue
        if g_alias == alias or (g_alias is None and alias is None):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# L5 — tautological aggregation
# ═══════════════════════════════════════════════════════════════════════════

def check_tautological_aggregation_ast(sql: str, result) -> None:
    """
    COUNT(DISTINCT x) / SUM(x) / AVG(x) where x is in the SAME scope's GROUP BY.

    Grouping by x means every row in a group carries the same x, so
    COUNT(DISTINCT x) is 1, and SUM(x)/AVG(x) return x back. Deterministically
    wrong, which is why L5 may hard_fail — but only when "the same scope" is
    literally true, which is what this rewrite guarantees and the regex could
    not.
    """
    for scope in _scopes(sql):
        select_node = scope.expression
        if not isinstance(select_node, exp.Select):
            continue

        grouped = _group_by_columns(select_node)
        if not grouped:
            continue

        for agg in select_node.find_all(*_AGG_NODES):
            # Only aggregates belonging to THIS scope. find_all descends into
            # nested subqueries, whose aggregates answer to their own GROUP BY.
            if agg.find_ancestor(exp.Select) is not select_node:
                continue
            if _is_windowed(agg):
                continue

            is_count = isinstance(agg, exp.Count)
            has_distinct = any(True for _ in agg.find_all(exp.Distinct))

            # COUNT(x) without DISTINCT counts rows, which is a legitimate
            # thing to do inside a group. Only COUNT(DISTINCT x) collapses to 1.
            if is_count and not has_distinct:
                continue
            if isinstance(agg, (exp.Min, exp.Max)):
                continue   # idempotent; returning x back is the point

            for col in agg.find_all(exp.Column):
                alias = (col.table or None) and col.table.lower()
                column = (col.name or "").lower()
                if not _matches_group_column(grouped, alias, column):
                    continue
                ref = f"{alias}.{column}" if alias else column
                if is_count:
                    detail = (
                        f"COUNT(DISTINCT {ref}) with GROUP BY {ref} always "
                        f"produces 1. This is a tautological aggregation."
                    )
                else:
                    fn = "SUM" if isinstance(agg, exp.Sum) else "AVG"
                    detail = (
                        f"{fn}({ref}) with GROUP BY {ref} is tautological -- "
                        f"every row in the group carries the same {column}, so "
                        f"{fn} just returns that value back rather than "
                        f"aggregating across the matching rows."
                    )
                result.add_warning("L5", detail, penalty=0.10)
                result.hard_fail = True
                return


# ═══════════════════════════════════════════════════════════════════════════
# L8 — value filter inside a LEFT JOIN ON clause
# ═══════════════════════════════════════════════════════════════════════════

def check_left_join_on_filter_ast(sql: str, result) -> None:
    """
    A LEFT JOIN whose ON clause carries a filter on the RIGHT table, where the
    query never uses the right table in a way that makes the placement
    deliberate, is a filter the join semantics silently discard.

    ONE structural rule replaces the three regex exemptions this check
    accumulated (anti-join, chained anti-join, optional attachment) plus a
    fourth that testing revealed was missing (bridge join):

        Is the right-hand alias referenced ANYWHERE in this scope other than
        inside its own ON clause?

    If yes, the LEFT JOIN contributes something and the ON placement is
    deliberate — the filter scopes WHICH row attaches, and hoisting it to WHERE
    would delete left rows that have no match. That single question subsumes
    every idiom, because each of them is just a different place the alias gets
    used:

        WHERE ea.id IS NULL                  anti-join           (Q33)
        LEFT JOIN akr ON akr.ak_id = ak.id   chained anti-join   (Q1)
        SELECT ep.review_required            optional attachment (Q47, Q35)
        COUNT(ea.id)                         conditional agg
        LEFT JOIN em ON em.attempt_id = ea.id  bridge            (Q43)

    If no, the alias is write-only: the join cannot affect the result set
    except by multiplying rows, and the filter inside it is unreachable. Q50 of
    batch run 20260814_102207 is exactly that — `LEFT JOIN script_assignment sa
    ON sa.script_id = ascr.id AND sa.is_active = TRUE` where `sa` appears
    nowhere else in the query.

    Enumerating idioms is what produced six reactive corrections; asking
    whether the join is used at all is a property of the tree and needs no
    maintenance as new idioms appear.
    """
    for scope in _scopes(sql):
        select_node = scope.expression
        if not isinstance(select_node, exp.Select):
            continue

        joins = select_node.args.get("joins") or []

        for join in joins:
            if (join.side or "").upper() != "LEFT":
                continue
            on_clause = join.args.get("on")
            tables = list(join.find_all(exp.Table))
            if on_clause is None or len(tables) != 1:
                continue
            table = tables[0]
            name = (table.name or "").lower()
            alias = (table.alias or name).lower()
            if not alias:
                continue

            # Filters: a comparison between this alias's column and a LITERAL.
            # Column-to-column equalities are join keys, not filters.
            filters: list[str] = []
            for cmp_node in on_clause.find_all(
                exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Like,
            ):
                left, right = cmp_node.args.get("this"), cmp_node.args.get("expression")
                if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                    continue
                for side in (left, right):
                    if isinstance(side, exp.Column) and (side.table or "").lower() == alias:
                        filters.append(cmp_node.sql(dialect="postgres").lower())
                        break
            if not filters:
                continue

            # THE rule: is this alias used anywhere outside its own ON clause?
            used_elsewhere = False
            for col in select_node.find_all(exp.Column):
                if (col.table or "").lower() != alias:
                    continue
                if col.find_ancestor(exp.Join) is join:
                    continue   # inside the ON clause under judgement
                used_elsewhere = True
                break
            if used_elsewhere:
                continue

            result.add_warning(
                "L8",
                f"LEFT JOIN {name} {alias!r} has filter predicate(s) "
                f"{sorted(set(filters))} inside the ON clause, and {alias} "
                f"is referenced nowhere else in the query — not projected, not "
                f"aggregated, not NULL-tested, not joined onward. LEFT JOIN "
                f"keeps every left-side row regardless of the ON clause, so the "
                f"filter changes nothing and the join only multiplies rows. "
                f"Drop the join, or move the filter to WHERE and use an INNER "
                f"JOIN if you meant to restrict rows.",
                penalty=0.10,
            )
            result.hard_fail = True
            return
