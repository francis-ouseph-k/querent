"""
validation/ast/cardinality.py
─────────────────────────────
CardinalityValidator (item #6) — rejects an aggregate computed over a join
fan-out.

THE DEFECT

    SELECT b.id, COUNT(ea.id) AS attempts, AVG(hs.final_amount) AS honorarium
    FROM   board b
    JOIN   evaluation_attempt ea ON ea.board_id = b.id
    JOIN   honorarium_summary hs ON hs.board_id = b.id
    GROUP  BY b.id

Both joins are one-to-many from `board`, so the join produces
|attempts| x |honoraria| rows per board. COUNT(ea.id) is multiplied by the
number of honorarium rows and AVG(hs.final_amount) is weighted by the number
of attempts. Every column exists, every join key is a real foreign key,
EXPLAIN is happy, and the query returns a plausible number that is wrong by an
integer factor nobody can see.

This is the single largest remaining source of silently-wrong answers in the
benchmark. Hand review of batch run 20260814_102207 found it in Q29, Q34, Q46
and Q49 — all shipped as Success with confidence >= 0.87 — while the existing
`semantic_cartesian_explosion` check fired only twice, because it looks for a
JOIN with no ON clause, which is a different (and much rarer) defect.

WHY THIS ONE CAN HARD-FAIL

It is decided from the DDL, not from question keywords. A join is to-one when
the joined side's ON column is that table's primary key or is covered by a
UNIQUE index; otherwise it is to-many. Two to-many branches in one scope
multiply. That is arithmetic, not a heuristic, which puts it on the same
footing as the join-key domain check rather than with the NL-vs-SQL keyword
family that Run-4 demoted to advisory.

WHAT IT DELIBERATELY DOES NOT FLAG

  * COUNT(DISTINCT x) — duplicates collapse, so a fan-out cannot change the
    answer. This is the standard fix and must never be rejected.
  * A single to-many branch — no second branch to multiply against.
  * MIN / MAX — idempotent under duplication.
  * Scopes where the aggregate touches only to-one tables.
  * Anything it cannot resolve. An unresolvable alias, a derived table, or a
    join key with no PK/UNIQUE evidence all return "unknown", and unknown is
    never treated as to-many. A check that guesses here fires on correct SQL.

The fix the message asks for — aggregate each branch in its own CTE and join
the pre-aggregated results — is the same one a reviewer would ask for, and the
model produces it reliably when told.
"""

from __future__ import annotations

import sqlglot.expressions as exp

from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)


# MIN/MAX are unaffected by duplicate rows. COUNT is only affected without
# DISTINCT. SUM/AVG are always affected.
_MULTIPLYING_AGGS = (exp.Sum, exp.Avg, exp.Count)


def _unique_columns(inventory) -> set[str]:
    """
    Columns that identify at most one row: primary keys, plus single-column
    UNIQUE indexes. A composite UNIQUE does not qualify — matching one of its
    columns still permits many rows.
    """
    out: set[str] = set()
    for name, col in (getattr(inventory, "columns", {}) or {}).items():
        if getattr(col, "is_pk", False):
            out.add(name.lower())
    for idx in getattr(inventory, "indexes", []) or []:
        if not getattr(idx, "is_unique", False):
            continue
        if getattr(idx, "is_partial", False):
            # A partial UNIQUE only constrains the rows matching its predicate,
            # which this check cannot evaluate.
            continue
        cols = getattr(idx, "columns", []) or []
        if len(cols) == 1:
            out.add(str(cols[0]).lower())
    return out


class CardinalityValidator(BaseValidationStep):
    """Rejects SUM/AVG/COUNT computed across two or more to-many join branches."""

    name = "CardinalityValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        sql = ctx.working_sql or ctx.sql

        if not ctx.ast or not ctx.schema_map:
            return ValidationResult(passed=True, step="semantic", sql=sql)

        try:
            for stmt in ctx.ast:
                if stmt is None:
                    continue
                cte_names = {(c.alias or "").lower() for c in stmt.find_all(exp.CTE)}
                for select_node in stmt.find_all(exp.Select):
                    finding = self._check_scope(select_node, ctx.schema_map, cte_names)
                    if finding is None:
                        continue
                    agg_sql, branches = finding
                    logger.warning(
                        component="sql_validator",
                        event="aggregate_over_join_fanout",
                        aggregate=agg_sql,
                        to_many_branches=branches,
                        sql_preview=sql[:120],
                    )
                    return ValidationResult(
                        passed=False, step="semantic",
                        message=(
                            f"`{agg_sql}` is computed across a join fan-out. This "
                            f"scope joins {len(branches)} one-to-many branches off "
                            f"the SAME parent row ({', '.join(branches)}), so the "
                            f"join emits the PRODUCT of their row counts and the "
                            f"aggregate is multiplied by the size of the sibling "
                            f"branches. "
                            f"Aggregate each branch in its own CTE and join the "
                            f"pre-aggregated results, or use COUNT(DISTINCT ...) "
                            f"if you only need a distinct count."
                        ),
                        sql=sql,
                    )
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="cardinality_check_error",
                error=str(exc),
                note="fan-out check skipped due to AST error",
            )

        return ValidationResult(passed=True, step="semantic", sql=sql)

    # ── scope analysis ────────────────────────────────────────────────────
    def _check_scope(
        self, select_node: exp.Select, schema_map: dict, cte_names: set[str],
    ) -> tuple[str, list[str]] | None:
        alias_to_table: dict[str, str] = {}

        from_node = select_node.args.get("from") or select_node.args.get("from_")
        if from_node:
            for tbl in from_node.find_all(exp.Table):
                name = (tbl.name or "").lower()
                if name and name not in cte_names:
                    alias_to_table[(tbl.alias or name).lower()] = name

        joins = select_node.args.get("joins", []) or []
        if len(joins) < 2:
            # One join cannot fan out against a sibling.
            return None

        # alias -> (table, parent_alias). The parent is the alias on the OTHER
        # side of the ON equality — i.e. the row this join hangs off.
        to_many: dict[str, tuple[str, str]] = {}

        for join in joins:
            tables = [t for t in join.find_all(exp.Table)]
            if len(tables) != 1:
                # A joined derived table or a multi-table construct — unknown.
                continue
            tbl = tables[0]
            name = (tbl.name or "").lower()
            alias = (tbl.alias or name).lower()
            if not name:
                continue
            if name in cte_names:
                # A CTE's cardinality is whatever its body produced; treating it
                # as to-many would fire on the very rewrite this check asks for.
                alias_to_table[alias] = name
                continue
            alias_to_table[alias] = name

            on_clause = join.args.get("on")
            if on_clause is None:
                continue   # cartesian: JoinValidator owns that case

            inv = schema_map.get(name)
            if inv is None:
                continue
            uniques = _unique_columns(inv)

            # Does any ON equality bind this table on a column that identifies
            # at most one of its rows? If so the join is to-one and safe.
            bound_to_one = False
            saw_own_column = False
            parent_alias = ""
            for eq in on_clause.find_all(exp.EQ):
                if not (isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column)):
                    continue
                for side, other in ((eq.left, eq.right), (eq.right, eq.left)):
                    if (side.table or "").lower() != alias:
                        continue
                    saw_own_column = True
                    if (side.name or "").lower() in uniques:
                        bound_to_one = True
                    other_alias = (other.table or "").lower()
                    if other_alias and other_alias != alias and not parent_alias:
                        parent_alias = other_alias
            if saw_own_column and not bound_to_one and parent_alias:
                to_many[alias] = (name, parent_alias)

        # SIBLINGS, NOT CHAINS. Two to-many joins multiply only when they hang
        # off the SAME parent row:
        #
        #   board b JOIN attempt ea ON ea.board_id = b.id      <- child of b
        #           JOIN honorarium hs ON hs.board_id = b.id    <- child of b
        #     => |ea| x |hs| rows per board. A product.
        #
        #   attempt_rule ar JOIN arg  ON arg.attempt_rule_id = ar.id
        #                   JOIN argq ON argq.group_id = arg.id  <- child of arg
        #     => one row per leaf. A chain, and perfectly correct.
        #
        # Q15 of batch run 20260814_132132 is the chain above; an earlier draft
        # of this check flagged it purely because two to-many joins appeared in
        # one scope. Requiring a shared parent is what makes the rule an
        # arithmetic fact rather than a guess.
        siblings: dict[str, list[str]] = {}
        for alias, (_table, parent) in to_many.items():
            siblings.setdefault(parent, []).append(alias)
        fanning = {a for group in siblings.values() if len(group) >= 2 for a in group}
        if not fanning:
            return None

        # A multiplying aggregate that reads one of the fanning branches.
        for projection in select_node.expressions:
            for agg in projection.find_all(*_MULTIPLYING_AGGS):
                # A window aggregate reads the post-join row set by design; its
                # correctness depends on the PARTITION BY, which this check does
                # not model. Leave it alone rather than guess.
                if agg.find_ancestor(exp.Window) is not None or agg.args.get("over"):
                    continue
                # sqlglot models COUNT(DISTINCT x) as Count(this=Distinct(...)),
                # not as a `distinct` arg — checking args.get("distinct") is
                # always False and would flag every COUNT(DISTINCT ...) in the
                # corpus. Duplicates collapse under DISTINCT, so a fan-out
                # cannot change the answer and this must never fire.
                if isinstance(agg, exp.Count) and any(
                    True for _ in agg.find_all(exp.Distinct)
                ):
                    continue
                for col in agg.find_all(exp.Column):
                    if (col.table or "").lower() in fanning:
                        return (
                            agg.sql(dialect="postgres")[:80],
                            sorted(f"{a} ({to_many[a][0]})" for a in fanning),
                        )
        return None
