"""
validation/semantic/zero_suppression.py
───────────────────────────────────────
ZeroSuppressionValidator — rejects an average-of-counts whose population is
structurally incapable of containing a zero.

THE DEFECT

    SELECT AVG(correction_count)
    FROM (
      SELECT r.id, COUNT(rh.id) AS correction_count
      FROM   result r
      JOIN   result_history rh ON rh.result_id = r.id     -- INNER
      WHERE  r.published_at IS NOT NULL
      GROUP  BY r.id
    ) sub

The question was "what is the average number of history corrections per
published result". The inner block groups by `result.id` and counts an
INNER-joined child, so a result with no corrections produces no group at all.
Every surviving group has a count of at least one, and the outer AVG is
therefore an average over "results that have corrections" — a different, and
always larger, number than the one asked for.

Six shapes of this appeared in batch run 20260818_085111 (Q12, Q26, Q49, Q105,
Q108, Q127). Each returned a plausible number, each shipped as Success, and
each is wrong by a factor that depends on how many parents have zero children —
a quantity nobody can see in the output.

WHY THIS CHECK, AND NOT THE KEYWORD ONE

`semantic_checks.per_entity_left_join_error` already targets this family, and
it is correctly advisory: it triggers on the words "per" / "for each" in the
question, and Run-4 showed the retry it forced turned a correct LEFT JOIN
query (Q20) into an INNER JOIN one. A check that compares question words to
SQL words cannot tell "per X, include zeroes" from "per X, among those that
have any", so it must not block.

This check reads neither. It fires only on a structural fact:

    an aggregate over a grouped block, where
      · the grouping key is (or is functionally determined by) the parent's
        own primary key or a single-column UNIQUE, and
      · the aggregate inside that block is COUNT(...) over a table reached by
        an INNER join whose ON binds that child's foreign key to the grouping
        parent, and
      · the outer expression averages or minimises that count.

Under those conditions the inner COUNT is provably >= 1 for every row the outer
aggregate sees. That is arithmetic, not intent, which puts it on the same
footing as the cardinality and satisfiability validators rather than with the
NL-keyword family.

It is not a dead end for the legitimate reading. If the author really did mean
"among parents that have at least one child", the LEFT JOIN rewrite plus an
explicit `HAVING COUNT(...) > 0` says so, states it in the SQL where a reader
can see it, and passes this check.

WHAT IT DELIBERATELY DOES NOT FLAG

  * SUM / MAX over the inner count — zero-groups do not change a SUM's total or
    a MAX, so their absence is not an error.
  * A LEFT / RIGHT / FULL join to the child. That is the correct shape.
  * A grouped block whose grouping key is not row-unique for the parent —
    without that, "one group per parent" is not established and the
    zero-suppression argument does not hold.
  * Anything it cannot resolve: an unresolvable alias, a child with no FK
    evidence, or a grouping key it cannot tie to a table. Unknown is never
    treated as a violation.
"""

from __future__ import annotations

import sqlglot.expressions as exp

from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)


# Outer aggregates whose value is changed by the absence of zero-valued groups.
# SUM is unaffected (adding zeroes changes nothing) and so is MAX.
_ZERO_SENSITIVE_OUTER = (exp.Avg, exp.Min)


def _row_unique_columns(inventory) -> set[str]:
    """Columns that identify at most one row of this table."""
    out: set[str] = set()
    for name, col in (getattr(inventory, "columns", {}) or {}).items():
        if getattr(col, "is_pk", False):
            out.add(name.lower())
    for idx in getattr(inventory, "indexes", []) or []:
        if not getattr(idx, "is_unique", False) or getattr(idx, "is_partial", False):
            continue
        cols = getattr(idx, "columns", []) or []
        if len(cols) == 1:
            out.add(str(cols[0]).lower())
    return out


def _scope_aliases(select_node: exp.Select, cte_names: set[str]) -> dict[str, str]:
    """alias -> base table name, for real tables only (CTEs excluded)."""
    aliases: dict[str, str] = {}
    from_node = select_node.args.get("from") or select_node.args.get("from_")
    sources = []
    if from_node is not None:
        sources.extend(from_node.find_all(exp.Table))
    for join in select_node.args.get("joins", []) or []:
        if isinstance(join.this, exp.Table):
            sources.append(join.this)
    for tbl in sources:
        name = (tbl.name or "").lower()
        if not name or name in cte_names:
            continue
        aliases[(tbl.alias or name).lower()] = name
    return aliases


class ZeroSuppressionValidator(BaseValidationStep):
    """Rejects AVG/MIN over a COUNT that an INNER join guarantees is non-zero."""

    name = "ZeroSuppressionValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        sql = ctx.working_sql or ctx.sql
        if not ctx.ast or not ctx.schema_map:
            return ValidationResult(passed=True, step="semantic", sql=sql)

        try:
            for stmt in ctx.ast:
                if stmt is None:
                    continue
                cte_names = {(c.alias or "").lower() for c in stmt.find_all(exp.CTE)}
                finding = self._check_statement(stmt, ctx.schema_map, cte_names)
                if finding is None:
                    continue

                parent, child, fk, outer = finding
                logger.warning(
                    component="sql_validator",
                    event="zero_suppressed_population",
                    parent_table=parent,
                    child_table=child,
                    join_key=fk,
                    outer_aggregate=outer,
                    sql_preview=sql[:120],
                )
                return ValidationResult(
                    passed=False, step="semantic",
                    message=(
                        f"`{outer}` is computed over a population that cannot "
                        f"contain a zero. The inner block groups one row per "
                        f"`{parent}` and counts `{child}` rows reached by an "
                        f"INNER JOIN on {fk}, so a `{parent}` with no "
                        f"`{child}` rows produces no group at all and is "
                        f"excluded from the average — which is therefore an "
                        f"average over `{parent}` rows that HAVE at least one "
                        f"`{child}`, not over all of them. "
                        f"Use LEFT JOIN {child} so zero-count `{parent}` rows "
                        f"survive with a count of 0. If you deliberately want "
                        f"only `{parent}` rows that have at least one "
                        f"`{child}`, keep the LEFT JOIN and state it as "
                        f"HAVING COUNT(...) > 0 so the restriction is visible."
                    ),
                    sql=sql,
                )
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="zero_suppression_check_error",
                error=str(exc),
                note="check skipped due to AST error",
            )

        return ValidationResult(passed=True, step="semantic", sql=sql)

    # ── analysis ──────────────────────────────────────────────────────────
    def _check_statement(
        self, stmt: exp.Expression, schema_map: dict, cte_names: set[str],
    ) -> tuple[str, str, str, str] | None:
        # Every grouped block that could be the inner population, keyed by the
        # name it is exposed under (CTE alias or derived-table alias).
        populations: dict[str, tuple[str, str, str]] = {}
        self._inner_aggregates: list[exp.Expression] = []

        for cte in stmt.find_all(exp.CTE):
            body = cte.this
            if isinstance(body, exp.Select):
                info = self._grouped_inner_count(body, schema_map, cte_names)
                if info is not None:
                    populations[(cte.alias_or_name or "").lower()] = info
                    self._inner_aggregates.extend(body.find_all(*_ZERO_SENSITIVE_OUTER))

        for sub in stmt.find_all(exp.Subquery):
            body = sub.this
            if not isinstance(body, exp.Select):
                continue
            info = self._grouped_inner_count(body, schema_map, cte_names)
            if info is None:
                continue
            populations[(sub.alias_or_name or "").lower()] = info
            self._inner_aggregates.extend(body.find_all(*_ZERO_SENSITIVE_OUTER))

        if not populations:
            return None

        # An outer AVG/MIN whose argument reads one of those populations.
        #
        # The population may be referenced three ways, and all three occur in
        # the corpus:
        #   AVG(sub.n)   -- derived table, referenced by its own alias
        #   AVG(cc.n)    -- CTE joined under a DIFFERENT alias (`counts cc`)
        #   AVG(n)       -- unqualified, when only one source is in scope
        # Resolving only the first is why an earlier draft caught two of the
        # six instances in run 20260818_085111 and missed the rest.
        for outer_select in stmt.find_all(exp.Select):
            local = self._population_aliases(outer_select, populations)
            if not local:
                continue
            for agg in outer_select.find_all(*_ZERO_SENSITIVE_OUTER):
                if agg.find_ancestor(exp.Window) is not None or agg.args.get("over"):
                    continue
                # An AVG *inside* the grouped population is a different
                # expression -- do not treat a block as its own consumer.
                if any(agg is inner for inner in self._inner_aggregates):
                    continue
                for col in agg.find_all(exp.Column):
                    key = (col.table or "").lower()
                    info = local.get(key)
                    if info is None and not key and len(set(local.values())) == 1:
                        # Unqualified: unambiguous only when one population is
                        # in scope. Otherwise say nothing.
                        info = next(iter(local.values()))
                    if info is None:
                        continue
                    parent, child, fk = info
                    return (parent, child, fk, agg.sql(dialect="postgres")[:80])
        return None

    @staticmethod
    def _population_aliases(
        select_node: exp.Select, populations: dict[str, tuple[str, str, str]],
    ) -> dict[str, tuple[str, str, str]]:
        """
        Map every name this scope can refer a population by -- the source's
        own name AND any alias it was given here -- onto that population.
        """
        local: dict[str, tuple[str, str, str]] = {}
        sources: list[exp.Expression] = []
        from_node = select_node.args.get("from") or select_node.args.get("from_")
        if from_node is not None:
            sources.extend(from_node.find_all(exp.Table))
            sources.extend(from_node.find_all(exp.Subquery))
        for join in select_node.args.get("joins", []) or []:
            if isinstance(join.this, (exp.Table, exp.Subquery)):
                sources.append(join.this)

        for src in sources:
            name = (getattr(src, "name", "") or "").lower()
            alias = (src.alias or "").lower()
            info = populations.get(name) or populations.get(alias)
            if info is None:
                continue
            for key in (name, alias):
                if key:
                    local[key] = info
            # Also reachable unqualified.
            local.setdefault("", info)
        return local

    @staticmethod
    def _grouped_inner_count(
        select_node: exp.Select, schema_map: dict, cte_names: set[str],
    ) -> tuple[str, str, str] | None:
        """
        Is this block "one row per parent, counting an INNER-joined child"?
        Returns (parent_table, child_table, join_key_description) or None.
        """
        group = select_node.args.get("group")
        if group is None:
            return None

        aliases = _scope_aliases(select_node, cte_names)
        if not aliases:
            return None

        # The grouping key must be row-unique for exactly one parent alias.
        parent_alias = ""
        parent_table = ""
        for g in group.expressions:
            if not isinstance(g, exp.Column):
                continue
            alias = (g.table or "").lower()
            table = aliases.get(alias)
            if not table:
                continue
            inv = schema_map.get(table)
            if inv is None:
                continue
            if (g.name or "").lower() in _row_unique_columns(inv):
                parent_alias, parent_table = alias, table
                break
        if not parent_alias:
            return None

        # An INNER-joined child bound to that parent's key.
        for join in select_node.args.get("joins", []) or []:
            side = (getattr(join, "side", "") or "").upper()
            kind = (getattr(join, "kind", "") or "").upper()
            if side in ("LEFT", "RIGHT", "FULL") or kind == "CROSS":
                continue
            if not isinstance(join.this, exp.Table):
                continue
            child_table = (join.this.name or "").lower()
            if not child_table or child_table in cte_names:
                continue
            child_alias = (join.this.alias or child_table).lower()
            on_clause = join.args.get("on")
            if on_clause is None:
                continue

            for eq in on_clause.find_all(exp.EQ):
                if not (isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column)):
                    continue
                sides = {(eq.left.table or "").lower(): eq.left,
                         (eq.right.table or "").lower(): eq.right}
                if parent_alias not in sides or child_alias not in sides:
                    continue
                # The COUNT must read this child.
                for agg in select_node.find_all(exp.Count):
                    reads_child = any(
                        (c.table or "").lower() == child_alias
                        for c in agg.find_all(exp.Column)
                    )
                    if not reads_child:
                        continue
                    if any(True for _ in agg.find_all(exp.Distinct)):
                        # COUNT(DISTINCT ...) is still >= 1 here; the
                        # zero-suppression argument is unaffected by DISTINCT.
                        pass
                    fk = (
                        f"{sides[child_alias].sql(dialect='postgres')} = "
                        f"{sides[parent_alias].sql(dialect='postgres')}"
                    )
                    return (parent_table, child_table, fk)
        return None
