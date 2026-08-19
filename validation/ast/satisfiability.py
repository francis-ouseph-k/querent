"""
validation/ast/satisfiability.py
────────────────────────────────
SatisfiabilityValidator — rejects predicates that no row can ever satisfy, and
predicates that every row always satisfies.

WHY THIS IS A SEPARATE CONCERN

Every other validator in this pipeline asks "does this query reference real
objects, join them legally, and cost acceptably?" -- all of which a
structurally unsatisfiable query passes trivially. `GROUP BY <primary key>
HAVING COUNT(*) > 1` names real tables, joins nothing illegally, and EXPLAINs
for pennies. It also cannot return a row, ever, on any data, because grouping
by a unique key puts exactly one row in each group. The query is not slow or
risky; it is *arithmetically incapable* of answering the question, and it
reports that incapacity as the number 0, which reads exactly like a true
finding of "none".

That is the whole reason this deserves a hard failure rather than a warning:
the defect is invisible in the output. A wrong join returns wrong rows someone
may notice. An unsatisfiable predicate returns an empty set that looks like a
legitimate answer and will be copied into a report.

WHAT MAKES THESE CHECKS SAFE TO BLOCK

Each rule below is decided from the AST plus the DDL's own key metadata. None
of them consults the natural-language question, none pattern-matches a known
bad query, and none encodes a value from any specific schema. They are the
same class of check as the join-key domain and cardinality validators: facts
about relational algebra, not guesses about intent.

The rules are deliberately narrow. Where satisfiability is undecidable in
general (arbitrary boolean algebra over unknown data), this module says
nothing at all rather than approximating -- an approximation with hard_fail
authority is how a validator starts rejecting correct SQL.
"""

from __future__ import annotations

import sqlglot.expressions as exp

from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _unique_column_sets(inventory) -> list[set[str]]:
    """
    Column sets that uniquely identify a row: the primary key, plus every
    UNIQUE index. Composite keys are kept as sets, because grouping by a
    composite key is just as row-unique as grouping by a single-column one.
    """
    out: list[set[str]] = []

    pk = {
        name.lower()
        for name, col in (getattr(inventory, "columns", {}) or {}).items()
        if getattr(col, "is_pk", False)
    }
    if pk:
        out.append(pk)

    for idx in getattr(inventory, "indexes", []) or []:
        if not getattr(idx, "is_unique", False):
            continue
        if getattr(idx, "is_partial", False):
            # A partial UNIQUE constrains only the rows matching its predicate,
            # which this check cannot evaluate.
            continue
        cols = {str(c).lower() for c in (getattr(idx, "columns", []) or [])}
        if cols:
            out.append(cols)
    return out


def _alias_map(stmt: exp.Expression) -> dict[str, str]:
    """alias (and bare table name) -> base table name, lowercased."""
    out: dict[str, str] = {}
    cte_names = {(c.alias or "").lower() for c in stmt.find_all(exp.CTE)}
    for tbl in stmt.find_all(exp.Table):
        name = (tbl.name or "").lower()
        if not name or name in cte_names:
            continue
        out[(tbl.alias or name).lower()] = name
        out.setdefault(name, name)
    return out


def _group_is_row_unique(
    select_node: exp.Select, aliases: dict[str, str], schema_map: dict,
) -> tuple[bool, str]:
    """
    True when this scope's GROUP BY contains a full unique key of some table it
    reads -- i.e. each group is guaranteed to hold exactly one row of that
    table.

    Only meaningful for a SINGLE-table scope. With a join, one base table's key
    being in the GROUP BY says nothing about how many joined rows land in the
    group, so the guarantee evaporates and this returns False.
    """
    from_node = select_node.args.get("from") or select_node.args.get("from_")
    if from_node is None:
        return False, ""
    if select_node.args.get("joins"):
        return False, ""

    tables = list(from_node.find_all(exp.Table))
    if len(tables) != 1:
        return False, ""

    canon = aliases.get((tables[0].alias or tables[0].name or "").lower())
    inventory = schema_map.get(canon) if canon else None
    if inventory is None:
        return False, ""

    group = select_node.args.get("group")
    if group is None:
        return False, ""
    grouped = {
        (c.name or "").lower()
        for c in group.find_all(exp.Column)
    }
    if not grouped:
        return False, ""

    for key in _unique_column_sets(inventory):
        if key and key.issubset(grouped):
            return True, f"{canon} ({', '.join(sorted(key))})"
    return False, ""


def _count_star_lower_bound(having: exp.Expression) -> exp.Expression | None:
    """
    Find a `COUNT(*) > n` / `>= n` / `= n` comparison with n > 1, which no
    single-row group can satisfy. Returns the offending node, or None.
    """
    for cmp_node in having.find_all(exp.GT, exp.GTE, exp.EQ):
        left, right = cmp_node.args.get("this"), cmp_node.args.get("expression")
        for agg_side, lit_side in ((left, right), (right, left)):
            if not isinstance(agg_side, exp.Count):
                continue
            # COUNT(*) only -- COUNT(col) skips NULLs and COUNT(DISTINCT col)
            # counts values, neither of which is pinned to 1 by row-uniqueness.
            arg = agg_side.args.get("this")
            if not isinstance(arg, exp.Star):
                continue
            if not isinstance(lit_side, exp.Literal) or lit_side.is_string:
                continue
            try:
                bound = float(lit_side.this)
            except (TypeError, ValueError):
                continue
            is_gt = isinstance(cmp_node, exp.GT) and bound >= 1
            is_gte_or_eq = isinstance(cmp_node, (exp.GTE, exp.EQ)) and bound > 1
            if is_gt or is_gte_or_eq:
                return cmp_node
    return None


def _find_range_tautology(node: exp.Expression) -> exp.Or | None:
    """
    Find `X <op> k OR X <op'> k` where the two comparisons between the SAME
    expression and the SAME constant together cover every possible value --
    e.g. `c = 0 OR c > 0` over a count (which is never negative), or
    `c >= k OR c < k` over anything.

    A disjunction that is always true is a filter the author believed was
    filtering. It is not a syntax error and PostgreSQL will not complain; the
    query simply returns everything, and the caller reads that as the filtered
    set.
    """
    for or_node in node.find_all(exp.Or):
        left, right = or_node.left, or_node.right
        if not isinstance(left, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ)):
            continue
        if not isinstance(right, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ)):
            continue

        def split(cmp_node):
            a, b = cmp_node.args.get("this"), cmp_node.args.get("expression")
            if isinstance(b, exp.Literal) and not b.is_string:
                return a, b
            if isinstance(a, exp.Literal) and not a.is_string:
                return b, a
            return None, None

        l_expr, l_lit = split(left)
        r_expr, r_lit = split(right)
        if l_expr is None or r_expr is None:
            continue
        if l_expr.sql() != r_expr.sql() or l_lit.this != r_lit.this:
            continue

        ops = {type(left), type(right)}
        # Complementary pairs that exhaust the number line.
        if ops in ({exp.GTE, exp.LT}, {exp.LTE, exp.GT}, {exp.EQ, exp.NEQ}):
            return or_node
        # `= k OR > k` and `= k OR < k` exhaust only when the expression cannot
        # go the other side of k. COUNT/aggregate results are non-negative, so
        # with k = 0 these are tautologies.
        if ops == {exp.EQ, exp.GT} and str(l_lit.this) == "0":
            if isinstance(l_expr, (exp.Count, exp.Sum)) or l_expr.find(exp.Count):
                return or_node
    return None


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    """Flatten a top-level AND chain into its individual predicates."""
    out: list[exp.Expression] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (exp.Where, exp.Having, exp.Paren)):
            # Unwrap the clause node itself, and any redundant parens,
            # so the caller may hand us select_node.args["where"] directly
            # rather than having to reach for its payload.
            stack.append(cur.this)
        elif isinstance(cur, exp.And):
            stack.extend([cur.this, cur.expression])
        elif cur is not None:
            out.append(cur)
    return out


def _string_equality(pred: exp.Expression) -> tuple[str, str] | None:
    """(qualified_column, literal) for `col = 'lit'`, else None."""
    if not isinstance(pred, exp.EQ):
        return None
    for col, lit in ((pred.this, pred.expression), (pred.expression, pred.this)):
        if isinstance(col, exp.Column) and isinstance(lit, exp.Literal) and lit.is_string:
            return (col.sql(dialect="postgres").lower(), lit.name)
    return None


def _find_equality_contradiction(
    clause: exp.Expression,
) -> tuple[str, str, str] | None:
    """
    Two ANDed predicates that pin the SAME column to DIFFERENT string values,
    or an equality ANDed with an IN list that excludes its value.

    A column holds one value per row, so `status = 'A' AND status = 'B'`
    matches nothing on any data. This is decided purely from the AST -- no
    schema value list, no question keywords -- and it is the same failure
    mode the rest of this module exists for: an empty result set that reads
    like a legitimate finding of \"none\".

    Only equalities on the SAME qualified column are compared, and only
    inside a pure AND chain. Anything reached through an OR is left alone --
    a disjunct may legitimately name a different value.
    """
    pinned: dict[str, tuple[str, str]] = {}
    preds = _conjuncts(clause)

    for pred in preds:
        eq = _string_equality(pred)
        if eq is None:
            continue
        col, value = eq
        prior = pinned.get(col)
        if prior is not None and prior[0] != value:
            return (col, prior[1], pred.sql(dialect="postgres"))
        pinned[col] = (value, pred.sql(dialect="postgres"))

    for pred in preds:
        target = pred.this if isinstance(pred, exp.Not) else pred
        negated = isinstance(pred, exp.Not)
        if isinstance(target, exp.Paren):
            target = target.this
        if not isinstance(target, exp.In) or not isinstance(target.this, exp.Column):
            continue
        col = target.this.sql(dialect="postgres").lower()
        prior = pinned.get(col)
        if prior is None:
            continue
        members = {
            e.name for e in target.expressions
            if isinstance(e, exp.Literal) and e.is_string
        }
        if not members:
            continue
        inside = prior[0] in members
        if inside is negated:
            return (col, prior[1], target.sql(dialect="postgres"))
    return None


def _is_constant_true(node: exp.Expression | None) -> bool:
    """TRUE, or a literal equality that is trivially true (`1 = 1`)."""
    if node is None:
        return False
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.EQ):
        left, right = node.left, node.right
        if isinstance(left, exp.Literal) and isinstance(right, exp.Literal):
            return str(left.this) == str(right.this)
    return False


def _is_provably_single_row(node: exp.Expression | None) -> bool:
    """
    True when a relation can hold at most one row: an ungrouped aggregate, or
    an explicit LIMIT 1.

    `JOIN one_row_cte ON TRUE` is a legitimate way to broadcast a scalar, and
    is exactly how a "current academic year" CTE is normally attached. Rule 4
    must not touch it.
    """
    if node is None:
        return False
    if isinstance(node, exp.Subquery):
        node = node.this
    if not isinstance(node, exp.Select):
        return False
    limit = node.args.get("limit")
    if limit is not None:
        value = limit.expression if hasattr(limit, "expression") else None
        if isinstance(value, exp.Literal) and str(value.this) == "1":
            return True
    if node.args.get("group"):
        return False
    return any(
        isinstance(projection, exp.AggFunc)
        or any(isinstance(sub, exp.AggFunc) for sub in projection.find_all(exp.AggFunc))
        for projection in node.expressions
    )


def _resolve_relation(
    node: exp.Expression | None, cte_bodies: dict[str, exp.Expression],
) -> exp.Expression | None:
    """The Select behind a join target, following a CTE reference by name."""
    if node is None:
        return None
    if isinstance(node, exp.Table) and node.name:
        return cte_bodies.get(node.name.lower())
    if isinstance(node, exp.Subquery):
        return node.this
    return node


def _null_rejecting_aliases(clause: exp.Expression | None) -> set[str]:
    """
    Aliases that a WHERE clause proves NON-NULL, by referencing one of their
    columns in a predicate that a NULL row cannot satisfy.

    Only top-level conjuncts are considered — a predicate under an OR proves
    nothing — and `IS NULL` / `IS NOT NULL` are excluded, since those are
    precisely the predicates that tolerate a null-extended row.
    """
    if clause is None:
        return set()
    out: set[str] = set()
    for conjunct in _conjuncts(clause):
        if isinstance(conjunct, exp.Is):
            continue
        if any(isinstance(sub, exp.Is) for sub in conjunct.find_all(exp.Is)):
            continue
        if isinstance(conjunct, (exp.Or, exp.Not)):
            continue
        for column in conjunct.find_all(exp.Column):
            if column.table:
                out.add(column.table.lower())
    return out


def _null_asserted_columns(clause: exp.Expression | None) -> list[tuple[str, str]]:
    """(alias, column) for every top-level `alias.column IS NULL` conjunct."""
    if clause is None:
        return []
    out: list[tuple[str, str]] = []
    for conjunct in _conjuncts(clause):
        if not isinstance(conjunct, exp.Is):
            continue
        if not isinstance(conjunct.expression, exp.Null):
            continue
        target = conjunct.this
        if isinstance(target, exp.Column) and target.table:
            out.append((target.table.lower(), (target.name or "").lower()))
    return out


def _inner_join_groups(select_node: exp.Select) -> list[set[str]]:
    """
    Aliases grouped by "cannot be null-extended relative to one another".

    The FROM root plus every INNER-joined relation form one group: if any of
    them is present in an output row, all of them are. Each OUTER-joined
    relation starts its own group, because it is exactly the thing that CAN be
    null-extended. A RIGHT JOIN inverts the roles, so everything accumulated so
    far becomes null-supplied and the right side becomes the surviving root.
    """
    from_node = select_node.args.get("from") or select_node.args.get("from_")
    if from_node is None:
        return []

    def alias_of(node: exp.Expression) -> str | None:
        if isinstance(node, (exp.Table, exp.Subquery)):
            return (node.alias or getattr(node, "name", "") or "").lower() or None
        return None

    root: set[str] = set()
    groups: list[set[str]] = []
    for table in from_node.find_all(exp.Table):
        name = alias_of(table)
        if name:
            root.add(name)

    current = root
    for join in select_node.args.get("joins") or []:
        side = (join.args.get("side") or "").upper()
        kind = (join.args.get("kind") or "").upper()
        name = alias_of(join.this)
        if name is None:
            continue
        if side == "LEFT":
            groups.append({name})
        elif side == "RIGHT":
            # Everything to the left becomes null-supplied; the right side is
            # the new non-null root.
            groups.append(current)
            current = {name}
        elif side == "FULL":
            groups.append(current)
            groups.append({name})
            current = set()
        elif kind == "CROSS":
            current = current | {name}
        else:
            current = current | {name}
    groups.append(current)
    return [group for group in groups if group]


def _strip_scalar_factors(node: exp.Expression) -> exp.Expression:
    """`COUNT(*) * 100.0` -> `COUNT(*)`; `NULLIF(x, 0)` -> `x`."""
    while True:
        if isinstance(node, exp.Paren):
            node = node.this
            continue
        if isinstance(node, exp.Nullif):
            node = node.this
            continue
        if isinstance(node, exp.Cast):
            node = node.this
            continue
        if isinstance(node, exp.Mul):
            left, right = node.this, node.expression
            if isinstance(right, exp.Literal) and not right.is_string:
                node = left
                continue
            if isinstance(left, exp.Literal) and not left.is_string:
                node = right
                continue
        return node


class SatisfiabilityValidator(BaseValidationStep):
    """Rejects always-empty and always-true predicates."""

    name = "SatisfiabilityValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        sql = ctx.working_sql or ctx.sql
        if not ctx.ast or not ctx.schema_map:
            return ValidationResult(passed=True, step="semantic", sql=sql)

        try:
            for stmt in ctx.ast:
                if stmt is None:
                    continue
                aliases = _alias_map(stmt)
                cte_bodies = {
                    (cte.alias or "").lower(): cte.this
                    for cte in stmt.find_all(exp.CTE)
                    if cte.alias
                }

                for select_node in stmt.find_all(exp.Select):
                    having = select_node.args.get("having")

                    # ── Rule 4: JOIN ... ON TRUE against a multi-row relation ─
                    # Writing ON at all declares an intent to relate the two
                    # sides; asserting nothing produces a cross product whose
                    # every downstream aggregate is multiplied. A LATERAL is
                    # correlated through its body rather than its ON clause, and
                    # a provably single-row relation is a scalar broadcast — both
                    # are legitimate and excluded.
                    for join in select_node.args.get("joins") or []:
                        if not _is_constant_true(join.args.get("on")):
                            continue
                        if isinstance(join.this, exp.Lateral) or join.args.get("lateral"):
                            continue
                        relation = _resolve_relation(join.this, cte_bodies)
                        if _is_provably_single_row(relation):
                            continue
                        target = (
                            join.this.alias_or_name
                            if hasattr(join.this, "alias_or_name") else "the joined relation"
                        )
                        logger.warning(
                            component="sql_validator",
                            event="constant_true_join_predicate",
                            target=target,
                            sql_preview=sql[:120],
                        )
                        return ValidationResult(
                            passed=False, step="safety",
                            message=(
                                f"`JOIN {target} ON TRUE` asserts no "
                                f"relationship, so every row of {target} is "
                                f"paired with every row on the other side. The "
                                f"result is a cross product: counts and averages "
                                f"downstream are multiplied by the size of "
                                f"{target}, and any column projected from the "
                                f"other side is unrelated to the row it appears "
                                f"beside. State the key that connects the two "
                                f"sides in the ON clause, or — if they are "
                                f"genuinely independent facts about the same "
                                f"entity — aggregate each one separately and "
                                f"join the aggregates on that entity's key."
                            ),
                            sql=sql,
                        )

                    # ── Rule 5: outer-join nullability contradiction ─────────
                    # Within one inner-join group either every member is present
                    # or none is. A predicate that rejects NULL on one member
                    # therefore proves the whole group present, which
                    # contradicts an IS NULL test on a NOT NULL column of any
                    # other member of that same group. The classic anti-join
                    # (LEFT JOIN + IS NULL) is unaffected: the outer side is its
                    # own group.
                    where = select_node.args.get("where")
                    contradiction = self._null_contradiction(
                        select_node, where, aliases, ctx.schema_map,
                    )
                    if contradiction is not None:
                        proof_alias, null_alias, null_column = contradiction
                        logger.warning(
                            component="sql_validator",
                            event="outer_join_nullability_contradiction",
                            proof_alias=proof_alias,
                            null_alias=null_alias,
                            null_column=null_column,
                            sql_preview=sql[:120],
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                f"`{null_alias}.{null_column} IS NULL` can never "
                                f"be true here. `{null_column}` is NOT NULL, so "
                                f"the only way it reads as NULL is a "
                                f"null-extended outer join — but "
                                f"`{null_alias}` is INNER-joined to "
                                f"`{proof_alias}`, and the WHERE clause has a "
                                f"predicate on `{proof_alias}` that no "
                                f"null-extended row can satisfy. The two "
                                f"together force the row to be both present and "
                                f"absent, and the query returns nothing on any "
                                f"data. If an anti-join was intended, LEFT JOIN "
                                f"the table being tested for absence and put no "
                                f"other WHERE predicate on its group."
                            ),
                            sql=sql,
                        )

                    # ── Rule 6: ratio of an expression to itself ─────────────
                    tautological = self._self_ratio(select_node)
                    if tautological is not None:
                        pred = tautological.sql(dialect="postgres")
                        logger.warning(
                            component="sql_validator",
                            event="self_identical_ratio",
                            expression=pred[:160],
                            sql_preview=sql[:120],
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                f"`{pred}` divides an expression by itself, so "
                                f"it evaluates to the same constant on every "
                                f"dataset regardless of the data. A percentage "
                                f"needs a numerator that counts the SUBSET being "
                                f"measured and a denominator that counts the "
                                f"whole population — for example "
                                f"`COUNT(*) FILTER (WHERE <condition>) * 100.0 / "
                                f"NULLIF(COUNT(*), 0)`. State the condition that "
                                f"distinguishes the subset."
                            ),
                            sql=sql,
                        )

                    # ── Rule 1: unique GROUP BY + COUNT(*) > 1 ──────────────
                    if having is not None:
                        unique, key_desc = _group_is_row_unique(
                            select_node, aliases, ctx.schema_map,
                        )
                        if unique:
                            offender = _count_star_lower_bound(having)
                            if offender is not None:
                                pred = offender.sql(dialect="postgres")
                                logger.warning(
                                    component="sql_validator",
                                    event="unsatisfiable_group_predicate",
                                    predicate=pred,
                                    unique_key=key_desc,
                                    sql_preview=sql[:120],
                                )
                                return ValidationResult(
                                    passed=False, step="semantic",
                                    message=(
                                        f"`{pred}` can never be true. This scope "
                                        f"groups by {key_desc}, which uniquely "
                                        f"identifies a row, so every group holds "
                                        f"exactly one row and COUNT(*) is always "
                                        f"1. The query returns an empty result on "
                                        f"any data. If you meant to find records "
                                        f"changed more than once, count the "
                                        f"history/version evidence instead — e.g. "
                                        f"a version column, or rows in a related "
                                        f"history table — rather than grouping by "
                                        f"the key itself."
                                    ),
                                    sql=sql,
                                )

                    # ── Rule 2: always-true disjunction ─────────────────────
                    for clause_key in ("having", "where"):
                        clause = select_node.args.get(clause_key)
                        if clause is None:
                            continue
                        taut = _find_range_tautology(clause)
                        if taut is not None:
                            pred = taut.sql(dialect="postgres")
                            logger.warning(
                                component="sql_validator",
                                event="tautological_predicate",
                                clause=clause_key.upper(),
                                predicate=pred,
                                sql_preview=sql[:120],
                            )
                            return ValidationResult(
                                passed=False, step="semantic",
                                message=(
                                    f"`{pred}` in the {clause_key.upper()} clause "
                                    f"is always true — the two branches together "
                                    f"cover every possible value, so nothing is "
                                    f"filtered out. Drop the clause if no filter "
                                    f"was intended, or state the single condition "
                                    f"you actually want to keep."
                                ),
                                sql=sql,
                            )

                    # ── Rule 3: contradictory equality conjuncts ────────────
                    for clause_key in ("having", "where"):
                        clause = select_node.args.get(clause_key)
                        if clause is None:
                            continue
                        clash = _find_equality_contradiction(clause)
                        if clash is None:
                            continue
                        column, first, second = clash
                        logger.warning(
                            component="sql_validator",
                            event="contradictory_equality_predicate",
                            clause=clause_key.upper(),
                            column=column,
                            first=first,
                            second=second,
                            sql_preview=sql[:120],
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                f"`{first}` and `{second}` are ANDed together "
                                f"in the {clause_key.upper()} clause and cannot "
                                f"both hold: a row carries ONE value for "
                                f"`{column}`. The query returns an empty result "
                                f"on any data, which reads like a true finding of "
                                f"'none'. If the question names several "
                                f"acceptable values, use IN (...) or OR; if it "
                                f"names one, keep only that predicate."
                            ),
                            sql=sql,
                        )
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="satisfiability_check_error",
                error=str(exc),
                note="check skipped due to AST error",
            )

        return ValidationResult(passed=True, step="semantic", sql=sql)

    @staticmethod
    def _null_contradiction(
        select_node: exp.Select,
        where: exp.Expression | None,
        aliases: dict[str, str],
        schema_map: dict,
    ) -> tuple[str, str, str] | None:
        """(alias_proving_presence, alias_asserted_null, column) or None."""
        null_assertions = _null_asserted_columns(where)
        if not null_assertions:
            return None
        proofs = _null_rejecting_aliases(where)
        if not proofs:
            return None

        groups = _inner_join_groups(select_node)
        for null_alias, null_column in null_assertions:
            table = aliases.get(null_alias)
            inventory = (schema_map or {}).get(table) if table else None
            if inventory is None:
                continue
            column = (getattr(inventory, "columns", {}) or {}).get(null_column)
            if column is None:
                continue
            # Only a NOT NULL column makes the contradiction decidable: a
            # nullable column can read NULL for reasons unrelated to joining.
            if getattr(column, "nullable", True) and not getattr(column, "is_pk", False):
                continue
            for group in groups:
                if null_alias not in group:
                    continue
                for proof_alias in sorted(proofs & group):
                    if proof_alias != null_alias:
                        return proof_alias, null_alias, null_column
        return None

    @staticmethod
    def _self_ratio(select_node: exp.Select) -> exp.Expression | None:
        """A division whose numerator and denominator are the same expression."""
        for projection in select_node.expressions:
            for div in projection.find_all(exp.Div):
                numerator = _strip_scalar_factors(div.this)
                denominator = _strip_scalar_factors(div.expression)
                if numerator is None or denominator is None:
                    continue
                left = numerator.sql(dialect="postgres").lower()
                right = denominator.sql(dialect="postgres").lower()
                # A bare literal ratio is arithmetic, not a defect.
                if isinstance(numerator, exp.Literal):
                    continue
                if left and left == right:
                    return div
        return None
