"""
validation/ast/joins.py
───────────────────────
JoinValidator (pipeline step 5, reports `safety`).

Catches Cartesian products — a JOIN with no ON/USING clause — which silently
multiply rows and corrupt every downstream aggregate. It is FK-graph aware: the
graph loaded at bootstrap lets it reason about whether a join path is legitimate
rather than only checking for a missing ON clause. Reported under the `safety`
label because an accidental cross join is a correctness hazard, not a cosmetic
issue.
"""

import re
import sqlglot.errors
import sqlglot.expressions as exp
from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Cartesian regex — fallback only for unparseable SQL
CARTESIAN_PATTERN = re.compile(r"FROM\s+\w+\s*,\s*\w+", re.IGNORECASE)

def _cte_projection_sources(stmt: exp.Expression) -> dict[str, dict[str, tuple[str, str]]]:
    """
    Map each CTE's output columns back to the base table column they select.

    Returns {cte_name: {output_column: (source_alias, source_column)}}. Only
    plain column projections are recorded — an expression, aggregate, or literal
    has no single source column and is deliberately left unmapped so the caller
    treats it as unknown.

    This exists because the join defects worth catching are usually written
    against a CTE, not against a base table, so a check that only understands
    base-table columns sees nothing.
    """
    out: dict[str, dict[str, tuple[str, str]]] = {}
    for cte in stmt.find_all(exp.CTE):
        cte_name = (cte.alias or "").lower()
        if not cte_name:
            continue
        inner = cte.this
        if not isinstance(inner, exp.Select):
            continue
        mapping: dict[str, tuple[str, str]] = {}
        for projection in inner.expressions:
            target = projection
            output_name = projection.alias_or_name
            if isinstance(projection, exp.Alias):
                target = projection.this
            if not isinstance(target, exp.Column) or not target.table:
                continue
            if not output_name:
                continue
            mapping[output_name.lower()] = (
                target.table.lower(), (target.name or "").lower()
            )
        out[cte_name] = mapping
    return out


def _derived_table_sources(
    stmt: exp.Expression,
) -> dict[str, dict[str, tuple[str, str]]]:
    """
    Same contract as _cte_projection_sources, for INLINE derived tables.

    `JOIN (SELECT ea.assignment_id, ... FROM evaluation_attempt ea) AS ed
        ON ed.assignment_id = ascr.id`

    is the identical defect to the CTE form and was invisible to this check,
    because a subquery alias is not an exp.CTE. Batch run 20260814_102207:
    join_key_domain_mismatch fired only 4 times while hand review found the
    same defect shape in queries the check never saw.
    """
    out: dict[str, dict[str, tuple[str, str]]] = {}
    for sub in stmt.find_all(exp.Subquery):
        alias = (sub.alias or "").lower()
        if not alias:
            continue
        inner = sub.this
        if not isinstance(inner, exp.Select):
            continue
        mapping: dict[str, tuple[str, str]] = {}
        for projection in inner.expressions:
            target = projection
            output_name = projection.alias_or_name
            if isinstance(projection, exp.Alias):
                target = projection.this
            if not isinstance(target, exp.Column) or not target.table:
                continue
            if not output_name:
                continue
            mapping[output_name.lower()] = (
                target.table.lower(), (target.name or "").lower()
            )
        out[alias] = mapping
    return out


def _local_alias_map(stmt: exp.Expression) -> dict[str, str]:
    """alias -> base table name for every table source in the statement."""
    alias_map: dict[str, str] = {}
    for table in stmt.find_all(exp.Table):
        if table.name:
            alias_map[(table.alias or table.name).lower()] = table.name.lower()
    return alias_map


def _referent_entity(
    alias: str,
    column: str,
    alias_map: dict[str, str],
    cte_sources: dict[str, dict[str, tuple[str, str]]],
    schema_map: dict,
    _depth: int = 0,
) -> str | None:
    """
    Resolve which ENTITY a column identifies, or None when it cannot be known.

    A column identifies an entity in one of two ways: it is a foreign key, in
    which case it identifies rows of the table it references; or it is the
    table's own primary key, in which case it identifies rows of that table.
    Anything else — a measure, a status, a polymorphic id with no declared FK —
    returns None, and the caller stays silent. None means "unknown", never
    "mismatch": a check that guesses here would fire on legitimate joins.

    CTE columns are resolved one hop at a time back to their source column, so a
    join written against a CTE is judged on the base column it came from.
    """
    if _depth > 4:
        return None

    # A CTE is referenced at the join site by its local alias ('ae'), while
    # cte_sources is keyed by the CTE's declared name ('alg_exam'). Resolve the
    # alias before looking it up, or every aliased CTE reads as unknown.
    cte_key = alias if alias in cte_sources else alias_map.get(alias, alias)

    if cte_key in cte_sources:
        source = cte_sources[cte_key].get(column)
        if source is None:
            return None
        return _referent_entity(
            source[0], source[1], alias_map, cte_sources, schema_map, _depth + 1
        )

    table = alias_map.get(alias)
    if not table or table not in schema_map:
        return None
    inventory = schema_map[table]

    for fk in getattr(inventory, "foreign_keys", []) or []:
        if (fk.from_col or "").lower() == column:
            return (fk.to_table or "").lower()

    col_info = getattr(inventory, "columns", {}).get(column)
    if col_info is not None and getattr(col_info, "is_pk", False):
        return table

    return None


class JoinValidator(BaseValidationStep):
    name = "JoinValidator"

    def __init__(self, fk_graph):
        self.fk_graph = fk_graph

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 3b: Cartesian Join Check
        Detects implicit Cartesian joins (e.g., joins that lack 'ON' or 'USING' conditions).
        """
        sql = ctx.working_sql or ctx.sql
        cartesian_detected = False

        if ctx.ast:
            try:
                for stmt in ctx.ast:
                    if stmt is None:
                        continue
                    for join in stmt.find_all(exp.Join):
                        has_on = join.args.get("on") is not None
                        has_using = join.args.get("using") is not None
                        join_kind = (join.args.get("kind") or "").upper()
                        
                        if not has_on and not has_using and join_kind != "CROSS":
                            cartesian_detected = True
                            break
                    if cartesian_detected:
                        break
            except Exception as exc:
                logger.warning("cartesian_check_ast_error", error=str(exc))
                cartesian_detected = bool(CARTESIAN_PATTERN.search(sql))
        else:
            cartesian_detected = bool(CARTESIAN_PATTERN.search(sql))

        if cartesian_detected:
            return ValidationResult(
                passed=False, step="safety",
                message="Cartesian join detected (JOIN without ON or USING clause). "
                        "Use explicit JOIN ... ON syntax.",
                sql=sql,
            )

        mismatch = self._check_join_key_domains(ctx)
        if mismatch is not None:
            return mismatch

        return ValidationResult(passed=True, step="safety", sql=sql)

    def _check_join_key_domains(self, ctx: ValidationContext) -> ValidationResult | None:
        """
        Reject a join that equates two columns identifying DIFFERENT entities.

        `ON ep.board_id = ae.exam_id` parses, type-checks, and executes — both
        sides are integers — but a board id and an exam id are different
        domains, so the predicate matches rows that have nothing to do with one
        another. Every column reference exists, so schema validation passes; the
        query returns a plausible-looking wrong answer.

        The FK graph makes this decidable rather than a guess: a column that is
        a declared foreign key identifies rows of its target table, and a
        primary key identifies rows of its own table. When both sides of a join
        equality resolve to a known entity and those entities differ, the join
        is wrong. When either side is unknown — a polymorphic id with no FK, a
        computed CTE column — the check stays silent.

        This is what the FK graph was passed to this validator for; until now it
        was stored and never consulted.
        """
        if not ctx.ast or not ctx.schema_map:
            return None

        try:
            for stmt in ctx.ast:
                if stmt is None:
                    continue

                alias_map = _local_alias_map(stmt)
                cte_sources = _cte_projection_sources(stmt)
                # Derived tables resolve exactly like CTEs; merging them into
                # one map means _referent_entity needs no change. CTE names win
                # on collision — a CTE is declared once, a subquery alias is
                # local, so the CTE is the more reliable binding.
                merged_sources = {**_derived_table_sources(stmt), **cte_sources}
                cte_sources = merged_sources

                for join in stmt.find_all(exp.Join):
                    on_clause = join.args.get("on")
                    if on_clause is None:
                        continue

                    for eq in on_clause.find_all(exp.EQ):
                        left, right = eq.left, eq.right
                        if not isinstance(left, exp.Column):
                            continue
                        if not isinstance(right, exp.Column):
                            continue
                        if not left.table or not right.table:
                            continue

                        left_alias = left.table.lower()
                        right_alias = right.table.lower()
                        left_col = (left.name or "").lower()
                        right_col = (right.name or "").lower()

                        left_entity = _referent_entity(
                            left_alias, left_col, alias_map, cte_sources, ctx.schema_map
                        )
                        if left_entity is None:
                            continue
                        right_entity = _referent_entity(
                            right_alias, right_col, alias_map, cte_sources, ctx.schema_map
                        )
                        if right_entity is None:
                            continue
                        if left_entity == right_entity:
                            continue

                        logger.warning(
                            component="sql_validator",
                            event="join_key_domain_mismatch",
                            left=f"{left_alias}.{left_col}",
                            left_entity=left_entity,
                            right=f"{right_alias}.{right_col}",
                            right_entity=right_entity,
                        )
                        return ValidationResult(
                            passed=False, step="safety",
                            message=(
                                f"Join key domain mismatch: "
                                f"'{left_alias}.{left_col}' identifies "
                                f"{left_entity} rows, but "
                                f"'{right_alias}.{right_col}' identifies "
                                f"{right_entity} rows. Equating them matches "
                                f"unrelated records. Join through the foreign "
                                f"key that actually connects "
                                f"{left_entity} and {right_entity}."
                            ),
                            sql=ctx.working_sql or ctx.sql,
                        )
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="join_domain_check_error",
                error=str(exc),
                note="Join key domain check skipped due to AST error",
            )

        return None