"""
validation/semantic/reference_data.py
─────────────────────────────────────
ReferenceDataValidator — rejects a query whose literals contradict a fact the
DDL's own seed rows already state.

THE DEFECT

    -- "active cross-listing relationships between courses and departments"
    JOIN academic_unit dept ON dept.id = aur.from_unit_id
                           AND dept.unit_type = 'DEPARTMENT'
    WHERE aur.relationship_type = 'CROSS_LISTING'

The seeded catalog row is
`('CROSS_LISTING', ..., from_unit_type='COURSE', to_unit_type='DEPARTMENT')`.
The edge runs COURSE -> DEPARTMENT; the query asserts the reverse. The model
read the ENGLISH ORDER OF THE QUESTION as the COLUMN ORDER of the
relationship.

Nothing else in the pipeline can see this. Both tables exist, both columns
exist, both join domains are `academic_unit`, both literals are legitimate
catalog values, and EXPLAIN costs pennies. The query returns zero rows on any
database and reads as a genuine "there are none".

WHY THIS IS A GENERAL MECHANISM AND NOT A RULE ABOUT THIS SCHEMA

No table name, column name or value appears below. The link is reconstructed
from foreign keys alone:

    S.<catalog_col>   -> C            (S rows are typed by a catalog row)
    C.<role_col>      -> V            (the catalog states a kind, drawn from V)
    E.<discriminator> -> V            (the entity's own kind, from the SAME V)
    S.<entity_col>    -> E            (which entity fills that role)

When a query pins the catalog key AND pins the entity's discriminator, the
seed row says what the discriminator must be. Sharing vocabulary V is what
makes the two values comparable at all, and is the reason this cannot fire on
two unrelated string columns.

The single heuristic is which role column governs which entity column
(`from_unit_type` <-> `from_unit_id`), resolved by longest shared name prefix
with a three-character floor. On a tie, or with no shared prefix, the pairing
is abandoned and the check stays silent — a wrong pairing here would reject
correct SQL, which is strictly worse than missing a defect.
"""

from __future__ import annotations

import sqlglot.expressions as exp

from ..core.base import BaseValidationStep
from ..core.context import ValidationContext
from ..utils.roles import discriminator_columns_for, pinned_roles
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

_MIN_PREFIX = 3


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


def _shared_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _fk_targets(inventory) -> dict[str, str]:
    """{from_column: to_table} for every declared FK, lowercased."""
    return {
        (fk.from_col or "").lower(): (fk.to_table or "").lower()
        for fk in (getattr(inventory, "foreign_keys", []) or [])
        if fk.from_col and fk.to_table
    }


def _catalog_links(schema_map: dict) -> list[dict]:
    """
    Every (subject, catalog, role_col, entity_col, vocabulary) link the FK
    graph supports. Computed once per validation; the schema is small and this
    keeps the check stateless.
    """
    links: list[dict] = []

    for subject_name, subject in (schema_map or {}).items():
        subject_fks = _fk_targets(subject)
        for catalog_col, catalog_name in subject_fks.items():
            catalog = (schema_map or {}).get(catalog_name)
            if catalog is None or catalog_name == subject_name:
                continue

            # The catalog key must be that table's whole primary key, or a
            # "type" column on it does not describe one row.
            catalog_pk = {
                n.lower()
                for n, c in (getattr(catalog, "columns", {}) or {}).items()
                if getattr(c, "is_pk", False)
            }
            if len(catalog_pk) != 1:
                continue

            catalog_fks = _fk_targets(catalog)
            for role_col, vocabulary in catalog_fks.items():
                # Candidate entity columns on the subject whose target table
                # is discriminated by the SAME vocabulary.
                for entity_col, entity_name in subject_fks.items():
                    if entity_col == catalog_col:
                        continue
                    entity = (schema_map or {}).get(entity_name)
                    if entity is None:
                        continue
                    entity_fks = _fk_targets(entity)
                    discriminators = {
                        d for d in discriminator_columns_for(entity, schema_map)
                        if entity_fks.get(d) == vocabulary
                    }
                    if not discriminators:
                        continue
                    links.append({
                        "subject": subject_name,
                        "catalog": catalog_name,
                        "catalog_col": catalog_col,
                        "catalog_key": next(iter(catalog_pk)),
                        "role_col": role_col,
                        "entity": entity_name,
                        "entity_col": entity_col,
                        "discriminators": discriminators,
                        "prefix": _shared_prefix_len(role_col, entity_col),
                    })
    return links


def _pair_roles_to_entities(links: list[dict]) -> list[dict]:
    """
    Keep, for each (subject, catalog, entity_col), the single role column with
    the longest shared name prefix. Ties and sub-threshold prefixes are
    dropped: an unresolved pairing must not drive a rejection.
    """
    best: dict[tuple[str, str, str], list[dict]] = {}
    for link in links:
        if link["prefix"] < _MIN_PREFIX:
            continue
        key = (link["subject"], link["catalog"], link["entity_col"])
        current = best.get(key)
        if current is None or link["prefix"] > current[0]["prefix"]:
            best[key] = [link]
        elif link["prefix"] == current[0]["prefix"]:
            current.append(link)
    return [group[0] for group in best.values() if len(group) == 1]


def _joined_alias_for(
    select_node: exp.Select, alias: str, column: str,
) -> list[tuple[str, str]]:
    """
    Aliases equated with `alias.column` anywhere in this scope's ON/WHERE
    clauses, as (other_alias, other_column).
    """
    out: list[tuple[str, str]] = []
    sources = [select_node.args.get("where")]
    sources.extend(join.args.get("on") for join in (select_node.args.get("joins") or []))
    for source in sources:
        if source is None:
            continue
        for eq in source.find_all(exp.EQ):
            left, right = eq.left, eq.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            for near, far in ((left, right), (right, left)):
                if (
                    near.table
                    and near.table.lower() == alias
                    and (near.name or "").lower() == column
                    and far.table
                ):
                    out.append((far.table.lower(), (far.name or "").lower()))
    return out


class ReferenceDataValidator(BaseValidationStep):
    """Rejects literal combinations the DDL's seeded catalog rows contradict."""

    name = "ReferenceDataValidator"

    def __init__(self, seed_index: dict[str, list[dict[str, str]]] | None = None):
        self.seed_index = seed_index or {}

    def run(self, ctx: ValidationContext) -> ValidationResult:
        sql = ctx.working_sql or ctx.sql
        if not ctx.ast or not ctx.schema_map or not self.seed_index:
            return ValidationResult(passed=True, step="semantic", sql=sql)

        try:
            pairings = _pair_roles_to_entities(_catalog_links(ctx.schema_map))
            if not pairings:
                return ValidationResult(passed=True, step="semantic", sql=sql)

            for stmt in ctx.ast:
                if stmt is None:
                    continue
                alias_map = _alias_map(stmt)

                for select_node in stmt.find_all(exp.Select):
                    roles = pinned_roles(select_node, alias_map, ctx.schema_map)
                    conflict = self._check_scope(
                        select_node, alias_map, roles, pairings, ctx.schema_map,
                    )
                    if conflict is None:
                        continue

                    subject_alias, link, pinned, expected, catalog_value = conflict
                    logger.warning(
                        component="sql_validator",
                        event="reference_data_contradiction",
                        catalog=link["catalog"],
                        catalog_value=catalog_value,
                        role_col=link["role_col"],
                        expected=expected,
                        entity_col=link["entity_col"],
                        pinned=pinned,
                        sql_preview=sql[:120],
                    )
                    return ValidationResult(
                        passed=False, step="semantic",
                        message=(
                            f"The query pins "
                            f"`{subject_alias}.{link['catalog_col']} = "
                            f"'{catalog_value}'` and then requires the row "
                            f"reached through `{link['entity_col']}` to be "
                            f"'{pinned}'. The seeded {link['catalog']} row for "
                            f"'{catalog_value}' states "
                            f"{link['role_col']} = '{expected}', so that side "
                            f"is always a '{expected}' and the query returns "
                            f"zero rows on any data — which reads like a true "
                            f"finding of 'none'. The relationship direction is "
                            f"reversed: swap the two sides, or filter on "
                            f"'{expected}' here."
                        ),
                        sql=sql,
                    )
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="reference_data_check_error",
                error=f"{type(exc).__name__}: {exc}",
                note="check skipped due to AST error",
            )

        return ValidationResult(passed=True, step="semantic", sql=sql)

    def _check_scope(
        self,
        select_node: exp.Select,
        alias_map: dict[str, str],
        roles: dict[str, str],
        pairings: list[dict],
        schema_map: dict,
    ):
        for link in pairings:
            seed_rows = self.seed_index.get(link["catalog"]) or []
            if not seed_rows:
                continue

            for subject_alias, subject_table in alias_map.items():
                if subject_table != link["subject"]:
                    continue

                catalog_value = self._pinned_value(
                    select_node, subject_alias, link["catalog_col"],
                )
                if catalog_value is None:
                    continue

                expected = None
                for row in seed_rows:
                    if row.get(link["catalog_key"]) == catalog_value:
                        expected = row.get(link["role_col"])
                        break
                if not expected:
                    continue

                for other_alias, other_col in _joined_alias_for(
                    select_node, subject_alias, link["entity_col"],
                ):
                    if alias_map.get(other_alias) != link["entity"]:
                        continue
                    entity = schema_map.get(link["entity"])
                    key_columns = {
                        n.lower()
                        for n, c in (getattr(entity, "columns", {}) or {}).items()
                        if getattr(c, "is_pk", False)
                    }
                    if other_col not in key_columns:
                        continue
                    pinned = roles.get(other_alias)
                    if pinned is None or pinned == expected.upper():
                        continue
                    return subject_alias, link, pinned, expected.upper(), catalog_value
        return None

    @staticmethod
    def _pinned_value(select_node: exp.Select, alias: str, column: str) -> str | None:
        """The single string literal `alias.column` is pinned to in this scope."""
        found: set[str] = set()
        sources = [select_node.args.get("where")]
        sources.extend(
            join.args.get("on") for join in (select_node.args.get("joins") or [])
        )
        for source in sources:
            if source is None:
                continue
            for eq in source.find_all(exp.EQ):
                for column_side, value_side in (
                    (eq.left, eq.right), (eq.right, eq.left),
                ):
                    if not isinstance(column_side, exp.Column):
                        continue
                    if not column_side.table or column_side.table.lower() != alias:
                        continue
                    if (column_side.name or "").lower() != column:
                        continue
                    if isinstance(value_side, exp.Literal) and value_side.is_string:
                        found.add(str(value_side.this))
        return next(iter(found)) if len(found) == 1 else None
