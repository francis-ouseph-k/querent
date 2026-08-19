"""
validation/semantic/closure_semantics.py
────────────────────────────────────────
ClosureSemanticsValidator — a transitive-closure table contains a row from
every node to ITSELF, so membership alone proves nothing.

THE DEFECT

    -- "Count academic units by whether they have children."
    LEFT JOIN academic_unit_closure auc ON auc.ancestor_id = au.id
    ...
    CASE WHEN uc.child_count > 0 THEN 'HAS_CHILDREN' ELSE 'NO_CHILDREN' END

Every node is its own ancestor at depth 0, so `child_count` is at least 1 for
every row and the CASE has one reachable branch. The query reported all seven
seeded units as HAS_CHILDREN — including three leaf courses.

The same shape appears as `EXISTS (SELECT 1 FROM <closure> WHERE ancestor_id =
X)`, which is a tautology for any X that exists at all.

This is neither a join defect nor an unsatisfiable predicate: the join is
legal, the predicate is satisfiable, and the query returns rows. It is a fact
about what a closure table stores, and it is decidable from the schema.

RECOGNISING A CLOSURE TABLE WITHOUT NAMING ONE

A table qualifies when it has exactly two foreign key columns, both targeting
the SAME table, both part of its primary key, plus at least one integer column
outside that key. That is the closure/adjacency-with-distance shape and
nothing else has it: a plain junction table between two roles of one entity
carries no distance column, and a table with a distance column but only one
self-FK is not a closure.

WHEN THIS FIRES

Only when the closure table is used to DERIVE a related set — a JOIN or a
subquery — and the depth column is referenced nowhere in the statement.
Referencing it at all is taken as evidence the author knows self-rows exist,
whether they filter (`depth > 0`), project it, or aggregate it. Selecting
straight out of the closure table with no join is left alone: `SELECT
MAX(depth) FROM <closure>` is a legitimate question about the closure itself.
"""

from __future__ import annotations

import sqlglot.expressions as exp

from ..core.base import BaseValidationStep
from ..core.context import ValidationContext
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

_INTEGER_TYPES = ("int", "serial", "numeric", "decimal", "smallint", "bigint")


def closure_tables(schema_map: dict) -> dict[str, set[str]]:
    """
    {table_name: {distance_column, ...}} for every closure-shaped table.

    See the module docstring for what "closure-shaped" means; the test is
    structural and mentions no table or column name.
    """
    out: dict[str, set[str]] = {}
    for name, inventory in (schema_map or {}).items():
        fks = [
            fk for fk in (getattr(inventory, "foreign_keys", []) or [])
            if fk.from_col and fk.to_table
        ]
        if len(fks) != 2:
            continue
        targets = {(fk.to_table or "").lower() for fk in fks}
        if len(targets) != 1:
            continue

        columns = getattr(inventory, "columns", {}) or {}
        key_columns = {
            n.lower() for n, c in columns.items() if getattr(c, "is_pk", False)
        }
        role_columns = {(fk.from_col or "").lower() for fk in fks}
        if not role_columns.issubset(key_columns):
            continue

        distance = {
            n.lower() for n, c in columns.items()
            if n.lower() not in key_columns
            and any(t in (getattr(c, "data_type", "") or "").lower() for t in _INTEGER_TYPES)
        }
        if distance:
            out[name.lower()] = distance
    return out


class ClosureSemanticsValidator(BaseValidationStep):
    """Rejects closure-table membership tests that ignore the self-row."""

    name = "ClosureSemanticsValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        sql = ctx.working_sql or ctx.sql
        if not ctx.ast or not ctx.schema_map:
            return ValidationResult(passed=True, step="semantic", sql=sql)

        try:
            shapes = closure_tables(ctx.schema_map)
            if not shapes:
                return ValidationResult(passed=True, step="semantic", sql=sql)

            for stmt in ctx.ast:
                if stmt is None:
                    continue

                referenced_columns = {
                    (col.name or "").lower() for col in stmt.find_all(exp.Column)
                }

                for select_node in stmt.find_all(exp.Select):
                    for join in select_node.args.get("joins") or []:
                        table = self._joined_table(join)
                        if table is None or table not in shapes:
                            continue
                        distance = shapes[table]
                        if distance & referenced_columns:
                            continue
                        column_name = sorted(distance)[0]

                        logger.warning(
                            component="sql_validator",
                            event="closure_self_row_unfiltered",
                            table=table,
                            distance_column=column_name,
                            sql_preview=sql[:120],
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                f"`{table}` is a transitive-closure table: it "
                                f"holds a row from every node to ITSELF at "
                                f"{column_name} = 0. This query joins it "
                                f"without constraining `{column_name}`, so "
                                f"every node matches itself and the test is "
                                f"true for every row — a descendant count is "
                                f"never 0 and an EXISTS check never fails. Add "
                                f"`{column_name} > 0` to exclude the self-row, "
                                f"or reference `{column_name}` explicitly if "
                                f"the self-row is genuinely wanted."
                            ),
                            sql=sql,
                        )
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="closure_semantics_check_error",
                error=f"{type(exc).__name__}: {exc}",
                note="check skipped due to AST error",
            )

        return ValidationResult(passed=True, step="semantic", sql=sql)

    @staticmethod
    def _joined_table(join: exp.Join) -> str | None:
        target = join.this
        if isinstance(target, exp.Table) and target.name:
            return target.name.lower()
        return None
