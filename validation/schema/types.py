"""
validation/schema/types.py
──────────────────────────
validate_types — the type/enum sub-check of Schema validation (step 4, reports
`schema`). Runs after tables and columns are confirmed to exist.

Checks that comparisons are type-sensible (e.g. not comparing a text column to an
integer literal) and, most usefully, that a literal compared against a column with
a CHECK constraint is one of the allowed values — so `status = 'ONGOING'` fails
when the enum only permits {'OPEN','CLOSED',...}. Enum membership is derived from
the CHECK constraints captured by the DDL parser, so it stays in sync with the
schema without a hand-maintained list.
"""

import sqlglot.expressions as exp
from ..core.context import ValidationContext
from models.schema import ValidationResult
from utils.logging_config import get_logger
from validation.utils.blocklist import (
    classify_pg_type as _classify_pg_type,
    check_literal_type_compat as _check_literal_type_compat,
)

logger = get_logger(__name__)

def validate_types(ctx: ValidationContext) -> ValidationResult | None:
    """
    Performs best-effort column data type and CHECK constraint enum checks.
    """
    type_errors: list[str] = []
    enum_errors: list[str] = []
    sql = ctx.working_sql or ctx.sql
    if ctx.ast is None:
        return None

    try:
        _comparison_types = (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ, exp.Is)

        for stmt in ctx.ast:
            if stmt is None:
                continue

            cte_names = set()
            for cte in stmt.find_all(exp.CTE):
                if cte.alias:
                    cte_names.add(cte.alias.lower())

            for cmp_node in stmt.find_all(*_comparison_types):
                left, right = cmp_node.left, cmp_node.right
                pairs: list[tuple[exp.Expression, exp.Expression]] = []
                
                if isinstance(left, exp.Column):
                    pairs.append((left, right))
                if isinstance(right, exp.Column):
                    pairs.append((right, left))

                for col_node, val_node in pairs:
                    col_name = (col_node.name or "").lower()
                    tbl_part = (col_node.table or "").lower()
                    if not col_name:
                        continue

                    resolved = ctx.alias_map.get(tbl_part) if tbl_part else (
                        list(ctx.sql_tables - cte_names)[0]
                        if len(ctx.sql_tables - cte_names) == 1 else None
                    )
                    if not resolved or resolved not in ctx.schema_map:
                        continue

                    col_info = ctx.schema_map[resolved].columns.get(col_name)
                    if col_info is None:
                        continue

                    if (
                        getattr(col_info, "allowed_values", None) is not None
                        and isinstance(cmp_node, (exp.EQ, exp.NEQ))
                        and isinstance(val_node, exp.Literal)
                        and val_node.is_string
                        and val_node.this not in col_info.allowed_values
                    ):
                        enum_errors.append(
                            f"{resolved}.{col_name} = '{val_node.this}' is not a "
                            f"valid value for this column. Allowed values: "
                            f"{', '.join(sorted(col_info.allowed_values))}"
                        )
                        continue

                    family = _classify_pg_type(col_info.data_type)
                    if family is None:
                        continue

                    err = _check_literal_type_compat(family, val_node)
                    if err:
                        type_errors.append(
                            f"{resolved}.{col_name} ({col_info.data_type}): {err}"
                        )

            for in_node in stmt.find_all(exp.In):
                col_node = in_node.this
                if not isinstance(col_node, exp.Column):
                    continue
                col_name = (col_node.name or "").lower()
                tbl_part = (col_node.table or "").lower()
                if not col_name:
                    continue

                resolved = ctx.alias_map.get(tbl_part) if tbl_part else (
                    list(ctx.sql_tables - cte_names)[0]
                    if len(ctx.sql_tables - cte_names) == 1 else None
                )
                if not resolved or resolved not in ctx.schema_map:
                    continue

                col_info = ctx.schema_map[resolved].columns.get(col_name)
                if col_info is None:
                    continue

                family = _classify_pg_type(col_info.data_type)
                if family is None:
                    continue

                for val_node in in_node.expressions:
                    err = _check_literal_type_compat(family, val_node)
                    if err:
                        type_errors.append(
                            f"{resolved}.{col_name} ({col_info.data_type}): {err}"
                        )

            for between_node in stmt.find_all(exp.Between):
                col_node = between_node.this
                if not isinstance(col_node, exp.Column):
                    continue
                col_name = (col_node.name or "").lower()
                tbl_part = (col_node.table or "").lower()
                if not col_name:
                    continue

                resolved = ctx.alias_map.get(tbl_part) if tbl_part else (
                    list(ctx.sql_tables - cte_names)[0]
                    if len(ctx.sql_tables - cte_names) == 1 else None
                )
                if not resolved or resolved not in ctx.schema_map:
                    continue

                col_info = ctx.schema_map[resolved].columns.get(col_name)
                if col_info is None:
                    continue

                family = _classify_pg_type(col_info.data_type)
                if family is None:
                    continue

                for bound in (between_node.args.get("low"), between_node.args.get("high")):
                    if bound is None:
                        continue
                    err = _check_literal_type_compat(family, bound)
                    if err:
                        type_errors.append(
                            f"{resolved}.{col_name} ({col_info.data_type}) "
                            f"BETWEEN bound: {err}"
                        )

    except Exception as exc:
        logger.warning(
            component="sql_validator",
            event="type_check_error",
            error=str(exc),
            note="Type-compatibility check skipped due to AST error",
        )

    if enum_errors:
        first = enum_errors[0]
        return ValidationResult(
            passed=False,
            step="schema",
            message=(
                f"Invalid value: {first}. "
                f"Tip — this column is constrained to a fixed set of values "
                f"by a CHECK constraint; only the listed values can ever exist "
                f"in the data."
                + (f" (and {len(enum_errors)-1} more)" if len(enum_errors) > 1 else "")
            ),
            sql=sql,
        )

    if type_errors:
        first = type_errors[0]
        return ValidationResult(
            passed=False,
            step="schema",
            message=(
                f"Type mismatch: {first}. "
                f"Tip — integer/bigint columns need unquoted numeric literals "
                f"(e.g. board_id = 5, not board_id = '5'). "
                f"If you are filtering by a human-readable label like a course "
                f"code or name, use the .code or .name VARCHAR column instead "
                f"of the numeric .id column."
                + (f" (and {len(type_errors)-1} more)" if len(type_errors) > 1 else "")
            ),
            sql=sql,
        )

    return None
