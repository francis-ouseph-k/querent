import sqlglot.expressions as exp
from ..core.context import ValidationContext
from models.schema import ValidationResult
from utils.logging_config import get_logger
from validation.utils.blocklist import COLUMN_BLOCKLIST as _COLUMN_BLOCKLIST

logger = get_logger(__name__)

def validate_columns(ctx: ValidationContext) -> ValidationResult | None:
    """
    Performs column-level existence checks by walking column nodes in the AST.
    """
    col_errors: list[str] = []
    sql = ctx.working_sql or ctx.sql
    if ctx.ast is None:
        return None

    try:
        for stmt in ctx.ast:
            if stmt is None:
                continue

            cte_names = set()
            for cte in stmt.find_all(exp.CTE):
                if cte.alias:
                    cte_names.add(cte.alias.lower())

            derived_table_aliases = set()
            for subq in stmt.find_all(exp.Subquery):
                alias = subq.alias
                if alias:
                    derived_table_aliases.add(alias.lower())

            projection_aliases = set()
            for a in stmt.find_all(exp.Alias):
                if a.alias:
                    projection_aliases.add(a.alias.lower())

            inner_subquery_col_ids: set[int] = set()
            for subq in stmt.find_all(exp.Subquery):
                if subq.alias:
                    for inner_col in subq.find_all(exp.Column):
                        inner_subquery_col_ids.add(id(inner_col))

            for col_node in stmt.find_all(exp.Column):
                col_name = (col_node.name or "").lower()
                tbl_part = (col_node.table or "").lower()

                if not col_name or col_name == "*":
                    continue
                if id(col_node) in inner_subquery_col_ids:
                    continue
                if tbl_part in cte_names:
                    continue
                if tbl_part in derived_table_aliases:
                    continue
                if not tbl_part and col_name in projection_aliases:
                    continue

                resolved_table: str | None = None
                if tbl_part:
                    resolved_table = ctx.alias_map.get(tbl_part)
                    if not resolved_table and tbl_part in ctx.sql_tables:
                        resolved_table = tbl_part
                    
                    if not resolved_table:
                        return ValidationResult(
                            passed=False, step="schema",
                            message=f"Unknown table or alias '{tbl_part}' referenced in '{tbl_part}.{col_name}'. Ensure it is declared in the FROM/JOIN clause.",
                            sql=sql
                        )
                elif len(ctx.sql_tables - cte_names) == 1:
                    remaining_real_tables = list(ctx.sql_tables - cte_names)
                    resolved_table = remaining_real_tables[0]

                if resolved_table and (resolved_table, col_name) in _COLUMN_BLOCKLIST:
                    return ValidationResult(
                        passed=False,
                        step="schema",
                        message=f"Column validation failed: {_COLUMN_BLOCKLIST[(resolved_table, col_name)]}",
                        sql=sql,
                    )

                if resolved_table and resolved_table in ctx.schema_map:
                    inv = ctx.schema_map[resolved_table]
                    if hasattr(inv, "columns") and col_name not in inv.columns:
                        col_errors.append(f"{resolved_table}.{col_name}")
                elif not resolved_table:
                    real_tables = ctx.sql_tables - cte_names
                    possible_tables = []
                    for t in real_tables:
                        if t in ctx.schema_map and hasattr(ctx.schema_map[t], "columns") and col_name in ctx.schema_map[t].columns:
                            possible_tables.append(t)
                    if len(possible_tables) > 1:
                        return ValidationResult(
                            passed=False, step="schema",
                            message=f"Ambiguous column reference: '{col_name}'. It exists in multiple tables ({', '.join(possible_tables)}). You must qualify it with a table alias.",
                            sql=sql
                        )
                    elif len(possible_tables) == 0 and len(real_tables) > 0:
                        col_errors.append(f"unaliased.{col_name}")

    except Exception as exc:
        logger.warning(
            component="sql_validator",
            event="column_check_error",
            error=str(exc),
            note="Column hallucination check skipped due to AST error",
        )

    if col_errors:
        return ValidationResult(
            passed=False,
            step="schema",
            message=f"Hallucinated column(s): {', '.join(col_errors[:5])}. "
                    f"Use only columns that exist in the schema.",
            sql=sql,
        )

    return None
