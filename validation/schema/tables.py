import sqlglot.expressions as exp
from ..core.context import ValidationContext
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

def validate_tables(ctx: ValidationContext) -> ValidationResult | None:
    """
    Validates that all tables referenced either in the pre-extracted `tables_used` list
    or within the SQL AST itself actually exist in the database schema inventory.
    Also constructs a mapping dictionary of table aliases to actual table names.
    """
    schema_map = ctx.schema_map
    tables_used = ctx.tables_used
    sql = ctx.working_sql or ctx.sql

    # Step A: Validate table names in the pre-extracted tables_used list.
    unknown_tables = [t for t in tables_used if t not in schema_map]
    if unknown_tables:
        return ValidationResult(
            passed=False,
            step="schema",
            message=f"Hallucinated table(s): {', '.join(unknown_tables)}. "
                    f"These tables do not exist in the schema.",
            sql=sql,
        )

    # Step B: Parse the SQL string into AST statements using sqlglot.
    if ctx.ast is None:
        return None

    sql_tables: set[str] = set()
    alias_map: dict[str, str] = {}
    cte_names: set[str] = set()

    for stmt in ctx.ast:
        if stmt is None:
            continue

        for cte in stmt.find_all(exp.CTE):
            if cte.alias:
                cte_names.add(cte.alias.lower())

        try:
            for tbl in stmt.find_all(exp.Table):
                if tbl.name:
                    name = tbl.name.lower()
                    if name not in cte_names:
                        sql_tables.add(name)

                    alias = (tbl.alias or "").lower()
                    if alias:
                        alias_map[alias] = name
                    alias_map[name] = name
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="ast_walk_error",
                error=str(exc),
                note="Partial schema grounding for this statement",
            )

    # Step C: Validate table names found inside the SQL AST statements.
    unknown_in_sql = [t for t in sql_tables if t not in schema_map]
    if unknown_in_sql:
        return ValidationResult(
            passed=False,
            step="schema",
            message=f"SQL references unknown table(s): {', '.join(unknown_in_sql)}. "
                    f"Use only tables that exist in the schema.",
            sql=sql,
        )

    # Update context with derived state
    ctx.sql_tables = sql_tables
    ctx.alias_map = alias_map
    ctx.cte_names = cte_names

    return None
