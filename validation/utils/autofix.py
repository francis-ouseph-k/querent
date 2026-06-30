import re
import sqlglot
import sqlglot.expressions as exp
from typing import Any
from utils.logging_config import get_logger

logger = get_logger(__name__)

_PG_COL_PERHAPS_RE = re.compile(
    r'column\s+"?([\w.]+)"?\s+does not exist.*?'
    r'Perhaps you meant to reference the column\s+"([^"]+)"',
    re.IGNORECASE | re.DOTALL,
)

def attempt_pg_autofix(
    sql: str,
    error_msg: str,
    schema_map: dict,
    run_explain: Any,
) -> tuple[str | None, str | None]:
    """
    Try a deterministic column-rename based on a PG planner hint.

    Returns (fixed_sql, fix_description) on success, or (None, None) on
    failure / no-applicable-hint.
    """
    m = _PG_COL_PERHAPS_RE.search(error_msg)
    if not m:
        return None, None
    bad_ref = m.group(1).strip().strip('"')
    hint    = m.group(2).strip().strip('"')

    if '.' not in bad_ref or '.' not in hint:
        logger.info(component="sql_validator", event="autofix_skipped_unqualified", bad=bad_ref, hint=hint)
        return None, None
    
    bad_tbl_or_alias, bad_col = bad_ref.lower().split('.', 1)
    good_tbl_or_alias, good_col = hint.lower().split('.', 1)

    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        logger.info(component="sql_validator", event="autofix_skipped_parse_error", bad=bad_ref, hint=hint)
        return None, None
    if not statements:
        logger.info(component="sql_validator", event="autofix_skipped_no_statements", bad=bad_ref, hint=hint)
        return None, None

    alias_to_table: dict[str, str] = {}
    cte_names: set[str] = set()
    for stmt in statements:
        if stmt is None:
            continue
        for cte in stmt.find_all(exp.CTE):
            if cte.alias:
                cte_names.add(cte.alias.lower())
        for tbl in stmt.find_all(exp.Table):
            name = (tbl.name or '').lower()
            if not name or name in cte_names:
                continue
            alias = (tbl.alias or '').lower()
            if alias:
                alias_to_table[alias] = name
            alias_to_table[name] = name

    if good_tbl_or_alias in alias_to_table:
        good_table = alias_to_table[good_tbl_or_alias]
    elif good_tbl_or_alias in schema_map:
        good_table = good_tbl_or_alias
    else:
        logger.info(
            component="sql_validator",
            event="autofix_skipped_target_not_in_scope",
            bad=bad_ref, hint=hint,
        )
        return None, None

    inv = schema_map.get(good_table)
    if inv is None or not hasattr(inv, 'columns'):
        logger.info(component="sql_validator", event="autofix_skipped_missing_schema", table=good_table)
        return None, None
    if good_col not in inv.columns:
        logger.info(
            component="sql_validator",
            event="autofix_skipped_col_not_in_ddl",
            bad=bad_ref, hint=hint,
            target=good_table,
        )
        return None, None

    chosen_alias = None
    for a, t in alias_to_table.items():
        if t == good_table and a != good_table:
            chosen_alias = a
            break
    rewrite_table_part = chosen_alias or good_table

    replacements_made = 0
    new_statements: list[exp.Expression] = []
    for stmt in statements:
        if stmt is None:
            new_statements.append(stmt)
            continue
        def _swap(node: exp.Expression) -> exp.Expression:
            nonlocal replacements_made
            if isinstance(node, exp.Column):
                if ((node.table or '').lower() == bad_tbl_or_alias
                        and (node.name or '').lower() == bad_col):
                    replacements_made += 1
                    return exp.Column(
                        this  = exp.to_identifier(good_col, quoted=False),
                        table = exp.to_identifier(rewrite_table_part, quoted=False),
                    )
            return node
        new_statements.append(stmt.transform(_swap, copy=True))

    if replacements_made == 0:
        return None, None

    new_sql_parts = []
    for stmt in new_statements:
        if stmt is None:
            continue
        new_sql_parts.append(stmt.sql(dialect="postgres"))
    new_sql = ";\n".join(new_sql_parts)
    if sql.rstrip().endswith(';') and not new_sql.endswith(';'):
        new_sql += ';'

    pgcode, err = run_explain(new_sql)
    if pgcode is None and err is None:
        desc = (
            f"autofix: replaced `{bad_ref}` with `{rewrite_table_part}."
            f"{good_col}` ({replacements_made} occurrence(s)) per "
            f"PostgreSQL planner hint"
        )
        logger.info(
            component="sql_validator",
            event="autofix_accepted",
            bad=bad_ref, hint=hint,
            target=f"{rewrite_table_part}.{good_col}",
            replacements=replacements_made,
        )
        return new_sql, desc

    logger.info(
        component="sql_validator",
        event="autofix_re_explain_failed",
        bad=bad_ref, hint=hint,
        new_err=str(err)[:120] if err else None,
    )
    return None, None
