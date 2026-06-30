import sqlglot
import sqlglot.expressions as exp
import re
from utils.logging_config import get_logger

logger = get_logger(__name__)

def has_eq_predicate(sql: str, col_name: str, value: int) -> bool:
    """Check if the SQL already contains a matching equality filter."""
    try:
        stmt = sqlglot.parse_one(sql, dialect="postgres")
        for eq in stmt.find_all(exp.EQ):
            left  = eq.left
            right = eq.right
            col_node = None
            val_node = None
            
            if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                col_node, val_node = left, right
            elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
                col_node, val_node = right, left
            
            if col_node and val_node:
                if col_node.name.lower() == col_name.lower():
                    try:
                        if int(val_node.this) == value:
                            return True
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return False

def inject_where(sql: str, col_name: str, value: int, schema_map: dict) -> str | None:
    """Inject a tenant isolation filter into the query using AST manipulation."""
    try:
        stmt = sqlglot.parse_one(sql, dialect="postgres")

        alias_map: dict[str, str] = {}
        for tbl in stmt.find_all(exp.Table):
            canon = tbl.name.lower()
            alias = (tbl.alias or "").lower()
            if alias:
                alias_map[alias] = canon
            alias_map[canon] = canon

        def _tables_and_aliases_in_scope(select_node: exp.Select) -> list[tuple[str, str]]:
            result = []
            from_node = select_node.args.get("from")
            if from_node:
                for tbl in from_node.find_all(exp.Table):
                    canon = tbl.name.lower()
                    alias = (tbl.alias or tbl.name or "").lower()
                    result.append((alias, canon))
            for join in select_node.args.get("joins", []):
                for tbl in join.find_all(exp.Table):
                    canon = tbl.name.lower()
                    alias = (tbl.alias or tbl.name or "").lower()
                    result.append((alias, canon))
            return result

        def _qualified_predicate(select_node: exp.Select) -> str | None:
            for alias, canon in _tables_and_aliases_in_scope(select_node):
                inv = schema_map.get(canon)
                if inv and hasattr(inv, "columns") and col_name in inv.columns:
                    qualifier = alias if alias else canon
                    return f"{qualifier}.{col_name} = {value}"
            return None

        outer_select = stmt if isinstance(stmt, exp.Select) else stmt.find(exp.Select)
        if outer_select:
            predicate = _qualified_predicate(outer_select)
            if predicate:
                injected = stmt.where(predicate, dialect="postgres")
                return injected.sql(dialect="postgres")

        for cte in stmt.find_all(exp.CTE):
            cte_select = cte.find(exp.Select)
            if cte_select is None:
                continue
            predicate = _qualified_predicate(cte_select)
            if predicate:
                modified = cte_select.where(predicate, dialect="postgres")
                cte.set("this", modified)
                return stmt.sql(dialect="postgres")

        predicate_unqualified = f"{col_name} = {value}"
        injected = stmt.where(predicate_unqualified, dialect="postgres")
        return injected.sql(dialect="postgres")

    except Exception as exc:
        logger.warning(
            component="sql_validator",
            event="tenant_filter_ast_failed",
            col_name=col_name,
            value=value,
            error=str(exc),
            note="AST injection failed; trying next scoping path",
        )
        return None
