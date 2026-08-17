
"""
validation/security/tenant_injector.py
──────────────────────────────────────
Tenant-filter injection helpers used by the SecurityTransformer (step 7).

    has_eq_predicate(...)   is the tenant column already constrained? (avoid
                            injecting a duplicate/again)
    inject_where(...)       AST-level insertion of `alias.tenant_col = value` into
                            each SELECT scope, qualified to the correct table alias

CTE- and scope-aware: the predicate is attached at the right SELECT so it filters
the base rows rather than a post-aggregation result. Reads the sqlglot FROM node
under both `from` (<=26.x) and `from_` (>=30.x) so scope detection survives a
driver upgrade. Falls back to an unqualified predicate only when alias resolution
fails.
"""

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

def scopes_requiring_tenant_filter(
    sql: str, tenant_scoped_tables: set[str]
) -> list[str]:
    """
    Names of every table in every SELECT scope that is tenant-scoped.

    Used to decide whether a query was FULLY scoped or only partially. A query
    with three CTEs each reading answer_script needs three predicates, not one.
    """
    out: list[str] = []
    try:
        stmt = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return out
    for select_node in stmt.find_all(exp.Select):
        for tbl in select_node.find_all(exp.Table):
            name = (tbl.name or "").lower()
            if name in tenant_scoped_tables:
                out.append(name)
    return out


def inject_where_all_scopes(
    sql: str,
    col_name: str,
    value: int,
    schema_map: dict,
    tenant_scoped_tables: set[str],
) -> tuple[str | None, list[str]]:
    """
    Inject `<alias>.<col_name> = <value>` into EVERY SELECT scope that reads a
    tenant-scoped table, not just the outermost one.

    G2 fix. The previous single-scope `inject_where` walked the outer SELECT,
    fell back to the FIRST matching CTE, and returned on the first success. A
    query shaped

        WITH a AS (SELECT ... FROM answer_script),
             b AS (SELECT ... FROM evaluation_attempt)
        SELECT ... FROM a JOIN b ...

    therefore got a predicate on `a` and nothing on `b`, and the outer SELECT
    (which reads only CTE names, never a base table) got nothing at all. The
    query then passed validation while reading another tenant's rows through
    `b`. That is a confidentiality gap, not a hygiene one.

    Returns (rewritten_sql, unscoped_tables). `unscoped_tables` is the list of
    tenant-scoped tables that could NOT be given a predicate — the caller is
    expected to fail closed when it is non-empty rather than ship a partially
    scoped query. Returns (None, [...]) when the SQL will not parse.
    """
    try:
        stmt = sqlglot.parse_one(sql, dialect="postgres")
    except Exception as exc:
        logger.warning(
            component="sql_validator",
            event="tenant_filter_parse_failed",
            error=str(exc),
            note="cannot scope an unparseable query; caller must fail closed",
        )
        return None, sorted(tenant_scoped_tables)

    unscoped: list[str] = []

    # Deepest-first: rewriting an inner SELECT must not invalidate the node
    # references held for an outer one.
    select_nodes = list(stmt.find_all(exp.Select))
    select_nodes.reverse()

    for select_node in select_nodes:
        # Only direct FROM/JOIN sources count. A reference to a CTE name is not
        # a base table and must not be scoped here — the CTE body was scoped in
        # its own pass.
        local: list[tuple[str, str]] = []
        from_node = select_node.args.get("from") or select_node.args.get("from_")
        if from_node:
            for tbl in from_node.find_all(exp.Table):
                local.append(((tbl.alias or tbl.name or "").lower(),
                              (tbl.name or "").lower()))
        for join in select_node.args.get("joins", []) or []:
            for tbl in join.find_all(exp.Table):
                local.append(((tbl.alias or tbl.name or "").lower(),
                              (tbl.name or "").lower()))

        needs = [(a, c) for a, c in local if c in tenant_scoped_tables]
        if not needs:
            continue

        predicate = None
        for alias, canon in needs:
            inv = schema_map.get(canon)
            if inv and hasattr(inv, "columns") and col_name in inv.columns:
                qualifier = alias if alias else canon
                predicate = f"{qualifier}.{col_name} = {value}"
                break

        if predicate is None:
            # Scope reads a tenant-scoped table but none of its sources carries
            # this scoping column (e.g. course_id scoping over a board_id-only
            # table). Record it; do NOT silently continue.
            unscoped.extend(c for _, c in needs)
            continue

        if _scope_has_eq_predicate(select_node, col_name, value):
            continue

        try:
            select_node.where(predicate, dialect="postgres", copy=False)
        except Exception as exc:
            logger.warning(
                component="sql_validator",
                event="tenant_filter_scope_inject_failed",
                predicate=predicate,
                error=str(exc),
            )
            unscoped.extend(c for _, c in needs)

    return stmt.sql(dialect="postgres"), sorted(set(unscoped))


def _scope_has_eq_predicate(select_node: exp.Select, col_name: str, value: int) -> bool:
    """Idempotence guard, evaluated per scope rather than per statement."""
    where = select_node.args.get("where")
    if where is None:
        return False
    for eq in where.find_all(exp.EQ):
        left, right = eq.left, eq.right
        col_node = val_node = None
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            col_node, val_node = left, right
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            col_node, val_node = right, left
        if col_node is None or val_node is None:
            continue
        if col_node.name.lower() != col_name.lower():
            continue
        try:
            if int(val_node.this) == value:
                return True
        except (ValueError, TypeError):
            continue
    return False


def inject_where(sql: str, col_name: str, value: int, schema_map: dict) -> str | None:
    """
    Single-scope injector. RETAINED for callers that genuinely want outer-scope
    only; SecurityTransformer now uses inject_where_all_scopes instead.
    """
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
            # sqlglot stores FROM under "from" (<=26.x) or "from_" (>=30.x).
            from_node = select_node.args.get("from") or select_node.args.get("from_")
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