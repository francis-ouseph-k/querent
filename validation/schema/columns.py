"""
validation/schema/columns.py
────────────────────────────
"""
import sqlglot.expressions as exp
from ..core.context import ValidationContext
from models.schema import ValidationResult
from utils.logging_config import get_logger
from validation.utils.blocklist import COLUMN_BLOCKLIST as _COLUMN_BLOCKLIST

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-SELECT scope resolution
# ─────────────────────────────────────────────────────────────────────────────
# FIX (scope-blind ambiguity, Q1/Q2/Q20/Q54): the previous implementation resolved
# every unqualified column against ctx.sql_tables — the UNION of every table in the
# whole statement — and only auto-resolved a bare column when the *entire statement*
# had exactly one table. Any unqualified column inside an IN/EXISTS/scalar subquery
# whose inner SELECT sqlglot does NOT wrap in exp.Subquery was therefore judged
# against all statement tables, producing bogus "Ambiguous column" / "unaliased"
# errors on correct SQL. This version resolves each column against its OWN enclosing
# SELECT scope (inner shadows outer, correlated refs fall back to outer scopes),
# which matches SQL name-resolution semantics.

def _local_scope(select_node):
    """
    (alias_map, tables) for the tables declared directly in this SELECT's
    FROM + JOINs. Derived tables (subquery in FROM) contribute only their
    alias — their inner columns are validated in their own scope.
    """
    alias_map: dict[str, str] = {}
    tables: set[str] = set()
    derived: set[str] = set()

    containers = []
    # sqlglot stores FROM under "from" (<=26.x) or "from_" (>=30.x). Support both.
    from_node = select_node.args.get("from") or select_node.args.get("from_")
    if from_node is not None:
        containers.append(from_node)
    containers.extend(select_node.args.get("joins", []) or [])

    for container in containers:
        # Only the direct table/derived-table target of the FROM/JOIN, not
        # tables nested inside subqueries (those are separate scopes).
        this = getattr(container, "this", container)
        if isinstance(this, exp.Table):
            name = (this.name or "").lower()
            alias = (this.alias or "").lower()
            if name:
                tables.add(name)
                alias_map[name] = name
            if alias:
                alias_map[alias] = name
        elif isinstance(this, exp.Subquery):
            alias = (this.alias or "").lower()
            if alias:
                derived.add(alias)
        else:
            # Fallback: pull any direct Table children (unusual join shapes)
            for tbl in (this.find_all(exp.Table) if hasattr(this, "find_all") else []):
                if tbl.find_ancestor(exp.Subquery) is not None:
                    continue
                name = (tbl.name or "").lower()
                alias = (tbl.alias or "").lower()
                if name:
                    tables.add(name)
                    alias_map[name] = name
                if alias:
                    alias_map[alias] = name
    return alias_map, tables, derived


def _scope_chain(col_node):
    """Enclosing SELECT then its ancestor SELECTs (inner-first) for correlated refs."""
    chain = []
    node = col_node
    while node is not None:
        sel = node.find_ancestor(exp.Select)
        if sel is None:
            break
        chain.append(sel)
        node = sel.parent
    return chain


def validate_columns(ctx: ValidationContext) -> ValidationResult | None:
    """
    Column-level existence / ambiguity checks, resolved per-SELECT scope.
    """
    col_errors: list[str] = []
    sql = ctx.working_sql or ctx.sql
    if ctx.ast is None:
        return None

    try:
        for stmt in ctx.ast:
            if stmt is None:
                continue

            cte_names = {c.alias.lower() for c in stmt.find_all(exp.CTE) if c.alias}
            projection_aliases = {a.alias.lower() for a in stmt.find_all(exp.Alias) if a.alias}

            # Precompute local scope per SELECT once.
            scope_cache: dict[int, tuple] = {}

            def scope_for(select_node):
                key = id(select_node)
                if key not in scope_cache:
                    scope_cache[key] = _local_scope(select_node)
                return scope_cache[key]

            for col_node in stmt.find_all(exp.Column):
                col_name = (col_node.name or "").lower()
                tbl_part = (col_node.table or "").lower()

                if not col_name or col_name == "*":
                    continue

                chain = _scope_chain(col_node)

                # ── Qualified reference: resolve alias inner→outer ──────────
                if tbl_part:
                    if tbl_part in cte_names:
                        continue
                    resolved_table = None
                    for sel in chain:
                        amap, _tables, derived = scope_for(sel)
                        if tbl_part in derived:
                            resolved_table = "__derived__"
                            break
                        if tbl_part in amap:
                            resolved_table = amap[tbl_part]
                            break
                    if resolved_table == "__derived__":
                        continue
                    if resolved_table is None:
                        return ValidationResult(
                            passed=False, step="schema",
                            message=f"Unknown table or alias '{tbl_part}' referenced in "
                                    f"'{tbl_part}.{col_name}'. Ensure it is declared in the "
                                    f"FROM/JOIN clause.",
                            sql=sql,
                        )
                    _blocklist_or_column(resolved_table, col_name, ctx, sql, col_errors)
                    res = _blocklist_check(resolved_table, col_name, ctx, sql)
                    if res:
                        return res
                    continue

                # ── Bare (unqualified) reference ───────────────────────────
                if col_name in projection_aliases:
                    continue  # references a SELECT-list alias (Q50)

                enclosing = chain[0] if chain else None
                if enclosing is None:
                    continue
                _amap, local_tables, _derived = scope_for(enclosing)
                local_tables = local_tables - cte_names

                local_hits = [
                    t for t in local_tables
                    if t in ctx.schema_map
                    and hasattr(ctx.schema_map[t], "columns")
                    and col_name in ctx.schema_map[t].columns
                ]

                if len(local_hits) == 1:
                    res = _blocklist_check(local_hits[0], col_name, ctx, sql)
                    if res:
                        return res
                    continue
                if len(local_hits) > 1:
                    return ValidationResult(
                        passed=False, step="schema",
                        message=f"Ambiguous column reference: '{col_name}'. It exists in "
                                f"multiple tables ({', '.join(sorted(local_hits))}). You must "
                                f"qualify it with a table alias.",
                        sql=sql,
                    )

                # Not found in local scope — try outer scopes (correlated ref).
                outer_hits = []
                for sel in chain[1:]:
                    _oa, otables, _od = scope_for(sel)
                    for t in (otables - cte_names):
                        if (t in ctx.schema_map
                                and hasattr(ctx.schema_map[t], "columns")
                                and col_name in ctx.schema_map[t].columns):
                            outer_hits.append(t)
                if len(outer_hits) >= 1:
                    continue  # correlated reference resolves to an outer table

                if local_tables:
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


def _blocklist_check(resolved_table, col_name, ctx, sql):
    if (resolved_table, col_name) in _COLUMN_BLOCKLIST:
        return ValidationResult(
            passed=False, step="schema",
            message=f"Column validation failed: {_COLUMN_BLOCKLIST[(resolved_table, col_name)]}",
            sql=sql,
        )
    return None


def _blocklist_or_column(resolved_table, col_name, ctx, sql, col_errors):
    if resolved_table in ctx.schema_map:
        inv = ctx.schema_map[resolved_table]
        if hasattr(inv, "columns") and col_name not in inv.columns:
            col_errors.append(f"{resolved_table}.{col_name}")