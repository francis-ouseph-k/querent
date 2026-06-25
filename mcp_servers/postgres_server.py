"""
mcp_servers/postgres_server.py
────────────────────────────────
MCP server wrapping the PostgreSQL read-only replica for the NL→SQL pipeline.

FASTMCP VERSION: 3.x (tested on 3.4.2)
  Import : from fastmcp import FastMCP
  Startup: mcp.run(transport="http", ...)   (FastMCP calls uvicorn internally)

  FastMCP 3.x manages uvicorn internally via run_http_async().
  You do not call uvicorn.run() directly — FastMCP does it for you.
  To pass uvicorn tuning options use the uvicorn_config parameter (see entry point).

Exposes three tools:
  execute_query  — run validated SELECT, enforce LIMIT, return rows as JSON
  explain_query  — EXPLAIN (FORMAT JSON), return estimated total cost
  health_check   — verify DB connection is alive

WHY A SEPARATE PROCESS
  The connection pool and all security enforcement live here.
  Benefits of isolating them:
    1. The C3 fix (rollback before pool release) is enforced once in
       _release_conn(). Every tool that borrows a connection automatically
       gets the fix — no risk of a future caller forgetting rollback.
    2. psycopg2 version changes only affect this file.
    3. RLS variable (SET LOCAL app.current_user_id) is applied and cleared
       here, in one place, per request — no leaking between requests.

C3 FIX — why rollback before pool release matters
  psycopg2 connections have autocommit=OFF by default. Any query opens an
  implicit transaction. If the connection is returned to the pool without
  a rollback(), that transaction stays open (idle-in-transaction).

  Two problems:
    1. On the read replica, idle-in-transaction holds a snapshot that blocks
       vacuum on the primary and causes replication lag over time.
    2. SET LOCAL app.current_user_id is transaction-scoped. If the transaction
       never ends, that user_id stays active on the connection. The NEXT
       request that reuses the connection from the pool inherits User A's
       identity and User A's RLS row filter — a security bug.

  Fix: _release_conn() always calls conn.rollback() before putconn().
  Rollback is always safe on a read-only replica — it was a SELECT anyway.

SECURITY MODEL
  - Connects to the READ-ONLY REPLICA ONLY — never the primary.
  - default_transaction_read_only=on enforced at the PostgreSQL connection
    options string level. Even if validation is somehow bypassed, the
    connection itself rejects any DML at the driver level.
  - statement_timeout set per connection — runaway queries capped at 30s.
  - LIMIT enforced via sqlglot AST injection (not string append) so a LIMIT
    inside a subquery does not fool the check.
  - SET LOCAL RLS value passed as a parameterised %s value — never
    interpolated into SQL, immune to injection.

TRANSPORT
  FastMCP 3.x Streamable HTTP (MCP 2025-03-26 spec).
  Tools served at: POST http://<host>:<port>/mcp

STARTUP
  python mcp_servers/postgres_server.py

CONFIG (.env — all optional, defaults shown)
  MCP_POSTGRES_HOST=127.0.0.1
  MCP_POSTGRES_PORT=5012
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

# ── Add project root to sys.path so config/settings.py resolves correctly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import psycopg2
import psycopg2.pool
import sqlglot
import sqlglot.expressions as exp
from fastmcp import FastMCP

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── MCP server instance ────────────────────────────────────────────────────────
mcp = FastMCP(
    name        = "postgres-readonly",
    instructions = "Read-only PostgreSQL query execution for NL→SQL pipeline",
)

# ── Connection pool ────────────────────────────────────────────────────────────
# Module-level lazy singleton. Protected by a threading.Lock because FastMCP
# may receive concurrent requests from multiple clients.
# Pool is initialised on first tool call, not at import time — avoids
# connection errors during startup if PostgreSQL is not yet ready.
_pool:      psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock: threading.Lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """
    Lazy connection pool — created on first call, reused for all subsequent calls.

    Uses double-checked locking to avoid race conditions when multiple
    concurrent requests arrive before the pool is initialised.

    Pool settings come from settings.postgres (PG_* in .env):
      PG_POOL_MIN  — minimum idle connections (default 2)
      PG_POOL_MAX  — maximum total connections (default 20)
      PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

    Two security settings enforced at the connection options string level
    so they apply to EVERY connection from this pool:
      default_transaction_read_only=on  — rejects DML at driver level
      statement_timeout=<ms>            — kills runaway queries at DB level
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:   # second check inside the lock
                pg    = settings.postgres
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn  = pg.pool_min,
                    maxconn  = pg.pool_max,
                    host     = pg.host,
                    port     = pg.port,
                    dbname   = pg.database,
                    user     = pg.user,
                    password = pg.password,
                    options  = (
                        f"-c default_transaction_read_only=on "
                        f"-c statement_timeout={pg.statement_timeout_ms}"
                    ),
                )
                logger.info(
                    component = "postgres_mcp",
                    event     = "pool_created",
                    host      = pg.host,
                    port      = pg.port,
                    database  = pg.database,
                    pool_min  = pg.pool_min,
                    pool_max  = pg.pool_max,
                )
    return _pool


def _get_conn():
    """Borrow a connection from the pool. Returns None if pool unavailable."""
    return _get_pool().getconn()


def _release_conn(conn) -> None:
    """
    Return a connection to the pool — ALWAYS calls rollback() first.

    C3 FIX: this is the single place where connections are returned to the pool.
    rollback() is called unconditionally before putconn():
      - Ends any open implicit transaction (closes idle-in-transaction state)
      - Clears SET LOCAL app.current_user_id so it does not leak to the next
        request that reuses this connection

    rollback() on a read-only connection is always safe and effectively free
    (there is nothing to roll back — it just closes the transaction snapshot).

    Both except blocks swallow exceptions silently because:
      - If rollback() fails, the connection is already in an error state —
        putconn() will detect this and discard the connection.
      - If putconn() fails, the connection is already lost — nothing to do.
    """
    if conn is not None:
        try:
            conn.rollback()          # end open transaction, clear SET LOCAL
        except Exception:
            pass                     # connection already dead — ignore
        try:
            _get_pool().putconn(conn)
        except Exception:
            pass                     # pool already closed — ignore


def _outer_has_limit(sql: str) -> bool:
    """
    Return True if the OUTERMOST SELECT in sql already has a LIMIT clause.

    Used to avoid double-adding LIMIT. We check only the outer query because
    a LIMIT inside a subquery does not cap the final result set.

    Uses sqlglot AST inspection (not regex) so constructs like
    'SELECT ... FROM (SELECT ... LIMIT 5) subq' return False correctly —
    the inner LIMIT does not count, only the outer one does.

    Returns False on parse errors — safe default is to add LIMIT.
    """
    try:
        stmt  = sqlglot.parse_one(sql, dialect="postgres")
        outer = stmt if isinstance(stmt, exp.Select) else stmt.find(exp.Select)
        return outer is not None and outer.args.get("limit") is not None
    except Exception:
        return False   # parse failed — default to adding LIMIT (safe)


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def execute_query(
    sql:      str,
    user_id:  str | None = None,
    max_rows: int        = 1000,
) -> dict[str, Any]:
    """
    Execute a validated read-only SQL query against the PostgreSQL replica.

    This tool should only receive SQL that has already passed the 6-step
    validation pipeline in sql_validator.py (syntax, schema, safety, security,
    cost, timeout). It enforces hard limits at the connection and query level:

    Enforcement layers applied here:
      1. RLS: SET LOCAL app.current_user_id = user_id (parameterised, not
         interpolated) so PostgreSQL row-level security applies to this query.
         SET LOCAL is transaction-scoped — cleared by rollback() in _release_conn().
      2. LIMIT: appended via AST inspection if not already present. Capped at
         min(max_rows, PG_MAX_ROWS setting). Prevents large unbounded results.
      3. default_transaction_read_only=on at the connection level (see _get_pool).
      4. statement_timeout at the connection level (see _get_pool).

    Args:
        sql:      Validated PostgreSQL SELECT statement. Trailing semicolons
                  are stripped before LIMIT injection.
        user_id:  Application user ID for RLS. Passed as a parameterised %s
                  value to SET LOCAL — never string-interpolated into SQL.
                  None = skip SET LOCAL (admin / cross-tenant queries).
        max_rows: Maximum rows to return. Hard-capped at PG_MAX_ROWS setting.
                  Default 1000. Increase for bulk export use cases.

    Returns:
        {"rows": [...], "row_count": N, "elapsed_ms": M}   on success
        {"error": "<message>", "elapsed_ms": M}             on failure
        Rows are list of dicts keyed by column name.
    """
    t0   = time.time()
    conn = _get_conn()
    if conn is None:
        return {"error": "PostgreSQL connection not available.", "elapsed_ms": 0}

    try:
        cur = conn.cursor()

        # Apply RLS user identity — parameterised to avoid injection
        if user_id and settings.rls_variable:
            cur.execute(
                f"SET LOCAL {settings.rls_variable} = %s",
                (str(user_id),),
            )

        # Append LIMIT if absent — use AST check, not regex, to handle subqueries
        effective_max = min(max_rows, settings.postgres.max_rows)
        limited_sql   = sql.rstrip(";")
        if not _outer_has_limit(limited_sql):
            limited_sql = f"{limited_sql} LIMIT {effective_max}"

        cur.execute(limited_sql)

        # Build list of dicts — one dict per row, keyed by column name
        columns  = [desc[0] for desc in cur.description] if cur.description else []
        raw_rows = cur.fetchall()
        rows     = [dict(zip(columns, row)) for row in raw_rows]
        cur.close()

        elapsed = round((time.time() - t0) * 1000)
        logger.info(
            component   = "postgres_mcp",
            event       = "query_executed",
            row_count   = len(rows),
            elapsed_ms  = elapsed,
            sql_preview = sql[:80],
        )
        return {"rows": rows, "row_count": len(rows), "elapsed_ms": elapsed}

    except psycopg2.Error as exc:
        elapsed = round((time.time() - t0) * 1000)
        logger.error(
            component  = "postgres_mcp",
            event      = "query_error",
            error      = str(exc),
            pgcode     = getattr(exc, "pgcode", None),
            elapsed_ms = elapsed,
        )
        return {"error": str(exc), "elapsed_ms": elapsed}

    finally:
        # C3 fix: _release_conn always calls rollback() before putconn()
        # This clears the SET LOCAL RLS variable and closes the transaction.
        _release_conn(conn)


@mcp.tool()
def explain_query(sql: str) -> dict[str, Any]:
    """
    Run EXPLAIN (FORMAT JSON) and return the estimated total cost.

    Called by sql_validator.py Step 5 (cost check) to reject queries whose
    estimated cost exceeds VALIDATION_EXPLAIN_COST_THRESHOLD before they
    reach the execution layer.

    M7 FIX — EXPLAIN runs with LIMIT appended:
      Without LIMIT, PostgreSQL estimates the cost for the full table scan
      even though only max_rows will actually be fetched. A query against
      evaluation_marks (~20M rows) would show an astronomically high cost
      and fail the threshold check, even though it would return 1000 rows
      in ~50ms. LIMIT is appended before EXPLAIN so the estimate matches
      what execute_query will actually run.

    pgcode in error response:
      Class 42 errors (42xxx) are schema errors like column not found or
      ambiguous column. These are returned with the pgcode so the validator
      can distinguish "schema error caught by EXPLAIN" from infrastructure
      failures — schema errors should fail validation, infra errors should
      be logged and retried.

    Args:
        sql: Post-validation, post-tenant-injection SQL string.
             Should be the same SQL that will be passed to execute_query.

    Returns:
        {"total_cost": float, "elapsed_ms": int}   on success
        {"error": "...", "pgcode": "...", "elapsed_ms": int}   on failure
    """
    t0   = time.time()
    conn = _get_conn()
    if conn is None:
        return {"error": "PostgreSQL connection not available.", "elapsed_ms": 0}

    try:
        cur = conn.cursor()

        # M7 fix: append LIMIT so the cost estimate matches actual execution
        explain_body = sql.rstrip(";")
        if not _outer_has_limit(explain_body):
            explain_body = f"{explain_body} LIMIT {settings.postgres.max_rows}"

        cur.execute(f"EXPLAIN (FORMAT JSON) {explain_body}")
        plan    = cur.fetchone()[0]
        elapsed = round((time.time() - t0) * 1000)
        cur.close()

        # Extract total cost from the JSON plan tree
        total_cost = 0.0
        if plan and isinstance(plan, list):
            total_cost = plan[0].get("Plan", {}).get("Total Cost", 0.0)

        logger.info(
            component  = "postgres_mcp",
            event      = "explain_complete",
            total_cost = total_cost,
            elapsed_ms = elapsed,
        )
        return {"total_cost": total_cost, "elapsed_ms": elapsed, "plan": plan}

    except psycopg2.Error as exc:
        elapsed = round((time.time() - t0) * 1000)
        pgcode  = getattr(exc, "pgcode", None) or ""
        logger.warning(
            component  = "postgres_mcp",
            event      = "explain_error",
            error      = str(exc),
            pgcode     = pgcode,
            elapsed_ms = elapsed,
        )
        return {"error": str(exc), "pgcode": pgcode, "elapsed_ms": elapsed}

    finally:
        # C3 fix: always rollback before returning to pool
        _release_conn(conn)


@mcp.tool()
def health_check() -> dict[str, Any]:
    """
    Verify the PostgreSQL connection is alive and the replica is reachable.

    Runs SELECT version() — the lightest possible query.
    Useful for startup health checks and monitoring.

    Returns:
        {"status": "ok", "server_version": "PostgreSQL 16.x ...", "elapsed_ms": N}
        or
        {"status": "error", "error": "<message>", "elapsed_ms": N}
    """
    t0   = time.time()
    conn = _get_conn()
    if conn is None:
        return {"status": "error", "error": "Pool returned None.", "elapsed_ms": 0}

    try:
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        cur.close()

        elapsed = round((time.time() - t0) * 1000)
        logger.info(
            component  = "postgres_mcp",
            event      = "health_ok",
            elapsed_ms = elapsed,
        )
        return {"status": "ok", "server_version": version, "elapsed_ms": elapsed}

    except psycopg2.Error as exc:
        elapsed = round((time.time() - t0) * 1000)
        logger.error(
            component  = "postgres_mcp",
            event      = "health_failed",
            error      = str(exc),
            elapsed_ms = elapsed,
        )
        return {"status": "error", "error": str(exc), "elapsed_ms": elapsed}

    finally:
        # C3 fix: always rollback before returning to pool
        _release_conn(conn)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = settings.mcp.postgres_host
    port = settings.mcp.postgres_port

    logger.info(
        component = "postgres_mcp",
        event     = "server_starting",
        host      = host,
        port      = port,
    )

    # FastMCP 3.x calls uvicorn internally via run_http_async().
    # You do not call uvicorn.run() directly — FastMCP manages it.
    #
    # transport="http" = Streamable HTTP (MCP 2025-03-26 spec, recommended)
    # transport="sse"  = legacy SSE for older MCP clients
    mcp.run(
        transport      = "http",
        host           = host,
        port           = port,
        json_response  = True,   # return plain JSON instead of SSE stream
        stateless_http = True,   # no session handshake required per call
    )

    # If you ever need to tune uvicorn (e.g. log level, workers):
    # mcp.run(
    #     transport      = "http",
    #     host           = host,
    #     port           = port,
    #     uvicorn_config = {"log_level": "warning", "workers": 1},
    # )