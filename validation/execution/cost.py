import time
import psycopg2
from typing import Any
from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from config.settings import settings
from utils.logging_config import get_logger
from validation.utils.blocklist import classify_pg_error as _classify_pg_error, outer_query_has_limit as _outer_query_has_limit
from mcp_tools.client import call_postgres_explain, MCPCallError
from ..utils.autofix import attempt_pg_autofix

logger = get_logger(__name__)

class CostValidator(BaseValidationStep):
    name = "CostValidator"

    def __init__(self, get_conn, release_conn, db_dsn):
        self._get_conn = get_conn
        self._release_conn = release_conn
        self.db_dsn = db_dsn

    def _inspect_plan_node(self, node: dict, warnings: list[str]) -> None:
        if not node:
            return
        node_type = node.get("Node Type", "")
        relation = node.get("Relation Name", "")
        plan_rows = node.get("Plan Rows", 0)
        total_cost = node.get("Total Cost", 0.0)

        if node_type == "Seq Scan" and relation:
            if plan_rows > 1000:
                warnings.append(
                    f"Sequential Scan on table '{relation}' (estimated {plan_rows} rows, cost {total_cost:.1f}). "
                    f"Consider adding specific filters (e.g. board_id, course_id, exam_id, or student_id) "
                    f"to allow the query planner to use existing indexes."
                )

        for child in node.get("Plans", []):
            self._inspect_plan_node(child, warnings)

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 5: Resource Cost Limit check - Run EXPLAIN and check estimated costs.
        """
        sql = ctx.working_sql or ctx.sql
        threshold = settings.validation.explain_cost_threshold

        def check_pgcode(pgcode: str, error_msg: str, run_explain=None) -> ValidationResult | None:
            if pgcode.startswith("42") or pgcode.startswith("22"):
                if run_explain is not None:
                    fixed_sql, desc = attempt_pg_autofix(
                        sql, error_msg, ctx.schema_map, run_explain
                    )
                    if fixed_sql is not None:
                        ctx.working_sql = fixed_sql
                        return ValidationResult(
                            passed  = True,
                            step    = "cost",
                            message = desc or "PG planner autofix accepted",
                            sql     = fixed_sql,
                        )

                step, message = _classify_pg_error(error_msg)
                return ValidationResult(
                    passed  = False,
                    step    = step,
                    message = message,
                    sql     = sql,
                )
            return None

        # ── Case A: MCP Connection Path ───────────────────────────────────
        if settings.use_mcp_servers:
            def _mcp_run_explain(new_sql: str):
                try:
                    r = call_postgres_explain(new_sql)
                except MCPCallError as e:
                    return ("08000", str(e))
                if "error" in r:
                    return (r.get("pgcode", ""), r["error"])
                return (None, None)

            try:
                result = call_postgres_explain(sql)
            except MCPCallError as exc:
                logger.warning(
                    component = "sql_validator",
                    event     = "explain_mcp_unavailable",
                    error     = str(exc),
                    note      = "Skipping cost check — MCP postgres server unreachable.",
                )
                return ValidationResult(passed=True, step="cost", sql=sql)

            if "error" in result:
                pgcode = result.get("pgcode", "")
                validation_err = check_pgcode(pgcode, result["error"], run_explain=_mcp_run_explain)
                if validation_err:
                    return validation_err
                    
                logger.warning(component="sql_validator", event="explain_failed", error=result["error"])
                return ValidationResult(passed=True, step="cost", sql=sql)

            total_cost = result.get("total_cost", 0.0)
            logger.info(component="sql_validator", event="explain_complete",
                        total_cost=total_cost, explain_ms=result.get("elapsed_ms"))

            plan = result.get("plan")
            warnings = []
            if plan and isinstance(plan, list):
                self._inspect_plan_node(plan[0].get("Plan", {}), warnings)

            if total_cost > threshold:
                msg = f"Query estimated cost {total_cost:.0f} exceeds threshold {threshold}. Add specific filters (board_id, exam_id, etc.)."
                if warnings:
                    msg += "\nPerformance issues:\n- " + "\n- ".join(warnings)
                return ValidationResult(
                    passed  = False,
                    step    = "cost",
                    message = msg,
                    sql     = sql,
                )
            return ValidationResult(passed=True, step="cost", sql=sql)

        # ── Case B: Direct psycopg2 Connection Path ───────────────────────
        conn       = None
        using_pool = False

        if self._get_conn is not None:
            try:
                conn = self._get_conn()
                using_pool = True
            except Exception as exc:
                logger.warning(
                    component="sql_validator",
                    event="cost_check_connection_failed",
                    error=str(exc),
                    note="Skipping cost check — could not acquire a connection.",
                )
                conn = None
                using_pool = False
        elif self.db_dsn:
            conn = psycopg2.connect(
                self.db_dsn,
                options = f"-c statement_timeout={settings.postgres.statement_timeout_ms}",
            )
            conn.set_session(readonly=True)

        if conn is None:
            return ValidationResult(passed=True, step="cost", sql=sql)

        try:
            cur = conn.cursor()
            explain_body = sql.rstrip(";")
            if not _outer_query_has_limit(explain_body):
                explain_body = f"{explain_body} LIMIT {settings.postgres.max_rows}"

            t0 = time.time()
            cur.execute(f"EXPLAIN (FORMAT JSON) {explain_body}")
            plan    = cur.fetchone()[0]
            elapsed = round((time.time() - t0) * 1000)
            cur.close()

            if not using_pool:
                conn.close()

            total_cost = 0.0
            if plan and isinstance(plan, list):
                total_cost = plan[0].get("Plan", {}).get("Total Cost", 0.0)

            logger.info(component="sql_validator", event="explain_complete",
                        total_cost=total_cost, explain_ms=elapsed)

            warnings = []
            if plan and isinstance(plan, list):
                self._inspect_plan_node(plan[0].get("Plan", {}), warnings)

            if total_cost > threshold:
                msg = f"Query estimated cost {total_cost:.0f} exceeds threshold {threshold}. Add specific filters (board_id, exam_id, etc.)."
                if warnings:
                    msg += "\nPerformance issues:\n- " + "\n- ".join(warnings)
                return ValidationResult(
                    passed  = False,
                    step    = "cost",
                    message = msg,
                    sql     = sql,
                )

        except psycopg2.Error as exc:
            logger.warning(
                component="sql_validator",
                event="explain_error",
                error=str(exc),
                sql_preview=sql[:80],
            )
            pgcode = getattr(exc, "pgcode", None) or ""

            def _direct_run_explain(new_sql: str):
                try:
                    if conn is not None:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                    cur2 = conn.cursor()
                    explain_body2 = new_sql.rstrip(";")
                    if not _outer_query_has_limit(explain_body2):
                        explain_body2 = f"{explain_body2} LIMIT {settings.postgres.max_rows}"
                    cur2.execute(f"EXPLAIN (FORMAT JSON) {explain_body2}")
                    cur2.fetchone()
                    cur2.close()
                    return (None, None)
                except psycopg2.Error as e2:
                    return (getattr(e2, "pgcode", None) or "", str(e2))
                except Exception as e2:
                    return ("UNKNOWN", str(e2))

            validation_err = check_pgcode(pgcode, str(exc), run_explain=_direct_run_explain)
            if validation_err:
                return validation_err

            logger.warning(component="sql_validator", event="explain_failed",
                           error=str(exc))

        finally:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            if using_pool and self._release_conn and conn is not None:
                self._release_conn(conn)

        return ValidationResult(passed=True, step="cost", sql=sql)
