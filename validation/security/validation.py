"""
validation/security/validation.py
─────────────────────────────────
SecurityTransformer (pipeline step 7, reports `security`).

The only step that REWRITES the SQL rather than just judging it. When a tenant
context is present (board_id / course_id, or an RLS variable), it injects the
appropriate scoping predicate into the query so a user can only see their own
rows, writing the result to ctx.working_sql for later steps to validate. The
actual AST surgery lives in tenant_injector.py; this class decides when to apply
it and handles the no-tenant-context case by passing through unchanged.
"""

import re
import sqlglot.expressions as exp
from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from config.settings import settings
from utils.logging_config import get_logger
from .tenant_injector import has_eq_predicate, inject_where

logger = get_logger(__name__)

class SecurityTransformer(BaseValidationStep):
    name = "SecurityTransformer"

    def __init__(self, tenant_scoped_tables: set[str]):
        self.tenant_scoped_tables = tenant_scoped_tables

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 4: Tenant Isolation - Inject tenant filters idempotently.
        """
        sql = ctx.working_sql or ctx.sql
        
        if not settings.rls_variable and not settings.tenant_column:
            return ValidationResult(passed=True, step="security", sql=sql)

        has_tenant_table = any(t in self.tenant_scoped_tables for t in ctx.tables_used)
        if not has_tenant_table:
            return ValidationResult(passed=True, step="security", sql=sql)

        rls_var = settings.rls_variable
        if rls_var:
            rls_key = rls_var.split(".")[-1].lower()
            try:
                if ctx.ast:
                    for stmt in ctx.ast:
                        if stmt is None:
                            continue
                        for node in stmt.walk():
                            if isinstance(node, exp.SetItem):
                                eq = node.find(exp.EQ)
                                if eq and rls_key in str(eq).lower():
                                    return ValidationResult(passed=True, step="security", sql=sql)
            except Exception:
                pass
                
            set_local_pattern = re.compile(
                rf"\bSET\s+LOCAL\s+{re.escape(rls_var)}\s*=", re.IGNORECASE
            )
            if set_local_pattern.search(sql):
                return ValidationResult(passed=True, step="security", sql=sql)

        # ── Path 1: Scoping by board_id ───────────────────────────────────
        board_id = ctx.user_context.get("board_id")
        if board_id:
            try:
                safe_board_id = int(board_id)
            except (ValueError, TypeError):
                return ValidationResult(
                    passed=False, step="security",
                    message=f"Invalid board_id in user context: {board_id!r}. board_id must be an integer.",
                    sql=sql,
                )
            
            if not has_eq_predicate(sql, "board_id", safe_board_id):
                injected_sql = inject_where(sql, "board_id", safe_board_id, ctx.schema_map)
                if injected_sql:
                    logger.info(
                        component="sql_validator",
                        event="tenant_filter_injected",
                        scope="board_id",
                        value=safe_board_id,
                    )
                    ctx.working_sql = injected_sql
                    return ValidationResult(passed=True, step="security", sql=injected_sql)

        # ── Path 2: Scoping by course_id ──────────────────────────────────
        course_id = ctx.user_context.get("course_id")
        if course_id:
            try:
                safe_course_id = int(course_id)
            except (ValueError, TypeError):
                return ValidationResult(
                    passed=False, step="security",
                    message=f"Invalid course_id in user context: {course_id!r}. course_id must be an integer.",
                    sql=sql,
                )
            
            if not has_eq_predicate(sql, "course_id", safe_course_id):
                injected_sql = inject_where(sql, "course_id", safe_course_id, ctx.schema_map)
                if injected_sql:
                    logger.info(
                        component="sql_validator",
                        event="tenant_filter_injected",
                        scope="course_id",
                        value=safe_course_id,
                    )
                    ctx.working_sql = injected_sql
                    return ValidationResult(passed=True, step="security", sql=injected_sql)

        # ── Path 3: Scoping by user_id via Row Level Security (RLS) ────────
        user_id = ctx.user_context.get("user_id")
        if user_id and rls_var:
            logger.info(
                component="sql_validator",
                event="tenant_filter_rls",
                scope="user_id",
                rls_var=rls_var,
            )
            return ValidationResult(passed=True, step="security", sql=sql)

        logger.warning(
            component="sql_validator",
            event="tenant_filter_unavailable",
            tables=ctx.tables_used,
            sql_preview=sql[:120],
            user_context_keys=list(ctx.user_context.keys()),
            note="Query touches tenant-scoped tables but no board_id / course_id / "
                 "user_id found in user_context. Allowed through — verify this is "
                 "an admin query.",
        )
        return ValidationResult(passed=True, step="security", sql=sql)
