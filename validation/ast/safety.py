import re
import sqlglot.errors
import sqlglot.expressions as exp
from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

class SafetyValidator(BaseValidationStep):
    name = "SafetyValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 3a: Guard rails - Block DML/DDL statements.
        """
        sql = ctx.working_sql or ctx.sql
        
        # ── Layer 1: AST (Abstract Syntax Tree) Inspection ────────────────
        _DML_NODES = (
            exp.Insert, exp.Update, exp.Delete,
            exp.Drop, exp.Create, exp.Command,
            exp.Grant, exp.Revoke,
        )

        if ctx.ast:
            for stmt in ctx.ast:
                if stmt is None:
                    continue
                if not isinstance(stmt, (exp.Select, exp.With)):
                    kind = type(stmt).__name__
                    return ValidationResult(
                        passed=False, step="safety",
                        message=f"Non-SELECT statement detected by AST: {kind}. "
                                f"Only SELECT queries are permitted.",
                        sql=sql,
                    )
                for node in stmt.walk():
                    if isinstance(node, _DML_NODES):
                        kind = type(node).__name__
                        return ValidationResult(
                            passed=False, step="safety",
                            message=f"DML/DDL node '{kind}' found in query tree. "
                                    f"Only SELECT queries are permitted.",
                            sql=sql,
                        )
        else:
            # ── Layer 2: Blocked-Pattern Regex Fallback ──────────────────────
            match = settings.validation.blocked_pattern.search(sql)
            if match:
                return ValidationResult(
                    passed=False, step="safety",
                    message=f"Blocked keyword '{match.group(0).upper()}' detected. "
                            f"Only SELECT queries are permitted.",
                    sql=sql,
                )

        return ValidationResult(passed=True, step="safety", sql=sql)
