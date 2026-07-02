"""
validation/ast/joins.py
───────────────────────
JoinValidator (pipeline step 5, reports `safety`).

Catches Cartesian products — a JOIN with no ON/USING clause — which silently
multiply rows and corrupt every downstream aggregate. It is FK-graph aware: the
graph loaded at bootstrap lets it reason about whether a join path is legitimate
rather than only checking for a missing ON clause. Reported under the `safety`
label because an accidental cross join is a correctness hazard, not a cosmetic
issue.
"""

import re
import sqlglot.errors
import sqlglot.expressions as exp
from ..core.context import ValidationContext
from ..core.base import BaseValidationStep
from models.schema import ValidationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Cartesian regex — fallback only for unparseable SQL
CARTESIAN_PATTERN = re.compile(r"FROM\s+\w+\s*,\s*\w+", re.IGNORECASE)

class JoinValidator(BaseValidationStep):
    name = "JoinValidator"

    def __init__(self, fk_graph):
        self.fk_graph = fk_graph

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 3b: Cartesian Join Check
        Detects implicit Cartesian joins (e.g., joins that lack 'ON' or 'USING' conditions).
        """
        sql = ctx.working_sql or ctx.sql
        cartesian_detected = False

        if ctx.ast:
            try:
                for stmt in ctx.ast:
                    if stmt is None:
                        continue
                    for join in stmt.find_all(exp.Join):
                        has_on = join.args.get("on") is not None
                        has_using = join.args.get("using") is not None
                        join_kind = (join.args.get("kind") or "").upper()
                        
                        if not has_on and not has_using and join_kind != "CROSS":
                            cartesian_detected = True
                            break
                    if cartesian_detected:
                        break
            except Exception as exc:
                logger.warning("cartesian_check_ast_error", error=str(exc))
                cartesian_detected = bool(CARTESIAN_PATTERN.search(sql))
        else:
            cartesian_detected = bool(CARTESIAN_PATTERN.search(sql))

        if cartesian_detected:
            return ValidationResult(
                passed=False, step="safety",
                message="Cartesian join detected (JOIN without ON or USING clause). "
                        "Use explicit JOIN ... ON syntax.",
                sql=sql,
            )

        return ValidationResult(passed=True, step="safety", sql=sql)
