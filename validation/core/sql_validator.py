"""
validation/sql_validator.py
────────────────────────────
10-step SQL validation pipeline orchestrator.
"""

from __future__ import annotations
from config.settings import settings
from typing import Any, Callable

import sqlglot

from models.schema import ValidationResult
from utils.logging_config import get_logger
from .context import ValidationContext

from ..ast.syntax import SyntaxValidator, PlaceholderValidator, AliasValidator
from ..ast.aggregation import AggregationValidator, GroupByAlignmentValidator
from ..schema.schema_validator import SchemaValidator
from ..ast.joins import JoinValidator
from ..ast.safety import SafetyValidator
from ..security.validation import SecurityTransformer
from ..execution.cost import CostValidator
from ..semantic.semantic_checks import SemanticValidator, HardcodedLiteralValidator

logger = get_logger(__name__)

def build_default_pipeline(
    fk_graph: Any = None,
    get_conn: Callable[[], Any] | None = None,
    release_conn: Callable[[Any], None] | None = None,
    db_dsn: str | None = None,
    tenant_scoped_tables: set[str] | None = None,
):
    """
    Build the default sequence of validation steps.
    """
    tenant_scoped_tables = tenant_scoped_tables or set()
    return [
        SyntaxValidator(),
        PlaceholderValidator(),
        AliasValidator(),
        SchemaValidator(),
        JoinValidator(fk_graph=fk_graph),
        SafetyValidator(),
        SecurityTransformer(tenant_scoped_tables=tenant_scoped_tables),
        GroupByAlignmentValidator(),
        CostValidator(get_conn=get_conn, release_conn=release_conn, db_dsn=db_dsn),
        SemanticValidator(),
        HardcodedLiteralValidator(),
        AggregationValidator(),
    ]

class SQLValidator:
    """
    Validates generated SQL through a pluggable validation pipeline.

    Usage:
        validator = SQLValidator(schema_map=tables, db_conn=conn)
        result    = validator.validate(sql, tables_used=["board"])
    """

    def __init__(
        self,
        schema_map: dict[str, Any],
        get_connection: Any = None,
        release_conn:   Any = None,
        db_dsn:         str | None = None,
        fk_graph:       Any = None,
        pipeline: list | None = None,
    ) -> None:
        self.schema_map = schema_map

        self._tenant_scoped_tables: set[str] = {
            name for name, inv in schema_map.items()
            if "board_id" in inv.columns or "course_id" in inv.columns
        }
        logger.info(
            component="sql_validator",
            event="tenant_tables_derived",
            count=len(self._tenant_scoped_tables),
            tables=sorted(self._tenant_scoped_tables),
        )

        self.pipeline = pipeline or build_default_pipeline(
            fk_graph=fk_graph,
            get_conn=get_connection,
            release_conn=release_conn,
            db_dsn=db_dsn,
            tenant_scoped_tables=self._tenant_scoped_tables,
        )

    def validate(
        self,
        sql:            str,
        tables_used:    list[str]  = None,
        user_context:   dict | None = None,
        original_query: str | None = None,
    ) -> ValidationResult:
        """
        Run all validation steps in order.
        Returns immediately on first failure.
        """
        tables_used  = tables_used  or []
        user_context = user_context or {}

        try:
            ast_list = sqlglot.parse(sql, dialect="postgres")
        except sqlglot.errors.ParseError:
            ast_list = None

        ctx = ValidationContext(
            sql=sql,
            ast=ast_list,
            schema_map=self.schema_map,
            fk_graph=None, # Already injected into JoinValidator
            tables_used=tables_used,
            user_context=user_context,
            original_query=original_query,
            sql_tables=set(),
            alias_map={},
            cte_names=set(),
            working_sql=sql
        )

        for step in self.pipeline:
            # Semantic checks depend on original_query
            if step.name in ("SemanticValidator", "HardcodedLiteralValidator"):
                if not original_query:
                    continue
            
            result = step.run(ctx)
            if not result.passed:
                return result

        final_sql = ctx.working_sql or ctx.sql
        logger.info(
            component="sql_validator",
            event="validation_passed",
            sql_preview=final_sql[:80],
            tables=tables_used,
        )
        return ValidationResult(passed=True, sql=final_sql)

class RetryValidator:
    """
    Wraps SQLValidator with self-correction retry / repair loop logic.

    If a validation step fails, this class orchestrates query recovery by:
      1. Formatting validation failures and original natural language questions into prompts.
      2. Querying the LLM generator to fix the error in the SQL.
      3. Running validation on the corrected SQL again.
    
    This loop continues until the query passes validation or the maximum retry limit is reached.
    """

    def __init__(
        self,
        validator:     SQLValidator,
        sql_generator,           # SQLGenerator ΓÇö injected to avoid circular import
        prompt_builder,          # PromptBuilder
    ) -> None:
        self.validator      = validator
        self.sql_generator  = sql_generator
        self.prompt_builder = prompt_builder

    def validate_with_retry(
        self,
        sql:             str,
        original_query:  str,
        tables_used:     list[str]  = None,
        user_context:    dict | None = None,
        schema_context:  str        = "",
        label_filters:   list[dict] = None,
        on_retry_fallback: callable = None,  # Callback to dynamically expand context on retry
        parsed_query                = None,
        max_retries:     int        = None,
        # ΓöÇΓöÇ Full-context params (Fix 1+4+5: retry context parity) ΓöÇΓöÇ
        schema_chunks:   list       = None,
        join_paths:      list[str]  = None,
        few_shots:       list       = None,
        tenant_context:  str        = "",
        # ΓöÇΓöÇ Column-cheatsheet pass-through (PHASE-1 FIX) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        # Forwarded as `tables=` into build_correction_prompt so the
        # retry sees the same positive-form aliasΓåÆcolumns list the
        # original attempt did.  Named schema_inventory here to avoid
        # confusion with `tables_used` (a list of table names).
        schema_inventory: dict      = None,
    ) -> tuple[ValidationResult, int]:
        """
        Validate SQL with up to max_retries correction attempts.

        Args:
            sql: The initial candidate SQL query string generated by the model.
            original_query: The user's natural language question.
            tables_used: List of database tables involved in the query.
            user_context: Key context mapping parameters (e.g. board_id, user_id).
            schema_context: Chunk schemas injected into the correction context.
            label_filters: RapidFuzz matching details to preserve quotes on strings.
            on_retry_fallback: Optional callback taking (attempt_num, tables_used, error_message)
                to fetch expanded context. Returns (chunks, join_paths) when full-context
                mode is active, or str when in legacy mode.
            schema_chunks: Full list of SemanticChunk objects for the correction prompt.
            join_paths: FK join path descriptions from the Steiner tree.
            few_shots: Few-shot example chunks for the correction prompt.
            tenant_context: RLS/security context string.

        Returns:
            A tuple of (final_ValidationResult, retry_count).
        """
        tables_used   = tables_used   or []
        user_context  = user_context  or {}
        label_filters = label_filters or []
        max_retries   = max_retries if max_retries is not None else settings.validation.max_retries

        # Perform the first check against the candidate SQL.
        # Pass original_query for Step 7 semantic heuristic checks.
        result  = self.validator.validate(sql, tables_used, user_context, original_query=original_query)
        retries = 0

        # Track the error signature so we can stop early when a correction makes
        # no progress. Re-retrieval on retry often returns the same chunks, so a
        # schema/semantic error can repeat verbatim for every remaining attempt
        # (observed: 48 queries hitting max retries with an identical message).
        # Bailing on the first identical repeat saves ~75% of failure latency and
        # stops the Phase-2 failure corpus filling with duplicate dead-ends.
        last_error_sig = (result.step, result.message) if not result.passed else None

        # Loop to correct/repair failed SQL candidate strings based on validation step failures.
        # This acts as a feedback loop between the validator (providing error diagnostics) and the generator (fixing errors).
        while not result.passed and retries < max_retries:
            retries += 1
            logger.info(
                component="retry_validator",
                event="retrying",
                attempt=retries,
                step=result.step,
                error=result.message[:100],
            )

            # Dynamically expand schema context if a fallback callback is provided.
            # On subsequent retries, the initial schema context might have been too restricted,
            # so we request broader database schema metadata (expanded context budget) to guide correction.
            # Fix 2: on_retry_fallback now receives the error message so it can extract
            # error-mentioned tables and add them to the retrieval scope.
            if on_retry_fallback:
                try:
                    expanded = on_retry_fallback(retries, tables_used, result.message)
                    # Full-context mode: callback returns (chunks, join_paths)
                    if isinstance(expanded, tuple) and len(expanded) == 2:
                        schema_chunks, join_paths = expanded
                    else:
                        # Legacy mode: callback returns a string
                        schema_context = expanded
                except Exception as exc:
                    logger.warning(
                        component="retry_validator",
                        event="retry_fallback_failed",
                        error=str(exc)
                    )

            # Build repair/correction instructions for the LLM.
            # Fix 1+4+5: when full context objects are available, the correction
            # prompt rebuilds the entire initial-quality prompt (via build()) and
            # appends the error feedback ΓÇö ensuring the retry sees ALL schema DDL,
            # join recipes, few-shots, and glossary, not just a 10-chunk subset.
            correction_prompt = self.prompt_builder.build_correction_prompt(
                original_query = original_query,
                failed_sql     = sql,
                error_message  = result.message,
                schema_context = schema_context,
                label_filters  = label_filters,
                parsed_query   = parsed_query,
                schema_chunks  = schema_chunks,
                join_paths     = join_paths,
                few_shots      = few_shots,
                tenant_context = tenant_context,
                tables         = schema_inventory,
            )

            # Run the SQL generator model on the correction prompt to generate a repaired candidate.
            corrected   = self.sql_generator.generate(correction_prompt)
            sql         = corrected.sql
            tables_used = corrected.tables_used or tables_used

            if not sql:
                # Break early if the generator returned empty SQL
                break

            # Validate the corrected SQL candidate again.
            # This loops back to check if the new query passes all validation steps.
            result = self.validator.validate(sql, tables_used, user_context, original_query=original_query)

            # Stall detection: if the correction reproduced the exact same error,
            # further identical retries will not help -- stop and return.
            new_sig = (result.step, result.message) if not result.passed else None
            if new_sig is not None and new_sig == last_error_sig:
                logger.info(
                    component="retry_validator",
                    event="retry_stalled",
                    attempt=retries,
                    step=result.step,
                    note="identical error reproduced; aborting retries early",
                )
                break
            last_error_sig = new_sig

        return result, retries