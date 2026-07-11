"""
pipeline/runner.py
──────────────────
End-to-end Phase 1B query pipeline.

Flow:
  QueryUnderstanding → Hybrid Retrieval → PromptBuilder
  → SQLGenerator → RetryValidator → Execution → QueryResult

Failure logging: every failed query (after retries exhausted) is written
to the failures/ directory as a JSON file. These become the training corpus
for Phase 2 fine-tuning.

Observability: every request produces a complete structured log entry
covering all pipeline stages, timings, and outcomes.

FIXES IN THIS VERSION
─────────────────────
C3  — _execute() success path and _step_cost() now call conn.rollback()
      before returning the connection to the pool.  Without this the
      connection was returned idle-in-transaction, which:
        1. Caused replication conflicts and vacuum bloat on the read replica.
        2. Left SET LOCAL app.current_user_id active on the connection,
           leaking User A's RLS identity to User B's query (security bug).
      rollback() is always safe on a read-only replica.
      Fix is applied in _release_connection() so all callers benefit
      automatically, including SQLValidator._step_cost().

M8  — schema_context passed to correction prompts is now built from chunks
      for tables mentioned in the failed SQL / error message rather than
      always the first 5 chunks.  This ensures the correction prompt contains
      the relevant schema when the failing column or table is not in the
      top-ranked chunks.

LOW — datetime.utcnow() replaced with datetime.now(timezone.utc) (Python 3.12+
      deprecation).
LOW — dry_run type hint corrected to bool | None.
LOW — traceback.print_exc() calls replaced with logger.exception().
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.pool
import sqlglot
import sqlglot.expressions as exp
import structlog


def _outer_query_has_limit(sql: str) -> bool:
    """
    Return True if the outermost SELECT in sql already has a LIMIT clause.
    Falls back to False (adds LIMIT) on any parse error — safe default.
    """
    try:
        stmt = sqlglot.parse_one(sql, dialect="postgres")
        if stmt is None:
            return False
        outer = stmt if isinstance(stmt, exp.Select) else stmt.find(exp.Select)
        if outer is None:
            return False
        return outer.args.get("limit") is not None
    except Exception:
        return False


def _extract_tables_from_error(error_message: str, valid_tables: set[str]) -> list[str]:
    """
    Fix 2: Extract table names mentioned in validator error messages.

    When the validator says 'Column X does not exist on table Y, suggests Z',
    the correct table (Z) may not be in tables_used.  This function finds
    valid table names in the error text so the expanded retrieval includes
    schema chunks for tables the LLM needs to fix the error.
    """
    words = set(re.findall(r'[a-z_]+', error_message.lower()))
    return [t for t in valid_tables if t in words]


from config.settings import settings
from generation.prompt_builder import PromptBuilder
from generation.query_understanding import QueryUnderstanding
from generation.sql_generator import SQLGenerator
from mcp_tools.client import (
    QdrantMCPClient,
    OpenSearchMCPClient,
    call_postgres_execute,
    call_postgres_explain,
    call_corpus_log_failure,
    MCPCallError,
)
from indexing.opensearch_indexer import OpenSearchIndexer
from indexing.qdrant_indexer import QdrantIndexer
from models.schema import QueryResult, TableInventory
from retrieval.orchestrator import RetrievalOrchestrator
from retrieval.reranker import CrossEncoderReranker
from utils.logging_config import get_logger
from validation.core.sql_validator import RetryValidator, SQLValidator
from validation.semantic.logical_audit import run_logical_audit
from validation.utils.autofix import attempt_tautological_autofix

logger = get_logger(__name__)

# ── Module-level connection pool ──────────────────────────────────────────────
_pg_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pg_pool_lock = threading.Lock()

# REVIEW FIX (NEW-M3): psycopg2.pool.ThreadedConnectionPool.getconn() has no
# native timeout — when all pool_max connections are checked out (e.g. under
# concurrent MCP server load), getconn() either blocks indefinitely or raises
# PoolError immediately depending on the exact contention pattern, with no
# bounded wait in between. A single stuck or slow query can starve every
# subsequent request with no way to recover except restarting the process.
# This constant bounds how long _acquire_connection_with_timeout() will poll
# before giving up and raising PoolTimeoutError — a distinct, catchable
# exception rather than a silent None (None already means "PG_HOST not
# configured" elsewhere in this file; conflating the two would make pool
# exhaustion look identical to "no database configured").
_POOL_ACQUIRE_TIMEOUT_SECONDS = 10
_POOL_ACQUIRE_POLL_INTERVAL_SECONDS = 0.1


class PoolTimeoutError(Exception):
    """
    Raised when a connection could not be acquired from the pool within
    _POOL_ACQUIRE_TIMEOUT_SECONDS. Distinct from psycopg2.pool.PoolError so
    callers can tell "pool exhausted, try again later" apart from other
    pool-level failures and from the unrelated "no DB configured" case
    (which _get_connection() signals by returning None, not raising).
    """
    pass


def _acquire_connection_with_timeout(pool: psycopg2.pool.ThreadedConnectionPool):
    """
    REVIEW FIX (NEW-M3): poll pool.getconn() with a bounded total wait
    instead of calling it once and letting it block forever or raise
    immediately. psycopg2's ThreadedConnectionPool.getconn() raises
    PoolError("connection pool exhausted") synchronously when no slot is
    free — there's no built-in wait-and-retry. This wrapper retries on that
    specific error for up to _POOL_ACQUIRE_TIMEOUT_SECONDS, polling every
    _POOL_ACQUIRE_POLL_INTERVAL_SECONDS, giving slow queries from other
    requests a chance to finish and free a slot before giving up.
    """
    import psycopg2.pool as _pool_mod

    deadline = time.time() + _POOL_ACQUIRE_TIMEOUT_SECONDS
    while True:
        try:
            return pool.getconn()
        except _pool_mod.PoolError as exc:
            if time.time() >= deadline:
                logger.error(
                    component="pipeline",
                    event="pool_acquire_timeout",
                    timeout_s=_POOL_ACQUIRE_TIMEOUT_SECONDS,
                    error=str(exc),
                )
                raise PoolTimeoutError(
                    f"Could not acquire a database connection within "
                    f"{_POOL_ACQUIRE_TIMEOUT_SECONDS}s — pool exhausted "
                    f"(max={settings.postgres.pool_max}). Try again shortly."
                ) from exc
            time.sleep(_POOL_ACQUIRE_POLL_INTERVAL_SECONDS)


def _get_connection():
    """
    Single authoritative connection factory shared by _execute() and
    SQLValidator._step_cost().  Returns None when PG_HOST is blank.

    REVIEW FIX (NEW-M3): raises PoolTimeoutError (not None) when PG_HOST IS
    configured but the pool is exhausted and stays exhausted past the
    timeout. Returning None for that case would be indistinguishable from
    "no database configured", which is a different situation requiring a
    different operator response (one is a config choice, the other is a
    capacity problem under load).
    """
    global _pg_pool

    pg = settings.postgres
    if not pg.host:
        return None

    connect_kwargs = dict(
        host     = pg.host,
        port     = pg.port,
        dbname   = pg.database,
        user     = pg.user,
        password = pg.password,
        options  = (
            f"-c statement_timeout={pg.statement_timeout_ms} "
            f"-c default_transaction_read_only=on"
        ),
    )

    if _pg_pool is None:
        with _pg_pool_lock:
            if _pg_pool is None:
                _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn = pg.pool_min,
                    maxconn = pg.pool_max,
                    **connect_kwargs,
                )
                logger.info(
                    component="pipeline",
                    event="pool_created",
                    min=pg.pool_min,
                    max=pg.pool_max,
                )

    return _acquire_connection_with_timeout(_pg_pool)


def _release_connection(conn) -> None:
    """
    Return a connection to the pool.

    FIX-C3: rollback() is called unconditionally before putconn() so the
    connection is never returned idle-in-transaction.  On a read-only replica
    every transaction is effectively a no-op and rollback() is always safe.
    This also ensures SET LOCAL app.current_user_id (applied inside _execute)
    is cleared before the connection is reused by another request — preventing
    RLS identity leaking between requests.
    """
    if _pg_pool is not None and conn is not None:
        try:
            conn.rollback()
        except Exception:
            pass
        _pg_pool.putconn(conn)


class PipelineRunner:
    """
    Orchestrates the full NL→SQL pipeline.

    All components are lazy-initialised on first use. The runner is
    designed to be instantiated once and reused across many requests.

    Usage:
        runner = PipelineRunner(tables=tables, fk_graph=G)
        result = runner.run("Show all scripts pending third evaluation in board 5")
    """

    def __init__(
        self,
        tables:   dict[str, TableInventory],
        fk_graph: nx.DiGraph,
    ) -> None:
        self.tables   = tables
        self.fk_graph = fk_graph

        # Components initialised lazily
        self._qdrant:     QdrantIndexer     | None = None
        self._opensearch: OpenSearchIndexer | None = None
        self._reranker:   CrossEncoderReranker | None = None
        self._orchestrator: RetrievalOrchestrator | None = None
        self._query_understanding: QueryUnderstanding | None = None
        self._prompt_builder:      PromptBuilder     | None = None
        self._sql_generator:       SQLGenerator      | None = None
        self._validator:           SQLValidator       | None = None
        self._retry_validator:     RetryValidator     | None = None

    # ─────────────────────────────────────────────────────────────────────
    # Lazy component accessors
    # ─────────────────────────────────────────────────────────────────────

    @property
    def qdrant(self) -> QdrantIndexer | QdrantMCPClient:
        if not self._qdrant:
            # Use MCP client when USE_MCP_SERVERS=true, direct client otherwise.
            # Both have identical method signatures — no other code changes needed.
            self._qdrant = (
                QdrantMCPClient() if settings.use_mcp_servers
                else QdrantIndexer()
            )
        return self._qdrant

    @property
    def opensearch(self) -> OpenSearchIndexer | OpenSearchMCPClient:
        if not self._opensearch:
            self._opensearch = (
                OpenSearchMCPClient() if settings.use_mcp_servers
                else OpenSearchIndexer()
            )
        return self._opensearch

    @property
    def reranker(self) -> CrossEncoderReranker | None:
        if settings.reranker.enabled and not self._reranker:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    @property
    def orchestrator(self) -> RetrievalOrchestrator:
        if not self._orchestrator:
            self._orchestrator = RetrievalOrchestrator(
                qdrant_indexer     = self.qdrant,
                opensearch_indexer = self.opensearch,
                fk_graph           = self.fk_graph,
                reranker           = self.reranker,
            )
        return self._orchestrator

    @property
    def query_understanding(self) -> QueryUnderstanding:
        if not self._query_understanding:
            self._query_understanding = QueryUnderstanding(settings.glossary_path)
        return self._query_understanding

    @property
    def prompt_builder(self) -> PromptBuilder:
        if not self._prompt_builder:
            self._prompt_builder = PromptBuilder()
        return self._prompt_builder

    @property
    def sql_generator(self) -> SQLGenerator:
        if not self._sql_generator:
            self._sql_generator = SQLGenerator()
        return self._sql_generator

    @property
    def validator(self) -> SQLValidator:
        if not self._validator:
            self._validator = SQLValidator(
                schema_map     = {k: v for k, v in self.tables.items()},
                get_connection = _get_connection,
                release_conn   = _release_connection,
                fk_graph       = self.fk_graph,
            )
        return self._validator

    @property
    def retry_validator(self) -> RetryValidator:
        if not self._retry_validator:
            self._retry_validator = RetryValidator(
                validator      = self.validator,
                sql_generator  = self.sql_generator,
                prompt_builder = self.prompt_builder,
            )
        return self._retry_validator

    # ─────────────────────────────────────────────────────────────────────
    # Main pipeline entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        nl_query:     str,
        dry_run:      bool | None = None,   # LOW: corrected type hint
        user_context: dict | None = None,
    ) -> QueryResult:
        """
        Execute the full NL→SQL pipeline for a user query.

        dry_run=True  — validate SQL without executing against the database
        dry_run=False — validate + execute (read-only replica)
        """
        dry_run      = settings.dry_run_default if dry_run is None else dry_run
        user_context = user_context or {}
        request_id   = str(uuid.uuid4())[:8]
        t_start      = time.time()
        timings:  dict[str, float] = {}

        structlog.contextvars.bind_contextvars(request_id=request_id)

        try:
            return self._run_pipeline(
                nl_query     = nl_query,
                dry_run      = dry_run,
                user_context = user_context,
                request_id   = request_id,
                t_start      = t_start,
                timings      = timings,
            )
        finally:
            structlog.contextvars.unbind_contextvars("request_id")

    def _run_pipeline(
        self,
        nl_query:     str,
        dry_run:      bool,
        user_context: dict,
        request_id:   str,
        t_start:      float,
        timings:      dict,
    ) -> QueryResult:
        """Inner pipeline body — always called from run() inside a try/finally."""
        logger.info(component="pipeline", event="request_start", query=nl_query[:100], dry_run=dry_run)

        # ── Step 1: Query Understanding ───────────────────────────────────
        t0     = time.time()
        parsed = self.query_understanding.process(nl_query)
        timings["understanding_ms"] = round((time.time() - t0) * 1000)

        if parsed.is_ambiguous:
            return QueryResult(
                nl_query     = nl_query,
                sql          = "",
                # Pass clarifications as a list — interface.py checks isinstance(list)
                # to detect ambiguous results and route to _handle_ambiguous().
                # Do NOT join to a string here — that loses the INCOMPLETE_PREFIX
                # signal that the CLI needs to choose between menu and text-prompt.
                explanation  = parsed.clarifications,
                tables_used  = [],
                confidence   = 0.0,
                intent       = parsed.intent.value,
                success      = False,
                error        = "ambiguous_query",
                latency_ms   = timings,
            )

        # Issue 5 fix: use clean_query (markers stripped) for retrieval and prompt.
        # parsed.normalised still contains "— specifically:" / "— value:" if this
        # is a clarified re-run. The LLM must never see those markers in [QUERY].
        # clean_query = query with markers removed.
        # clarification_note = extracted suffix (e.g. "below 40%") injected as
        # a separate [CLARIFICATION] block by prompt_builder.build().
        query_for_pipeline = parsed.clean_query

        # ── Step 2: Hybrid Retrieval ──────────────────────────────────────
        t0 = time.time()
        schema_chunks, retrieval_meta = self.orchestrator.retrieve(
            query_text    = query_for_pipeline,   # Issue 5: clean, marker-free
            entity_tables = parsed.entities,
            intent        = parsed.intent.value,
        )
        few_shots = self.orchestrator.get_few_shot_examples(
            query_text = query_for_pipeline,      # Issue 5: clean, marker-free
            top_k      = 3,
        )
        timings["retrieval_ms"] = round((time.time() - t0) * 1000)

        join_path_text: list[str] = retrieval_meta.get("join_paths", [])

        # ── Step 3: Prompt Construction ───────────────────────────────────
        # FIX-F1 — profile switch. "ft" serves the fine-tuned model the exact
        # training distribution (short system role + budgeted SCHEMA+QUESTION
        # user turn, rendered by the SAME function the preprocessor uses).
        # "full" (default) keeps the base-model rich prompt unchanged.
        t0 = time.time()
        _system: str | None = None
        if settings.llm.prompt_profile == "ft":
            ft_prompt = self.prompt_builder.build_ft(
                parsed_query  = parsed,
                schema_chunks = schema_chunks,
                join_paths    = join_path_text,
                few_shots     = few_shots,
            )
            prompt, _system = ft_prompt["user"], ft_prompt["system"]
        else:
            prompt = self.prompt_builder.build(
                parsed_query       = parsed,
                schema_chunks      = schema_chunks,
                join_paths         = join_path_text,
                few_shots          = few_shots,
                tenant_context     = user_context.get("tenant_context", ""),
                # Issue 5: pass clean query and note separately so prompt_builder
                # can emit them as distinct [QUERY] and [CLARIFICATION] blocks.
                # prompt_builder must use parsed.clean_query for [QUERY], not
                # parsed.normalised which may contain marker text.
                clarification_note = parsed.clarification_note,
                # RapidFuzz: pass resolved course code if available so prompt_builder
                # can inject a concrete JOIN hint in the [CLARIFICATION] block.
                course_code_match  = parsed.course_code_match,
                # PHASE-1 FIX (cheatsheet): pass full TableInventory map so the
                # builder can emit the [COLUMN CHEATSHEET] block.  See
                # prompt_builder.build() docstring for rationale and impact.
                tables             = self.tables,
            )
        timings["prompt_ms"] = round((time.time() - t0) * 1000)

        # ── Step 4: SQL Generation ────────────────────────────────────────
        t0          = time.time()
        generated   = self.sql_generator.generate(prompt, system=_system)
        timings["generation_ms"] = round((time.time() - t0) * 1000)

        if generated.prompt_tokens is not None:
            retrieval_meta["llm_prompt_tokens"] = generated.prompt_tokens
        if generated.completion_tokens is not None:
            retrieval_meta["llm_completion_tokens"] = generated.completion_tokens

        if not generated.sql:
            return self._failure_result(
                nl_query      = nl_query,
                error         = "LLM produced empty SQL output.",
                parsed_intent = parsed.intent.value,
                timings       = timings,
                retrieval_meta= retrieval_meta,
                request_id    = request_id,
            )

        # ── Step 5: Validation + Retry ────────────────────────────────────
        t0 = time.time()

        # ── Retry callback: expand retrieval on subsequent attempts ─────
        # Fix 1+2+4+5: the callback now returns (chunks, join_paths) tuples
        # instead of a pre-rendered string.  This lets the correction prompt
        # rebuild the full initial-quality prompt from structured chunks.
        # Fix 2: error_message is parsed for table names so the expanded
        # retrieval includes the correct table the LLM needs to fix the error.
        def get_expanded_context(attempt_num: int, current_tables: list[str], error_message: str):
            base_budget = settings.retrieval.context_budget_tokens
            max_budget = settings.retrieval.max_context_budget_tokens
            expanded_budget = min(int(base_budget * (1.3 ** attempt_num)), max_budget)

            # Fix 2: extract table names from error message so the retrieval
            # scope includes tables the LLM needs (e.g. the table where a
            # column actually exists, not the table the LLM wrongly used).
            error_tables = _extract_tables_from_error(
                error_message, set(t.lower() for t in self.tables.keys())
            )

            all_tables = list(set(
                (parsed.entities or []) + current_tables + error_tables
            ))

            logger.info(
                component="pipeline",
                event="expanding_retry_context",
                attempt=attempt_num,
                expanded_budget=expanded_budget,
                error_tables=error_tables,
                total_tables=len(all_tables),
            )

            expanded_chunks, expanded_meta = self.orchestrator.retrieve(
                query_text    = query_for_pipeline,
                entity_tables = all_tables,
                intent        = parsed.intent.value,
                budget_tokens = expanded_budget,
            )
            expanded_join_paths = expanded_meta.get("join_paths", [])
            return expanded_chunks, expanded_join_paths

        val_result, retries = self.retry_validator.validate_with_retry(
            sql            = generated.sql,
            original_query = query_for_pipeline,   # Issue 5: clean, marker-free
            tables_used    = generated.tables_used,
            user_context   = user_context,
            label_filters  = parsed.label_filters,
            on_retry_fallback = get_expanded_context,
            parsed_query   = parsed,
            # Fix 1+4+5: pass full context objects so the correction prompt
            # rebuilds the entire initial-quality prompt, not a 10-chunk subset.
            schema_chunks  = schema_chunks,
            join_paths     = join_path_text,
            few_shots      = few_shots,
            tenant_context = user_context.get("tenant_context", ""),
            # PHASE-1 FIX (cheatsheet): forward the full TableInventory map
            # so the correction prompt can re-emit the [COLUMN CHEATSHEET].
            schema_inventory = self.tables,
        )
        timings["validation_ms"] = round((time.time() - t0) * 1000)

        # ── Step 5.5: Confidence Calibration ──────────────────────────────
        # Implements Recommendation 7: Calibration model lowering confidence
        # for risky query features, triggering an automatic fallback self-correction.
        # H-2 fix: cap total retries (normal + calibration) to max_retries.
        if val_result.passed:
            validated_sql = val_result.sql or generated.sql
            sql_lower = validated_sql.lower()
            calibrated_conf = generated.confidence
            
            from utils.heuristics import HEURISTICS
            calib_rules = HEURISTICS.get('confidence_calibration', {})
            
            # Penalize ILIKE/LIKE on IDs
            ilike_rule = calib_rules.get('ilike_on_ids', {})
            ilike_regex = ilike_rule.get('regex', r'\b\w+_id\s+(i)?like\s+')
            if re.search(ilike_regex, sql_lower):
                calibrated_conf -= ilike_rule.get('penalty', 0.15)
                
            # Penalize excessive nesting
            nest_rule = calib_rules.get('excessive_nesting', {})
            if sql_lower.count("select") > nest_rule.get('max_select_count', 3):
                calibrated_conf -= nest_rule.get('penalty', 0.10)
                
            generated.confidence = max(0.0, round(calibrated_conf, 2))
            
            # H-2 fix: remaining_budget prevents total retries from exceeding max_retries.
            remaining_budget = settings.validation.max_retries - retries
            if generated.confidence < 0.80 and remaining_budget > 0:
                logger.info(
                    component="pipeline",
                    event="confidence_calibration_retry",
                    confidence=generated.confidence,
                    sql_preview=validated_sql[:80]
                )
                # Force one more retry with full context
                correction_prompt = self.prompt_builder.build_correction_prompt(
                    original_query = query_for_pipeline,
                    failed_sql     = validated_sql,
                    error_message  = f"Query confidence dropped to {generated.confidence} (below 0.80) due to risky patterns like ILIKE on IDs or excessive nesting. Please simplify the query and use exact matches for IDs.",
                    label_filters  = [],
                    parsed_query   = parsed,
                    schema_chunks  = schema_chunks,
                    join_paths     = join_path_text,
                    few_shots      = few_shots,
                    tenant_context = user_context.get("tenant_context", ""),
                    tables         = self.tables,
                )
                generated = self.sql_generator.generate(correction_prompt, system=_system)
                retries += 1
                
                if generated.sql:
                    # H-2 fix: limit validate_with_retry to remaining budget minus 1
                    capped_remaining = max(0, settings.validation.max_retries - retries)
                    val_result, additional_retries = self.retry_validator.validate_with_retry(
                        sql            = generated.sql,
                        original_query = query_for_pipeline,
                        tables_used    = generated.tables_used,
                        user_context   = user_context,
                        on_retry_fallback = get_expanded_context,
                        max_retries    = capped_remaining,
                        parsed_query   = parsed,
                        schema_chunks  = schema_chunks,
                        join_paths     = join_path_text,
                        few_shots      = few_shots,
                        tenant_context = user_context.get("tenant_context", ""),
                        schema_inventory = self.tables,
                    )
                    retries += additional_retries

        if not val_result.passed:
            # FIX: log the SQL that ACTUALLY failed. validate_with_retry may have
            # rewritten `generated.sql` across correction passes; val_result.sql
            # holds the final failing string that the error message refers to.
            # Logging generated.sql here recorded a clean original alongside an
            # error about a different (corrected) query -- confusing to debug and
            # corrupting the Phase 2 failure corpus.
            return self._failure_result(
                nl_query       = nl_query,
                error          = f"Validation failed ({val_result.step}): {val_result.message}",
                parsed_intent  = parsed.intent.value,
                timings        = timings,
                retrieval_meta = retrieval_meta,
                failed_sql     = val_result.sql or generated.sql,
                retries        = retries,
                request_id     = request_id,
            )

        validated_sql = val_result.sql or generated.sql

        # ── Step 5.8: Logical Audit ───────────────────────────────────────
        # Pure NL↔SQL alignment check: no DB access, no retries.
        # Detects semantic mismatches like missing AVG(), wrong anti-join
        # polarity, tautological aggregation.
        audit = run_logical_audit(
            nl_query=query_for_pipeline,
            sql=validated_sql,
            intent=parsed.intent.value,
            tables_used=generated.tables_used,
        )
        if audit.warnings:
            logger.warning(
                component="pipeline",
                event="logical_audit_warnings",
                warnings=audit.warnings,
                confidence_penalty=audit.confidence_penalty,
                requirement_coverage=audit.requirement_coverage,
                coverage_misses=audit.coverage_misses,
                sql_preview=validated_sql[:80],
            )
            generated.confidence = max(0.0, round(
                generated.confidence - audit.confidence_penalty, 2
            ))

        # ── Step 5.8b: Hard-fail handling (NEW, 2026-07-01) ────────────────
        # Some logical_audit checks (currently L4 anti-join polarity, L5
        # tautological aggregation) are deterministic -- there is no false-
        # positive risk once they fire, unlike the softer heuristic checks.
        # Previously ALL audit findings, including these, only reduced
        # confidence_penalty, which a batch run of confirmed-wrong queries
        # showed was insufficient: e.g. Q33 (anti-join reversed) displayed
        # confidence 0.8 after a 0.10 penalty from a 0.9 raw score, still
        # "green" and still labeled Success. audit.hard_fail=True routes
        # these through a real correction-or-block path instead of a
        # confidence adjustment on an otherwise-Success result.
        if audit.hard_fail:
            is_l5 = any(w.startswith("[L5]") for w in audit.warnings)
            fixed_sql = None
            fix_desc = None

            if is_l5:
                fixed_sql, fix_desc = attempt_tautological_autofix(validated_sql)
                if fixed_sql is not None:
                    re_val = self.validator.validate(
                        sql            = fixed_sql,
                        tables_used    = generated.tables_used,
                        user_context   = user_context,
                        original_query = query_for_pipeline,
                    )
                    if re_val.passed:
                        re_audit = run_logical_audit(
                            nl_query    = query_for_pipeline,
                            sql         = re_val.sql or fixed_sql,
                            intent      = parsed.intent.value,
                            tables_used = generated.tables_used,
                        )
                        if not re_audit.hard_fail:
                            validated_sql = re_val.sql or fixed_sql
                            audit = re_audit
                            logger.info(
                                component="pipeline",
                                event="hard_fail_autofix_accepted",
                                check="L5",
                                fix_description=fix_desc,
                            )
                        else:
                            fixed_sql = None  # re-audit still flags it -- don't trust it
                    else:
                        fixed_sql = None  # rewritten SQL failed real structural re-validation

            if fixed_sql is None:
                # No autofix available (L4) or autofix failed re-validation
                # (L5). Do not return this as Success -- block it here,
                # same as any other validation-step failure, rather than
                # let it fall through with a merely-reduced confidence
                # score. This does not claim the underlying question gets
                # answered on this attempt -- only that the pipeline stops
                # claiming it did.
                logger.warning(
                    component="pipeline",
                    event="hard_fail_blocked",
                    warnings=audit.warnings,
                    autofix_attempted=is_l5,
                )
                return self._failure_result(
                    nl_query       = nl_query,
                    error          = (
                        "Validation failed (logical_audit): "
                        + "; ".join(audit.warnings)
                    ),
                    parsed_intent  = parsed.intent.value,
                    timings        = timings,
                    retrieval_meta = retrieval_meta,
                    failed_sql     = validated_sql,
                    retries        = retries,
                    request_id     = request_id,
                )

        # NEW: log requirement_coverage even when no warnings fired —
        # gives observability into queries where the parser found a
        # contract and the SQL fully satisfied it.
        elif audit.requirement_coverage is not None:
            logger.info(
                component="pipeline",
                event="logical_audit_pass",
                requirement_coverage=audit.requirement_coverage,
            )

        # ── Step 5.9: Audit-driven retry (NEW) ────────────────────────────
        # When the NL→requirements audit found specific misses AND we
        # have retry budget remaining, regenerate once with the
        # misses fed into the correction prompt.  This closes the
        # loop between DETECTION (logical_audit.py L6-L9) and
        # CORRECTION — without this step, the audit would know what
        # is missing but the next attempt would have no idea what
        # to fix.
        #
        # Triggers only when:
        #   * audit.coverage_misses is non-empty (specific actionable items)
        #   * audit.requirement_coverage is below 0.7 (genuine gap, not
        #     a single soft miss)
        #   * retry budget remains
        #   * NOT a dry-run (saves a round-trip when there's no exec)
        #
        # On the regenerated SQL we re-run structural validation but
        # do NOT re-trigger another audit retry (one shot only — avoid
        # ping-pong between competing signals).
        remaining_budget = settings.validation.max_retries - retries
        should_audit_retry = (
            audit.requirement_coverage is not None
            and audit.requirement_coverage < 0.7
            and audit.coverage_misses
            and remaining_budget > 0
            and not dry_run
        )
        if should_audit_retry:
            logger.info(
                component="pipeline",
                event="audit_driven_retry",
                requirement_coverage=audit.requirement_coverage,
                coverage_misses=audit.coverage_misses,
            )
            correction_prompt = self.prompt_builder.build_correction_prompt(
                original_query = query_for_pipeline,
                failed_sql     = validated_sql,
                error_message  = (
                    f"The SQL passed structural validation but the audit "
                    f"found requirement_coverage={audit.requirement_coverage} "
                    f"— the SQL does not satisfy all requirements from the "
                    f"question."
                ),
                label_filters  = [],
                parsed_query   = parsed,
                schema_chunks  = schema_chunks,
                join_paths     = join_path_text,
                few_shots      = few_shots,
                tenant_context = user_context.get("tenant_context", ""),
                audit_misses   = audit.coverage_misses,    # NEW
                tables         = self.tables,
            )
            regenerated = self.sql_generator.generate(correction_prompt, system=_system)
            retries += 1
            if regenerated.sql:
                # Re-run structural validation on the regenerated SQL.
                # We do NOT call validate_with_retry here — one shot only;
                # the audit-retry is itself the corrective attempt.
                re_val = self.validator.validate(
                    sql            = regenerated.sql,
                    tables_used    = regenerated.tables_used,
                    user_context   = user_context,
                    original_query = query_for_pipeline,
                )
                if re_val.passed:
                    # Re-audit the regenerated SQL.  Use the NEW coverage
                    # as the final signal; if it improved, accept the
                    # regeneration.  If it regressed, keep the original.
                    re_audit = run_logical_audit(
                        nl_query    = query_for_pipeline,
                        sql         = re_val.sql or regenerated.sql,
                        intent      = parsed.intent.value,
                        tables_used = regenerated.tables_used,
                    )
                    re_cov = re_audit.requirement_coverage
                    accept_regen = (
                        re_cov is None
                        or audit.requirement_coverage is None
                        or re_cov >= audit.requirement_coverage
                    )
                    if accept_regen:
                        validated_sql = re_val.sql or regenerated.sql
                        generated     = regenerated
                        audit         = re_audit
                        logger.info(
                            component="pipeline",
                            event="audit_retry_accepted",
                            new_coverage=re_cov,
                        )
                    else:
                        logger.info(
                            component="pipeline",
                            event="audit_retry_rejected_regression",
                            old_coverage=audit.requirement_coverage,
                            new_coverage=re_cov,
                        )

        # ── Step 6: Execution ─────────────────────────────────────────────

        rows:      list[dict]  = []
        row_count: int         = 0

        if not dry_run:
            t0 = time.time()
            exec_result = self._execute(validated_sql, user_context)
            timings["execution_ms"] = round((time.time() - t0) * 1000)

            if "error" in exec_result:
                return self._failure_result(
                    nl_query       = nl_query,
                    error          = f"Execution error: {exec_result['error']}",
                    parsed_intent  = parsed.intent.value,
                    timings        = timings,
                    retrieval_meta = retrieval_meta,
                    failed_sql     = validated_sql,
                    retries        = retries,
                    request_id     = request_id,
                )
            rows      = exec_result["rows"]
            row_count = exec_result["row_count"]

        timings["total_ms"] = round((time.time() - t_start) * 1000)

        logger.info(
            component="pipeline",
            event="request_complete",
            intent=parsed.intent.value,
            tables=generated.tables_used,
            confidence=generated.confidence,
            retries=retries,
            dry_run=dry_run,
            row_count=row_count,
            **{k: v for k, v in timings.items()},
        )

        return QueryResult(
            nl_query       = nl_query,
            sql            = validated_sql,
            explanation    = generated.explanation,
            tables_used    = generated.tables_used,
            confidence     = generated.confidence,
            intent         = parsed.intent.value,
            rows           = rows,
            row_count      = row_count,
            dry_run        = dry_run,
            retries        = retries,
            success        = True,
            latency_ms     = timings,
            retrieval_meta = retrieval_meta,
            # NEW: structural confidence from NL-requirements audit
            requirement_coverage = audit.requirement_coverage,
            coverage_misses      = audit.coverage_misses,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Execution
    # ─────────────────────────────────────────────────────────────────────

    def _execute(self, sql: str, user_context: dict) -> dict[str, Any]:
        """
        Execute the validated SQL on the read-only PostgreSQL replica.

        Routes to MCP postgres_server (port 5012) when USE_MCP_SERVERS=true,
        or direct psycopg2 pool otherwise.
        C3 fix (rollback before pool release) is enforced in both paths:
          - MCP path: enforced inside postgres_server.py._release_conn()
          - Direct path: enforced in _release_connection() below
        """
        if settings.use_mcp_servers:
            # ── MCP path ───────────────────────────────────────────────────
            try:
                result = call_postgres_execute(
                    sql      = sql,
                    user_id  = user_context.get("user_id"),
                    max_rows = settings.postgres.max_rows,
                )
                if "error" in result:
                    return {"error": result["error"]}
                return {"rows": result["rows"], "row_count": result["row_count"]}
            except MCPCallError as exc:
                logger.exception("mcp_postgres_execute_error")
                return {"error": f"MCP postgres error: {exc}"}

        # ── Direct psycopg2 path (default) ─────────────────────────────────
        # REVIEW FIX (NEW-M3): PoolTimeoutError is now possible here when
        # PG_HOST is configured but the pool stayed exhausted past the
        # timeout — distinct from conn is None (PG_HOST not configured at
        # all), which is handled separately below.
        try:
            conn = _get_connection()
        except PoolTimeoutError as exc:
            logger.warning(component="pipeline", event="pool_timeout", error=str(exc))
            return {"error": str(exc)}

        if conn is None:
            return {"error": "PostgreSQL connection not configured."}

        try:
            cur = conn.cursor()
            rls_value = user_context.get("user_id")
            if settings.rls_variable and rls_value:
                # M-9 fix: wrap SET LOCAL in its own statement_timeout guard
                # to prevent hanging if the connection is degraded.
                cur.execute("SET LOCAL statement_timeout = '5s'")
                cur.execute(f"SET LOCAL {settings.rls_variable} = %s",
                            (str(rls_value),))
                # Restore the configured statement timeout for the actual query
                cur.execute(f"SET LOCAL statement_timeout = '{settings.postgres.statement_timeout_ms}ms'")

            limited_sql = sql.rstrip(";")
            if not _outer_query_has_limit(limited_sql):
                limited_sql = f"{limited_sql} LIMIT {settings.postgres.max_rows}"

            cur.execute(limited_sql)
            columns  = [desc[0] for desc in cur.description] if cur.description else []
            raw_rows = cur.fetchall()
            rows     = [dict(zip(columns, row)) for row in raw_rows]
            cur.close()
            return {"rows": rows, "row_count": len(rows)}

        except psycopg2.Error as exc:
            logger.exception("execution_error")
            return {"error": str(exc)}
        finally:
            # C3 fix: _release_connection calls rollback() before putconn()
            _release_connection(conn)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_correction_context(
        self,
        tables_used:   list[str],
        schema_chunks,
        join_paths:    list[str] = None,
    ) -> str:
        """
        FIX-M8: build a schema context string for correction prompts that
        prioritises chunks for the tables actually used in the generated SQL.

        Fuzzy grounding: resolves typoed/hallucinated table names in tables_used
        to the closest valid table in the inventory using rapidfuzz, ensuring
        that the retry prompt contains the correct schema context chunks.

        P2-FIX: when join_paths is provided, appends FK relationship paths
        to the context so the model can fix wrong-join errors using explicit
        FK metadata rather than guessing from column names.
        """
        from rapidfuzz import process, fuzz

        valid_tables = [t.lower() for t in self.tables.keys()]
        resolved_tables = set()

        for t in tables_used:
            t_lower = t.lower().strip()
            if not t_lower:
                continue
            if t_lower in valid_tables:
                resolved_tables.add(t_lower)
            else:
                # Fuzzy match to closest valid database table name
                match = process.extractOne(t_lower, valid_tables, scorer=fuzz.ratio)
                if match and match[1] >= 60.0:
                    resolved_tables.add(match[0])
                    logger.warning(
                        component="pipeline",
                        event="table_fuzzy_grounded",
                        hallucinated=t_lower,
                        resolved=match[0],
                        score=round(match[1], 2),
                    )

        relevant = [c for c in schema_chunks if c.table_name.lower() in resolved_tables]
        other    = [c for c in schema_chunks if c.table_name.lower() not in resolved_tables]

        # Change 4: Prioritise TABLE chunks (column definitions) for correction context.
        # A hallucinated column error needs TABLE chunks, not 8 FK_MAP chunks.
        from models.schema import ChunkType as _CT
        relevant_table = [c for c in relevant if c.chunk_type in (_CT.TABLE, _CT.VIEW)]
        relevant_fk    = [c for c in relevant if c.chunk_type == _CT.FK_MAP]
        relevant_other = [c for c in relevant if c.chunk_type not in (_CT.TABLE, _CT.VIEW, _CT.FK_MAP)]
        # TABLE chunks first (column definitions), then FK_MAP (join paths), then others
        # Fix 3: removed hard-caps ([:5], [:3], [:2]) that previously
        # truncated correction context to ~10 chunks regardless of budget.
        # Now includes all relevant chunks, with TABLE prioritised first.
        selected = relevant_table + relevant_fk + relevant_other + other[:3]
        context = "\n".join(c.text for c in selected)

        # P2-FIX: append FK relationship paths to the correction context.
        # These are computed by the NetworkX Steiner Tree traversal and tell
        # the model exactly which FK columns connect the relevant tables.
        # Without this, the model has to guess join paths from column names
        # alone — a key source of repeat failures on retry.
        if join_paths:
            context += "\n\n=== RELEVANT JOIN PATHS ===\n"
            context += "\n".join(join_paths)

        return context

    # ─────────────────────────────────────────────────────────────────────
    # Failure handling
    # ─────────────────────────────────────────────────────────────────────

    def _failure_result(
        self,
        nl_query:       str,
        error:          str,
        parsed_intent:  str    = "unknown",
        timings:        dict   = None,
        retrieval_meta: dict   = None,
        failed_sql:     str    = "",
        retries:        int    = 0,
        request_id:     str    = "",
    ) -> QueryResult:
        """
        Log failure to training corpus and return error QueryResult.

        When USE_MCP_SERVERS=true, _log_failure() returns the corpus entry_id
        from the MCP server. This is stored on the QueryResult so the CLI
        :correct command can call save_correction(entry_id, corrected_sql)
        without needing to scan the failures/ directory.
        """
        failure_entry_id = self._log_failure(
            nl_query, error, failed_sql, retries, request_id
        )

        return QueryResult(
            nl_query         = nl_query,
            sql              = failed_sql,
            explanation      = "",
            tables_used      = [],
            confidence       = 0.0,
            intent           = parsed_intent,
            success          = False,
            error            = error,
            retries          = retries,
            latency_ms       = timings or {},
            retrieval_meta   = retrieval_meta or {},
            failure_entry_id = failure_entry_id,
        )

    def _log_failure(
        self,
        nl_query:   str,
        error:      str,
        failed_sql: str,
        retries:    int,
        request_id: str = "",
    ) -> str | None:
        """
        Write failure to the training corpus.

        Returns the corpus entry_id string when USE_MCP_SERVERS=true so
        _failure_result() can store it on the QueryResult for the CLI
        :correct command to use.
        Returns None for the local path (entry_id not needed — CLI scans
        the failures/ directory directly).

        Routes to corpus_server.py MCP (port 5013) when USE_MCP_SERVERS=true,
        falling back to local file write if MCP is unreachable.
        """
        if settings.use_mcp_servers:
            # ── MCP path ───────────────────────────────────────────────────
            try:
                result = call_corpus_log_failure(
                    nl_query   = nl_query,
                    failed_sql = failed_sql,
                    error      = error,
                    retries    = retries,
                )
                entry_id = result.get("id")
                logger.info(
                    component = "pipeline",
                    event     = "failure_logged_via_mcp",
                    id        = entry_id,
                    error     = error[:100],
                )
                return entry_id
            except MCPCallError as exc:
                # MCP corpus server unreachable — fall back to local write
                logger.warning(
                    component = "pipeline",
                    event     = "corpus_mcp_fallback",
                    error     = str(exc),
                    note      = "Falling back to local file write.",
                )
                self._log_failure_local(nl_query, error, failed_sql,
                                        retries, request_id)
                return None

        # ── Direct local file path (default) ───────────────────────────────
        self._log_failure_local(nl_query, error, failed_sql, retries, request_id)
        return None

    def _log_failure_local(
        self,
        nl_query:   str,
        error:      str,
        failed_sql: str,
        retries:    int,
        request_id: str = "",
    ) -> None:
        """
        Write failure to local failures/ directory (atomic tmp + rename).
        Used when USE_MCP_SERVERS=false or when the corpus MCP server is
        unreachable (fallback from _log_failure MCP path).
        """
        failure_dir = Path(settings.failure_log_dir)
        failure_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)

        entry = {
            "timestamp":     now.isoformat(),
            "nl_query":      nl_query,
            "failed_sql":    failed_sql,
            "error":         error,
            "retries":       retries,
            "corrected_sql": "",
        }

        filename     = failure_dir / (
            f"{now.strftime('%Y%m%d_%H%M%S_%f')}_{request_id}.json"
        )
        tmp_filename = filename.with_suffix(".tmp")
        tmp_filename.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_filename, filename)

        logger.info(
            component = "pipeline",
            event     = "failure_logged",
            path      = str(filename),
            error     = error[:100],
        )