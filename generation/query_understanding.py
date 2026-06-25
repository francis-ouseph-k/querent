"""
generation/query_understanding.py
───────────────────────────────────
Query understanding layer — four sequential components:

  1. Query Rewriter    — expand abbreviations, normalise synonyms
  2. Intent Classifier — rule-based + keyword heuristics (no second LLM call)
  3. Entity Extractor  — identify tables, status codes, domain terms in the query
  4. Ambiguity Detector — data-driven: catches ambiguous terms and incomplete
                          queries (missing required parameters)

After extraction, a RapidFuzz course-code resolver normalises freeform course
identifiers (e.g. "MBA01", "mba 01", "M B A 01") to the canonical code form
so the downstream SQL can JOIN academic_unit on the correct string.

Design: no LLM call here. Everything runs in <2ms on CPU.

─────────────────────────────────────────────────────────────────────────────
FIXES IN THIS VERSION
─────────────────────────────────────────────────────────────────────────────

Issue 1 (corpus corruption fix) — removed "eligible", "eligibility", "barred",
  "not eligible", "not_eligible" from _DISAMBIGUATORS["failed"] and "eligible",
  "eligibility" from _DISAMBIGUATORS["passed"].
  Problem: "list eligible students who failed" contained "eligible", which was
  a resolver for "failed", so disambiguation was skipped. The LLM would then
  hallucinate a 40% threshold and the user's :correct entry would poison the
  Phase 2 corpus with a (query, wrong-threshold SQL) pair.
  Fix: disambiguation for "failed"/"passed" is now only skipped when the user
  explicitly provides a threshold value (percent, %, cutoff, threshold, below,
  above, etc.). Eligibility-related words no longer bypass the threshold prompt.

Issue 2 (corpus corruption fix) — removed the "Students with
  eligibility_status = NOT_ELIGIBLE (barred — did not sit the exam)" option
  from _AMBIGUOUS_TERMS["failed"].
  Problem: NOT_ELIGIBLE means the student was barred from sitting, not that
  they sat and failed. Presenting it as an interpretation of "failed" is
  semantically wrong; a user selecting it would receive students who never
  wrote the exam.
  Fix: option removed from "failed" menu entirely. Eligibility queries should
  be phrased as "not eligible" or "barred", which can be added as their own
  _AMBIGUOUS_TERMS entry if needed.
  The "passed" list's ELIGIBLE option is retained — ELIGIBLE genuinely means
  "sat the exam", which is a valid (if imprecise) interpretation of "passed".

Issue 3 (false-positive entity extraction) — \\b word boundaries now applied
  in _extract_entities() for term-to-table matching.
  Problem: re.search(re.escape(term), lower) was a raw substring search.
  "mark" matched inside "remark", "script" inside "postscript", "status"
  inside "statusquo". False table matches seed wrong mandatory chunks into
  the RAG prompt, increasing hallucination pressure.
  Fix: re.search(rf"\b{re.escape(term)}\b", lower) — same boundary logic
  already used in _rewrite() and _detect_ambiguity(). Multi-word terms like
  "evaluation attempt" work correctly because the space is a non-word char.

Issue 4 — added "percentage" and "percentile" to _DISAMBIGUATORS["failed"]
  and _DISAMBIGUATORS["passed"]. Users say "percentage" more than "percent".

Issue 5 (prompt marker pollution) — REFINED_MARKER and VALUE_MARKER are now
  stripped from parsed.normalised before it reaches the prompt builder and
  the retrieval layer. The markers are implementation detail; the LLM must
  never see "— specifically:" or "— value:" in the [QUERY] block.
  The clean NL question and the clarification note are injected as separate
  fields so the prompt builder can format them correctly.
  NOTE: runner.py calls _strip_refinement_markers() before building the prompt.
  ParsedQuery gains two new optional fields: clean_query and clarification_note.

Minor — "top" regex updated from `\\btop\\s+\\d+\\b` to `\\btop\\s*\\d+\\b` to handle
  "top10" without a space.

Minor — domain_terms deduplicated via set before return to avoid duplicates
  from overlapping glossary term matches.

RapidFuzz course-code normaliser — new method _resolve_course_code().
  Extracts probable course code tokens (e.g. "MBA01") from the query,
  fuzzy-matches them against known academic_unit codes loaded at init time,
  and returns a (raw_token, canonical_code, score) tuple for the highest
  confidence match. Runner uses this to inject a concrete JOIN condition
  rather than relying on the LLM to guess the BIGINT PK.
  Threshold: 85 (token_sort_ratio). Below threshold → no match → caller
  leaves the code as-is and relies on the LLM + schema chunks.
  The known codes list is loaded from data/academic_unit_codes.json, which
  ingest.py generates by querying SELECT DISTINCT code FROM academic_unit.

C1  — _rewrite() \\b word boundaries (prevents "eval" → "evaluationuation").
M5  — _classify_intent() deterministic tie-breaking via list order.

REVIEW FIX (NEW-M2) — _extract_label_filters()'s _STOPWORDS set was defined
  as a local set literal inside the innermost loop body (the `while True:`
  pos-scanning loop, itself inside the per-trigger `for` loop), so it was
  rebuilt from scratch on every iteration — once per occurrence of every
  trigger phrase found in the query. Moved to module level as a frozenset
  (_LABEL_FILTER_STOPWORDS) built once at import time. Renamed to avoid
  colliding with any local "_STOPWORDS" naming elsewhere and to make the
  module-level scope obvious at the call site.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

from models.schema import ParsedQuery, QueryIntent
from utils.logging_config import get_logger
from utils.heuristics import HEURISTICS

logger = get_logger(__name__)

# ── Optional RapidFuzz import ─────────────────────────────────────────────────
# rapidfuzz is a soft dependency — if absent the course-code normaliser is
# disabled and the system falls back to exact matching only. All other
# functionality is unaffected. Install: pip install rapidfuzz
try:
    from rapidfuzz import fuzz, process as rfuzz_process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning(
        component="query_understanding",
        event="rapidfuzz_not_installed",
        note="Course-code fuzzy matching disabled. pip install rapidfuzz to enable.",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Abbreviation / synonym rewrite map
# Merged with glossary aliases at runtime.
# ─────────────────────────────────────────────────────────────────────────────
_BASE_REWRITES: dict[str, str] = {
    "eval":         "evaluation",
    "evals":        "evaluations",
    "reval":        "revaluation",
    "revals":       "revaluations",
    "3rd":          "third",
    "3rd eval":     "third evaluation",
    "primary eval": "primary evaluation",
    "bd":           "board",
    "qp":           "question paper",
    # "mrks" → "marks": expand abbreviation without prejudging
    # table (evaluation_marks) vs column (marks_awarded).
    "mrks":         "marks",
    "script key":   "data encryption key",
    "dek":          "data_encryption_key",
    "kek":          "key_encryption_key",
}

# ─────────────────────────────────────────────────────────────────────────────
# Intent classification keyword patterns
#
# List order encodes tie-breaking priority (M5 fix):
#   AGGREGATION > TIME_SERIES > WORKFLOW_STATE > COMPARISON > LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
_INTENT_PATTERNS: list[tuple[QueryIntent, list[str]]] = [
    (QueryIntent.AGGREGATION,    [
        "count", "total", "sum", "average", "avg", "how many",
        "percentage", "distribution", "breakdown", "grouped", "group by",
        "top", "highest", "lowest", "most", "least", "ranking",
    ]),
    (QueryIntent.TIME_SERIES,    [
        "trend", "over time", "per month", "per week", "per day",
        "timeline", "history", "historical", "since", "between dates",
        "last month", "last year", "academic year",
    ]),
    (QueryIntent.WORKFLOW_STATE, [
        "pending", "status", "frozen", "submitted", "assigned",
        "in progress", "blocked", "malpractice", "hold", "held",
        "third evaluation", "reval", "revaluation", "primary", "review",
        "lifecycle", "not assigned", "not scanned",
    ]),
    (QueryIntent.COMPARISON,     [
        "compare", "difference", "versus", "vs", "higher than",
        "lower than", "more than", "less than", "differ",
    ]),
    (QueryIntent.LOOKUP,         [
        "show", "list", "find", "get", "fetch", "display",
        "what is", "who is", "which", "details",
    ]),
]

# ─────────────────────────────────────────────────────────────────────────────
# Table-name keyword map — term → canonical table name
# Fallback when data/query_understanding.json is absent (pre-ingestion).
# ─────────────────────────────────────────────────────────────────────────────
from utils.heuristics import HEURISTICS
_TABLE_KEYWORDS = HEURISTICS.get('table_keywords', {})

# Status code tokens recognised verbatim. Fallback for pre-ingestion.
_STATUS_CODES: set[str] = {
    "W", "P", "N", "THIRD", "PRIMARY", "REVIEW", "REVAL",
    "ASSIGNED", "IN_PROGRESS", "FROZEN", "SUBMITTED", "CLOSED",
    "NOT_ASSIGNED", "SCANNED", "NOT_SCANNED", "RESCAN_NEEDED",
    "NONE", "ABSENT", "BLOCKED", "MALPRACTICE", "NOT_ELIGIBLE",
    "ACTIVE", "EXPIRED", "REVOKED", "RETIRED", "CLOUD", "WORKSTATION",
    "PENDING", "APPROVED", "REJECTED", "EXPORTED",
    "ATTEMPTED", "ELIGIBLE", "ADMITTED", "BARRED",
}

# ─────────────────────────────────────────────────────────────────────────────
# AMBIGUITY DETECTION ROUTER
# ─────────────────────────────────────────────────────────────────────────────

from .disambiguation.spec import DisambiguationSpec
from .disambiguation.router import DisambiguationRouter
from .disambiguation.enricher import SchemaEnricher

def _top_matcher(query: str) -> bool:
    if "top-level" in query:
        return False
    if re.search(r"\btop\s*\d+\b", query):
        return False
    return bool(re.search(r"\btop\b", query))

_DISAMBIGUATION_SPECS = [
    DisambiguationSpec(
        term="failed",
        options=[
            "INCOMPLETE — pass threshold not stored in schema. Specify percentage: e.g. 'students who scored below 40%'",
            "Students scored below 40% of total marks (result.final_marks < question_paper.total_marks * 0.4)",
            "Students scored below 50% of total marks (result.final_marks < question_paper.total_marks * 0.5)",
        ],
        resolvers=[
            "percent", "percentage", "percentile", "%", "cutoff", "threshold",
            "below", "less than", "under",
            "erp", "push", "upload", "sync", "purge", "job", "run", "external system", "external", "system",
        ]
    ),
    DisambiguationSpec(
        term="passed",
        options=[
            "INCOMPLETE — pass threshold not stored in schema. Specify percentage: e.g. 'students who scored above 40%'",
            "Students scored 40% or above of total marks (result.final_marks >= question_paper.total_marks * 0.4)",
            "Students scored 50% or above of total marks (result.final_marks >= question_paper.total_marks * 0.5)",
            "Students with eligibility_status = ELIGIBLE (sat the exam — does NOT imply they passed on marks)",
        ],
        resolvers=[
            "percent", "percentage", "percentile", "%", "cutoff", "threshold",
            "above", "more than", "over", "at least", "greater than",
        ]
    ),
    DisambiguationSpec(
        term="recent",
        options=[
            "INCOMPLETE — time range needed. Specify: e.g. 'in the last 7 days', 'this month', 'AY 2025-2026'",
        ],
        resolvers=[
            "day", "days", "week", "weeks", "month", "months",
            "year", "years", "since", "last", "between",
            "today", "yesterday", "this",
            "2024", "2025", "2026",
        ]
    ),
    DisambiguationSpec(
        term="top",
        options=[
            "INCOMPLETE — count needed. Specify: e.g. 'top 10 evaluators by scripts completed'",
        ],
        resolvers=[],
        custom_matcher=_top_matcher
    ),
    DisambiguationSpec(
        term="pending",
        options=[
            "Scripts pending evaluation assignment (answer_script.evaluation_status = NOT_ASSIGNED)",
            "Scripts pending moderation review (answer_script.evaluation_status = SUBMITTED)",
            "Results pending ERP export (result.erp_push_status = PENDING)",
            "Honorarium payments pending approval (honorarium_summary.approval_status = PENDING)",
            "Revaluation requests pending decision (revaluation_request.approval_status = PENDING)",
        ],
        resolvers=[
            "evaluation", "moderation", "erp", "export",
            "honorarium", "payment", "revaluation", "approval",
            "assignment", "result",
            "primary", "review", "third", "reval",
            "not_assigned", "in_progress", "frozen", "submitted",
        ]
    ),
    DisambiguationSpec(
        term="status",
        options=[
            "Evaluation workflow status (answer_script.evaluation_status)",
            "Script scanning status (answer_script.scan_status)",
            "Student lifecycle status (answer_script.lifecycle_status)",
            "Board lifecycle status (board.status)",
            "ERP push status for results (result.erp_push_status)",
            "Honorarium approval status (honorarium_summary.approval_status)",
        ],
        resolvers=[
            "evaluation", "scan", "scanning", "lifecycle",
            "block", "blocking", "erp", "push",
            "approval", "board", "publication",
            "answer key", "answer_key", "bundle", "eligibility", "purge", "job", "external system", "external", "system",
        ]
    ),
    DisambiguationSpec(
        term="marks",
        options=[
            "Per-question marks during evaluation (evaluation_marks.marks_awarded) — use for question-level breakdown",
            "Total marks for one evaluation attempt (evaluation_attempt.total_marks) — use for evaluator-level view",
            "Final published marks (result.final_marks) — use for results, pass/fail, and ERP export",
        ],
        resolvers=[
            "total", "final", "published", "question", "per question",
            "awarded", "honorarium", "attempt",
        ]
    ),
    DisambiguationSpec(
        term="frozen",
        options=[
            "Answer scripts fully frozen (answer_script.evaluation_status = FROZEN — all sections done)",
            "Individual evaluation attempts frozen (evaluation_attempt.status = FROZEN — one evaluator done)",
        ],
        resolvers=[
            "script", "scripts", "attempt", "attempts",
            "evaluation attempt", "answer script",
        ]
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Marker strings — shared with cli/interface.py and pipeline/runner.py.
# Keep in sync: any rename here must be reflected in both callers.
# ─────────────────────────────────────────────────────────────────────────────
INCOMPLETE_PREFIX = "INCOMPLETE"     # option prefix → CLI shows free-text prompt
REFINED_MARKER    = "— specifically:" # appended to query after menu pick
VALUE_MARKER      = "— value:"        # appended to query after free-text input

# Regex to extract the clean question and optional clarification note
# from a refined query string.
_REFINED_SPLIT_RE  = re.compile(r"\s*" + re.escape(REFINED_MARKER) + r"\s*", re.IGNORECASE)
_VALUE_SPLIT_RE    = re.compile(r"\s*" + re.escape(VALUE_MARKER)   + r"\s*", re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# RapidFuzz course-code match result
# ─────────────────────────────────────────────────────────────────────────────

class CourseCodeMatch(NamedTuple):
    """Result of fuzzy course-code resolution."""
    raw_token:  str    # token as typed by user (e.g. "mba 01")
    canonical:  str    # matched code from academic_unit (e.g. "MBA01")
    score:      float  # token_sort_ratio score 0–100


# Minimum fuzzy score to accept a course-code match.
# 85 tolerates "MBA 01" / "mba01" / "Mba01" while rejecting unrelated strings.
_COURSE_CODE_MATCH_THRESHOLD = 85

# Regex to identify probable course-code tokens in a query:
# 2–4 uppercase letters followed by optional space then 1–4 digits.
# Matches: "MBA01", "MBA 01", "CS101", "B.COM1" (partial), "BTECH12"
_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4})\s*(\d{1,4})\b")

# ── Label-filter detection ────────────────────────────────────────────────────
# Alphanumeric identifiers (like MBA101, BUS2023, CS4) are text codes, not
# integer PKs. When the user writes "course id MBA101" or "board MBA101" the
# LLM must use .code / .name (VARCHAR) not .id (BIGINT).
#
# Maps trigger phrases → (table, text_column, hint_for_prompt).
# Longest-match-first to prevent "course" matching before "course id".
_LABEL_FILTER_TRIGGERS: list[tuple[str, str, str, str]] = [
    # Multi-word triggers first (longest match priority)
    ("course id",      "academic_unit", "code",        "academic_unit.code (VARCHAR) — NOT academic_unit.id (BIGINT)"),
    ("course code",    "academic_unit", "code",        "academic_unit.code (VARCHAR) — NOT academic_unit.id (BIGINT)"),
    ("course name",    "academic_unit", "name",        "academic_unit.name (VARCHAR)"),
    ("department id",  "academic_unit", "code",        "academic_unit.code (VARCHAR) — NOT academic_unit.id (BIGINT)"),
    ("student id",     "student_cache", "student_erp_id",  "student_cache.student_erp_id (VARCHAR) — NOT student_cache.id (BIGINT)"),
    ("student code",   "student_cache", "student_erp_id",  "student_cache.student_erp_id (VARCHAR)"),
    ("employee id",    "faculty_cache", "employee_erp_id", "faculty_cache.employee_erp_id (VARCHAR) — NOT faculty_cache.id (BIGINT)"),
    ("faculty id",     "faculty_cache", "employee_erp_id", "faculty_cache.employee_erp_id (VARCHAR) — NOT faculty_cache.id (BIGINT)"),
    ("exam id",        "exam_schedule_cache", "exam_erp_id", "exam_schedule_cache.exam_erp_id (VARCHAR) — NOT exam_schedule_cache.id (BIGINT)"),
    ("board code",     "board",         "id",          "board.id (BIGINT) — board is identified by integer id"),
    # Single-word triggers (lower priority)
    ("course",         "academic_unit", "code",        "academic_unit.code (VARCHAR) — NOT academic_unit.id (BIGINT)"),
    ("department",     "academic_unit", "code",        "academic_unit.code (VARCHAR) — NOT academic_unit.id (BIGINT)"),
]

# Regex to find alphanumeric label tokens: letters mixed with digits,
# OR pure uppercase letter sequences, that look like codes not integers.
# Matches: MBA101, CS4, BTECH2023, BUS — excludes pure integers like 5, 101.
_LABEL_TOKEN_RE = re.compile(
    r"\b([A-Z]{2,}(?:\s*\d{1,4})?|[A-Z]{1,4}\d{2,4})\b"
)

# REVIEW FIX (NEW-M2): moved out of _extract_label_filters()'s innermost loop
# body, where it was previously a local `set` literal rebuilt on every
# iteration of the `while True:` pos-scanning loop (itself nested inside the
# per-trigger `for` loop) — i.e. rebuilt once per occurrence of every trigger
# phrase found in the query. Defined once here as a frozenset at import time.
# frozenset (not set) since this is a fixed, never-mutated lookup table —
# using an immutable type makes that contract explicit and is marginally
# faster for membership tests.
_LABEL_FILTER_STOPWORDS: frozenset[str] = frozenset(HEURISTICS.get('stop_words', {}).get('query_understanding', []))


def _strip_refinement_markers(query: str) -> tuple[str, str | None]:
    """
    Split a refined query back into (clean_question, clarification_note).

    Called by runner.py before building the LLM prompt so the model never
    sees "— specifically:" or "— value:" in the [QUERY] block.

    Examples:
      "show failed students — value: below 40%"
        → ("show failed students", "below 40%")

      "show pending scripts — specifically: Scripts pending ERP export ..."
        → ("show pending scripts", "Scripts pending ERP export ...")

      "show frozen scripts in board 5"
        → ("show frozen scripts in board 5", None)
    """
    for pattern in (_REFINED_SPLIT_RE, _VALUE_SPLIT_RE):
        parts = pattern.split(query, maxsplit=1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return query.strip(), None


class QueryUnderstanding:
    """
    Processes a raw NL query into a ParsedQuery structure.

    Pipeline:
      raw_query
        → _rewrite()              expand abbreviations, normalise synonyms
        → _classify_intent()      AGGREGATION / TIME_SERIES / WORKFLOW / COMPARISON / LOOKUP
        → _extract_entities()     table names, status codes, domain terms
        → _resolve_course_codes() RapidFuzz: normalise "MBA01", "mba 01" → canonical code
        → _detect_ambiguity()     flag ambiguous terms or missing parameters

    Data sources (loaded once at __init__):
      data/glossary.json              — aliases + term→table mappings
      data/query_understanding.json   — DDL-derived keywords and status codes
      data/academic_unit_codes.json   — known course codes for fuzzy matching
    """

    def __init__(
        self,
        glossary_path:            str = "data/glossary.json",
        query_understanding_path: str = "data/query_understanding.json",
        academic_unit_codes_path: str = "data/academic_unit_codes.json",
    ) -> None:
        # ── Initialise maps from static fallbacks ─────────────────────────
        self._rewrites:      dict[str, str]       = dict(_BASE_REWRITES)
        self._term_to_table: dict[str, list[str]] = {k: list(v) for k, v in _TABLE_KEYWORDS.items()}
        self._status_codes:  set[str]       = set(_STATUS_CODES)

        # ── Overlay DDL-generated data (always up-to-date) ─────────────────
        self._load_query_understanding_data(query_understanding_path)

        # ── Overlay glossary aliases (highest specificity) ─────────────────
        self._load_glossary(glossary_path)

        # ── Load known course codes for RapidFuzz resolver ─────────────────
        # List of canonical code strings, e.g. ["MBA01", "BTECH11", "CS101"]
        self._course_codes: list[str] = self._load_course_codes(academic_unit_codes_path)

        # ── Pre-compile rewrite patterns with \b boundaries (C1 fix) ───────
        # Sorted longest-first: "3rd eval" matches before "eval".
        self.router = DisambiguationRouter(_DISAMBIGUATION_SPECS)
        self.schema_enricher = SchemaEnricher("data/fk_graph.json")

        self._rewrite_patterns: list[tuple[re.Pattern, str]] = [
            (
                re.compile(rf"\b{re.escape(abbrev)}\b", re.IGNORECASE),
                expansion,
            )
            for abbrev, expansion in sorted(
                self._rewrites.items(), key=lambda x: len(x[0]), reverse=True
            )
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_query_understanding_data(self, path: str) -> None:
        """
        Load DDL-derived keywords and status codes.
        Absent before first ingestion run — static fallbacks remain active.
        """
        p = Path(path)
        if not p.exists():
            logger.info(
                component="query_understanding",
                event="query_understanding_data_not_found",
                path=path,
                note="Using static fallback dicts. Run ingest.py to generate.",
            )
            return
        try:
            data     = json.loads(p.read_text(encoding="utf-8"))
            keywords = data.get("table_keywords", {})
            codes    = data.get("status_codes", [])
            # values may be str (old ingest format) or list[str] (new).
            # Merge — never overwrite; append new tables to existing list.
            for term, val in keywords.items():
                tables   = [val] if isinstance(val, str) else list(val)
                existing = self._term_to_table.setdefault(term, [])
                for t in tables:
                    if t not in existing:
                        existing.append(t)
            self._status_codes.update(codes)
            logger.info(
                component    = "query_understanding",
                event        = "query_understanding_data_loaded",
                keywords     = len(keywords),
                status_codes = len(codes),
            )
        except Exception as exc:
            logger.warning(
                component="query_understanding",
                event="query_understanding_data_load_failed",
                path=path, error=str(exc),
            )

    def _load_glossary(self, path: str) -> None:
        """
        Merge glossary aliases into rewrite map and glossary terms into
        entity map. Aliases become rewrite rules (alias → canonical term).
        related_tables[0] becomes the entity table for the term.
        """
        p = Path(path)
        if not p.exists():
            return
        try:
            entries = json.loads(p.read_text(encoding="utf-8"))
            for entry in entries:
                term    = entry.get("term", "").lower()
                tables  = entry.get("related_tables", [])
                aliases = entry.get("aliases", [])
                if term and tables:
                    existing = self._term_to_table.setdefault(term, [])
                    for t in tables:
                        if t not in existing:
                            existing.append(t)
                for alias in aliases:
                    self._rewrites[alias.lower()] = term
        except Exception as exc:
            logger.warning(
                component="query_understanding",
                event="glossary_load_failed", error=str(exc),
            )

    def _load_course_codes(self, path: str) -> list[str]:
        """
        Load canonical academic_unit codes for fuzzy matching.

        File format (generated by ingest.py from SELECT DISTINCT code FROM
        academic_unit WHERE unit_type = 'COURSE'):
          ["MBA01", "BTECH11", "CS101", "BCOM01", ...]

        Returns empty list when file absent — fuzzy matching disabled silently.
        The system continues to work; users must spell codes correctly.
        """
        p = Path(path)
        if not p.exists():
            logger.info(
                component="query_understanding",
                event="course_codes_not_found",
                path=path,
                note="Course-code fuzzy matching disabled. Run ingest.py.",
            )
            return []
        try:
            codes = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(codes, list):
                raise ValueError("Expected a JSON array of strings.")
            logger.info(
                component="query_understanding",
                event="course_codes_loaded",
                count=len(codes),
            )
            return [str(c).strip().upper() for c in codes if c]
        except Exception as exc:
            logger.warning(
                component="query_understanding",
                event="course_codes_load_failed",
                path=path, error=str(exc),
            )
            return []

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def process(self, raw_query: str) -> ParsedQuery:
        """
        Full pipeline: rewrite → intent → entities → course codes → ambiguity.

        If is_ambiguous=True, runner.py returns early before any LLM call
        and the CLI presents clarifications to the user.

        ParsedQuery.clean_query contains the query with markers stripped —
        runner.py uses this for retrieval and prompt construction so the
        LLM never sees "— specifically:" or "— value:" in the [QUERY] block.
        ParsedQuery.clarification_note contains the stripped suffix (if any)
        for the prompt builder to inject as a separate [CLARIFICATION] block.
        """
        normalised                     = self._rewrite(raw_query)
        intent                         = self._classify_intent(normalised)
        entities, status_codes, domain = self._extract_entities(normalised)
        course_match                   = self._resolve_course_codes(normalised)
        label_filters                  = self._extract_label_filters(raw_query)
        is_ambiguous, clarifications, resolved_choices = self._detect_ambiguity(normalised, entities)

        # Issue 5 fix: strip markers before exposing normalised to prompt layer.
        # If the SLM automatically resolved any ambiguity, append it to the query
        # so the downstream LLM sees the chosen context!
        if not is_ambiguous and resolved_choices:
            for choice in resolved_choices:
                normalised += f" {REFINED_MARKER} {choice}"
                
        # runner.py uses clean_query for retrieval + prompt, not normalised.
        clean_query, clarification_note = _strip_refinement_markers(normalised)

        parsed = ParsedQuery(
            original           = raw_query,
            normalised         = normalised,       # full string incl. markers (for logging)
            clean_query        = clean_query,       # marker-stripped (for LLM prompt)
            clarification_note = clarification_note, # extracted clarification (for prompt)
            intent             = intent,
            entities           = entities,
            status_codes       = status_codes,
            domain_terms       = domain,
            course_code_match  = course_match,
            label_filters      = label_filters,
            is_ambiguous       = is_ambiguous,
            clarifications     = clarifications,
        )

        logger.info(
            component          = "query_understanding",
            event              = "parsed",
            intent             = intent.value,
            entities           = entities,
            status_codes       = status_codes,
            is_ambiguous       = is_ambiguous,
            course_code_match  = (
                f"{course_match.raw_token}→{course_match.canonical}"
                f" ({course_match.score:.0f})" if course_match else None
            ),
        )
        return parsed

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline steps
    # ─────────────────────────────────────────────────────────────────────

    def _rewrite(self, query: str) -> str:
        """
        Expand abbreviations and normalise synonyms (C1 fix).

        Pre-compiled \\b-anchored patterns prevent short abbreviations from
        corrupting longer words ("eval" must not match inside "evaluation").
        Patterns are sorted longest-first at compile time so a longer phrase
        like "3rd eval" matches before the shorter "eval".
        """
        result = query.strip()
        for pattern, expansion in self._rewrite_patterns:
            result = pattern.sub(expansion, result)
        return result

    def _classify_intent(self, query: str) -> QueryIntent:
        """
        Score-based intent classification (M5 fix).

        Each keyword match scores +1. Highest total wins.
        Ties broken by list position in _INTENT_PATTERNS
        (AGGREGATION > TIME_SERIES > WORKFLOW_STATE > COMPARISON > LOOKUP).
        Returns LOOKUP when no keyword matches.
        """
        lower  = query.lower()
        scores = {intent: 0 for intent, _ in _INTENT_PATTERNS}

        for intent, keywords in _INTENT_PATTERNS:
            for kw in keywords:
                if kw in lower:
                    scores[intent] += 1

        best = max(scores.values())
        if best == 0:
            return QueryIntent.LOOKUP

        for intent, _ in _INTENT_PATTERNS:
            if scores[intent] == best:
                return intent

        return QueryIntent.LOOKUP

    def _extract_entities(
        self, query: str
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Extract three signal types:
          entities     — canonical table names (seeds mandatory RAG chunks)
          status_codes — recognised enum values (e.g. FROZEN, NOT_ASSIGNED)
          domain_terms — glossary terms (feeds GLOSSARY chunk retrieval)

        Issue 3 fix: term matching now uses \\b word boundaries to prevent
        substring false positives ("mark" inside "remark", "status" inside
        "statusquo", "script" inside "postscript").

        Longest-match-first + span-overlap check (M2 fix) prevents shorter
        terms from shadowing longer multi-word terms.

        Two-pass extraction:
          Pass 1 — exact keyword match (as before).
          Pass 2 — fuzzy fallback via rapidfuzz token_sort_ratio (threshold 82)
                   on unmatched query words/bigrams. Fills remaining slots only.

        domain_terms deduplicated via set (minor fix).
        """
        lower = query.lower()

        # ── Pass 1: Exact keyword match ───────────────────────────────────
        # _term_to_table values are list[str] — one term can map to multiple
        # tables (e.g. "marks" → ["evaluation_marks", "evaluation_attempt"]).
        # Longest-match-first prevents shorter terms shadowing multi-word terms.
        MAX_ENTITY_TABLES = 5

        entities:   list[str]             = []
        seen_spans: list[tuple[int, int]] = []

        for term, tables in sorted(
            self._term_to_table.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if len(entities) >= MAX_ENTITY_TABLES:
                break
            match = re.search(rf"\b{re.escape(term)}\b", lower)
            if not match:
                continue
            s, e = match.start(), match.end()
            if any(s < ex_e and ex_s < e for ex_s, ex_e in seen_spans):
                continue
            seen_spans.append((s, e))
            for table in tables:
                if table not in entities and len(entities) < MAX_ENTITY_TABLES:
                    entities.append(table)

        # ── Pass 2: Fuzzy fallback ────────────────────────────────────────
        # Runs only when Pass 1 left open slots AND rapidfuzz is available.
        # Matches 1-gram and 2-gram tokens from the query against keyword list.
        # token_sort_ratio handles word-order variation + minor typos.
        # Threshold 82: avoids noise on short words, catches typos/synonyms.
        # Min ngram length 4: skip stop words ("the", "for", "in", "of").
        _FUZZY_THRESHOLD = 82

        if _RAPIDFUZZ_AVAILABLE and len(entities) < MAX_ENTITY_TABLES:
            words  = re.findall(r"[a-z]+", lower)
            ngrams = words + [
                f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)
            ]
            keyword_list = list(self._term_to_table.keys())

            for ngram in ngrams:
                if len(entities) >= MAX_ENTITY_TABLES:
                    break
                if len(ngram) < 4:
                    continue
                # Skip ngram if it overlaps an already exact-matched span
                ng_match = re.search(re.escape(ngram), lower)
                if ng_match:
                    ns, ne = ng_match.start(), ng_match.end()
                    if any(ns < ex_e and ex_s < ne for ex_s, ex_e in seen_spans):
                        continue
                result = rfuzz_process.extractOne(
                    ngram,
                    keyword_list,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=_FUZZY_THRESHOLD,
                )
                if result is None:
                    continue
                matched_term, score, _ = result
                for table in self._term_to_table[matched_term]:
                    if table not in entities and len(entities) < MAX_ENTITY_TABLES:
                        entities.append(table)
                        logger.debug(
                            component="query_understanding",
                            event="fuzzy_entity_match",
                            ngram=ngram,
                            matched_term=matched_term,
                            score=score,
                            table=table,
                        )

        # ── Status codes ──────────────────────────────────────────────────
        # Match uppercase tokens (e.g. FROZEN) and single-quoted values
        status_codes: list[str] = []
        words = re.findall(r"[A-Z_]{2,}", query) + re.findall(r"'([^']+)'", query)
        for word in words:
            upper = word.upper()
            if upper in self._status_codes and upper not in status_codes:
                status_codes.append(upper)

        # ── Domain terms ──────────────────────────────────────────────────
        # Deduplication via set (minor fix — prevents "mark"+"marks" both appearing)
        domain_set:   set[str]  = set()
        domain_terms: list[str] = []
        for term in self._term_to_table:
            if term in lower and term not in domain_set:
                domain_set.add(term)
                domain_terms.append(term)

        return entities, status_codes, domain_terms

    def _extract_label_filters(self, query: str) -> list[dict]:
        """
        Detect alphanumeric label tokens (e.g. MBA101, CS4) paired with
        entity trigger phrases (e.g. "course id", "course", "student id").

        Returns a list of dicts:
          {
            "raw":     "MBA101",          # value as typed
            "table":   "academic_unit",   # target table
            "column":  "code",            # correct text column to filter on
            "hint":    "academic_unit.code (VARCHAR) — NOT academic_unit.id (BIGINT)",
            "context": "course id MBA101" # surrounding text for prompt context
          }

        Design:
          - Longest trigger match first (prevents "course" shadowing "course id").
          - Label tokens must be alphanumeric (not pure integers — those are valid
            PK literals and should not be flagged).
          - A token can match at most one trigger (first match wins).
          - Best-effort: returns [] on any parse failure.

        REVIEW FIX (NEW-M2): the stopword set previously lived inside this
        method's innermost loop body as a local `set` literal, rebuilt on
        every iteration. It is now the module-level frozenset
        _LABEL_FILTER_STOPWORDS, built once at import time — see its
        definition above for the full rationale.
        """
        upper = query.upper()
        results: list[dict] = []
        matched_spans: list[tuple[int, int]] = []

        # Sort triggers longest-first so "course id" beats "course"
        for trigger, table, column, hint in sorted(
            _LABEL_FILTER_TRIGGERS, key=lambda x: len(x[0]), reverse=True
        ):
            trigger_upper = trigger.upper()
            pos = 0
            while True:
                idx = upper.find(trigger_upper, pos)
                if idx == -1:
                    break
                pos = idx + 1

                # Look for a label token within 30 chars after the trigger
                search_start = idx + len(trigger_upper)
                search_end   = min(search_start + 30, len(upper))
                window       = upper[search_start:search_end]

                token_match = _LABEL_TOKEN_RE.search(window)
                if not token_match:
                    continue

                token     = token_match.group(1).strip()
                tok_start = search_start + token_match.start()
                tok_end   = search_start + token_match.end()

                # Skip pure integers — those are valid PK literals
                if token.isdigit():
                    continue

                # Skip tokens that are part of the trigger phrase itself
                # (e.g. "CODE" after "course code", "ID" after "course id")
                trigger_words = {w.upper() for w in trigger.split()}
                if token.upper() in trigger_words:
                    continue

                # Skip common English/domain stopwords that aren't identifiers.
                # REVIEW FIX (NEW-M2): now references the module-level
                # frozenset instead of constructing a new set here on every
                # iteration of this while loop.
                if token.upper() in _LABEL_FILTER_STOPWORDS:
                    continue

                # Skip if this token span already matched a longer trigger
                if any(s <= tok_start < e for s, e in matched_spans):
                    continue

                matched_spans.append((tok_start, tok_end))
                context = query[idx: min(tok_end, len(query))].strip()

                results.append({
                    "raw":     token,
                    "table":   table,
                    "column":  column,
                    "hint":    hint,
                    "context": context,
                })

        return results

    def _resolve_course_codes(self, query: str) -> CourseCodeMatch | None:
        """
        Fuzzy-match probable course-code tokens in the query against the
        known academic_unit codes loaded at init time.

        Uses RapidFuzz token_sort_ratio to handle:
          "MBA01" "MBA 01" "mba01" "Mba01" "M B A 01" → "MBA01"
          "btech 11" → "BTECH11"

        Why token_sort_ratio (not simple_ratio):
          token_sort_ratio sorts tokens alphabetically before comparing, so
          "MBA 01" and "01 MBA" both match "MBA01" correctly. It also
          normalises whitespace, making it robust to spacing variations.

        Returns None when:
          - rapidfuzz is not installed
          - no course-code pattern found in query
          - best match score < _COURSE_CODE_MATCH_THRESHOLD (85)
          - course codes list is empty (pre-ingestion)

        Return value is CourseCodeMatch(raw_token, canonical, score).
        runner.py uses this to:
          1. Log the normalisation for observability
          2. Inject the canonical code into the prompt [CLARIFICATION] block
             so the LLM uses the right code in the JOIN condition
        """
        if not _RAPIDFUZZ_AVAILABLE or not self._course_codes:
            return None

        # Find probable course-code tokens in the query
        # Pattern: 2-4 uppercase letters + optional space + 1-4 digits
        # Operate on the original query (not lowercased) to preserve case
        matches = _COURSE_CODE_RE.findall(query.upper())
        if not matches:
            return None

        best_match: CourseCodeMatch | None = None

        for letter_part, digit_part in matches:
            # Reconstruct both spaced and unspaced forms to give RapidFuzz
            # the best chance of matching
            candidate_forms = [
                f"{letter_part}{digit_part}",     # "MBA01"
                f"{letter_part} {digit_part}",    # "MBA 01"
            ]

            for candidate in candidate_forms:
                result = rfuzz_process.extractOne(
                    candidate,
                    self._course_codes,
                    scorer=fuzz.token_sort_ratio,
                )
                if result is None:
                    continue
                matched_code, score, _idx = result

                if score >= _COURSE_CODE_MATCH_THRESHOLD:
                    if best_match is None or score > best_match.score:
                        best_match = CourseCodeMatch(
                            raw_token = candidate,
                            canonical = matched_code,
                            score     = float(score),
                        )

        if best_match:
            logger.info(
                component  = "query_understanding",
                event      = "course_code_resolved",
                raw        = best_match.raw_token,
                canonical  = best_match.canonical,
                score      = best_match.score,
            )

        return best_match

    def _detect_ambiguity(
        self, query: str, entities: list[str]
    ) -> tuple[bool, list[str], list[str]]:
        """
        Data-driven ambiguity detection.

        Delegates to the Intent Router to check all registered DisambiguationSpecs,
        including Semantic SLM matching via MiniLM.
        """
        # Skip if query was already refined by the CLI disambiguation loop
        if REFINED_MARKER in query or VALUE_MARKER in query:
            return False, [], []

        return self.router.detect_ambiguity(
            query, 
            entities, 
            enrich_option=self.schema_enricher.enrich
        )
