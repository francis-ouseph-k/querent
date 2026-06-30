"""
validation/logical_audit.py
────────────────────────────
Lightweight NL↔SQL alignment checks that run AFTER structural validation passes.

PIPELINE POSITION:
  Step 1 (Understanding) → Step 2 (Retrieval) → Step 3 (Prompt) → Step 4 (LLM)
  → Step 5 (Structural Validation — sql_validator.py)
  → Step 5.8 (Logical Audit — THIS MODULE)       ← between validation and execution
  → Step 6 (Execution)

WHY THIS EXISTS:
  Batch evaluation of 191 queries revealed ~55 queries that passed all 9 structural
  validation steps but produced WRONG results. These are semantically valid SQL that
  is logically incorrect — e.g., using AVG(page_number) instead of AVG(COUNT(*)),
  or using EXISTS when NOT EXISTS was needed.

  The structural validator cannot catch these because the SQL is syntactically and
  schema-wise correct. This module bridges that gap with NL↔SQL alignment checks.

CHECKS:
  - L1: Noun coverage — NL entities should be reflected in SQL tables/columns
  - L2: GROUP BY alignment — "per X" in NL should match GROUP BY columns
  - L3: Aggregation match — "average" in NL should produce AVG() in SQL
  - L4: Anti-join polarity — "none/without" in NL needs NOT EXISTS/IS NULL
  - L5: Tautological aggregation — COUNT(DISTINCT x) GROUP BY x always = 1

DESIGN DECISIONS:
  - Pure functions, no DB access — zero latency impact
  - Soft confidence penalty (not hard reject) — avoids false positive rejections
  - Each check adds 0.05–0.10 penalty; cumulative ≥0.15 sets passed=False
  - Warnings are logged but do not block execution — they reduce confidence
    which is surfaced to the user in the CLI
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence
from utils.heuristics import HEURISTICS


@dataclass
class AuditResult:
    """
    Result of the logical audit.

    Fields:
        warnings:           List of human-readable warning strings, each
                            prefixed with the check ID (e.g. "[L3] NL asks...").
                            Logged by runner.py but not shown to the user.
        confidence_penalty: Cumulative penalty subtracted from the LLM's
                            self-assessed confidence. This is the primary
                            mechanism — rather than rejecting the query,
                            we degrade confidence so the CLI shows lower
                            reliability to the user.
        requirement_coverage:
                            (NEW, L6/L7) Fraction in [0.0, 1.0] of NL
                            requirements that the SQL appears to satisfy.
                            Computed from the QuestionRequirements contract.
                            UNLIKE confidence_penalty, this is an absolute
                            signal — it does NOT depend on the LLM's
                            self-reported confidence.  A correct SQL gets
                            1.0; missing one constraint out of four drops
                            it to 0.75.  Surfaced to the CLI so users see
                            an objective coverage number alongside the
                            LLM's opinion.  None means "not measured"
                            (no requirements were extracted from the NL).
        coverage_misses:    (NEW) Human-readable list of which specific
                            requirements were not satisfied.  Empty when
                            requirement_coverage is 1.0 or None.
    """
    warnings: list[str] = field(default_factory=list)
    confidence_penalty: float = 0.0
    requirement_coverage: float | None = None
    coverage_misses: list[str] = field(default_factory=list)

    def add_warning(self, check_id: str, message: str, penalty: float = 0.05):
        """Append a warning and accumulate its confidence penalty."""
        self.warnings.append(f"[{check_id}] {message}")
        self.confidence_penalty += penalty


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_nl_nouns(nl: str) -> set[str]:
    """
    Extract key nouns from the NL query.
    Returns lowercased, de-duplicated set of domain-relevant words.

    Strategy: tokenize on word boundaries, remove stop words and SQL/query
    verbs ("show", "find", "count"), keep domain nouns like "script",
    "board", "evaluator" that map to database entities.
    """
    # Stop words include English function words PLUS query-action verbs
    # ("show", "find", "list") that appear in NL but have no SQL meaning.
    # We also exclude aggregation keywords ("count", "total") to avoid
    # false positives in L1 noun coverage — these are handled by L3.
    _STOP_WORDS = frozenset(HEURISTICS.get('stop_words', {}).get('logical_audit', []))

    # Tokenize: split on non-alpha/underscore, keep underscored domain terms
    words = set(re.findall(r'[a-z_]+', nl.lower()))
    return words - _STOP_WORDS


def _sql_lower(sql: str) -> str:
    """Return lowered SQL with normalized whitespace."""
    return ' '.join(sql.lower().split())


# ═══════════════════════════════════════════════════════════════════════════════
# Check L1: Noun Coverage
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Q60 — NL says "bundles but no scripts" but SQL had NOT EXISTS(bundle),
#          reversing the logic. L1 would flag "scripts" missing from SQL if
#          the SQL only referenced bundle but not answer_script.
#
# RISK LEVEL: Lowest penalty (0.05) because of false-positive risk from
#   synonyms and abbreviations. A query about "papers" might legitimately
#   use question_paper through a JOIN without mentioning "paper" directly.
#
# HOW IT WORKS:
#   1. Extract domain nouns from the NL question
#   2. For each noun, look up expected SQL identifiers in the map below
#   3. If none of the expected identifiers appear in the SQL, flag it

# Maps NL nouns → list of SQL identifiers (table names, column name fragments)
# that we expect to see in the SQL if the NL mentions this entity.
# Multiple entries per noun allow for flexible matching — e.g., "evaluator"
# could appear as faculty_cache (the person) or evaluation_attempt (the action).
#
# 2026-06-25 (E5 consolidation): Base mappings loaded from entity_synonyms
# in heuristics.yaml. Enriched with additional SQL identifier variants
# that the L1 noun coverage check needs for flexible matching.
# This eliminates drift between entity_synonyms and this dict.

# Additional SQL identifiers beyond the base table name from entity_synonyms.
# These are table/column fragments that are valid in SQL for each noun.
_NOUN_SQL_ENRICHMENTS: dict[str, list[str]] = {
    'scripts': ['script_id', 'script'],
    'evaluators': ['evaluator', 'evaluation_attempt', 'script_assignment'],
    'students': ['student_id', 'student'],
    'coordinators': ['board_coordinator'],
    'bundles': ['bundle'],
    'transitions': ['workflow_state_transition'],
    'configurations': ['configuration', 'config_key'],
    'policies': ['evaluation_policy'],
    'annotations': ['evaluation_annotation', 'annotation'],
    'results': ['result'],
    'marks': ['evaluation_marks', 'marks', 'total_marks', 'max_marks'],
    'attempts': ['evaluation_attempt', 'attempt'],
}

# Build the patterns dict from entity_synonyms + enrichments
_entity_synonyms = HEURISTICS.get('entity_synonyms', {})
_NOUN_TO_SQL_PATTERNS: dict[str, list[str]] = {}

# Populate from entity_synonyms (each maps noun → single table name)
for noun, table in _entity_synonyms.items():
    patterns = [table]
    # Add enrichments if defined
    if noun in _NOUN_SQL_ENRICHMENTS:
        patterns.extend(_NOUN_SQL_ENRICHMENTS[noun])
    _NOUN_TO_SQL_PATTERNS[noun] = patterns

# Add singular/plural variants and domain-specific extras
# that aren't in entity_synonyms but are needed for L1 matching.
_EXTRA_NOUN_PATTERNS: dict[str, list[str]] = {
    'script': ['answer_script', 'script_id', 'script'],
    'board': ['board'],
    'evaluator': ['evaluator', 'faculty_cache', 'evaluation_attempt', 'script_assignment'],
    'student': ['student_cache', 'student_id', 'student'],
    'faculty': ['faculty_cache', 'faculty_course_mapping'],
    'coordinator': ['board_coordinator'],
    'bundle': ['bundle'],
    'scanner': ['scanner_device', 'scan_history'],
    'scan': ['scan_history', 'scan_status', 'scan_metadata'],
    'rubric': ['answer_key_rubric', 'rubric'],
    'rubrics': ['answer_key_rubric', 'rubric'],
    'annotation': ['evaluation_annotation', 'annotation'],
    'revaluation': ['revaluation_request'],
    'honorarium': ['honorarium_summary', 'honorarium'],
    'moderation': ['moderation_rule', 'moderation_application'],
    'result': ['result'],
    'attempt': ['evaluation_attempt', 'attempt'],
    'policy': ['evaluation_policy'],
    'question': ['question'],
    'paper': ['question_paper'],
    'department': ['academic_unit', 'department'],
    'departments': ['academic_unit', 'department'],
    'course': ['academic_unit', 'course'],
    'courses': ['academic_unit', 'course'],
    'exam': ['exam_schedule_cache', 'exam'],
    'exams': ['exam_schedule_cache', 'exam'],
    'hold': ['script_hold', 'hold'],
    'holds': ['script_hold', 'hold'],
    'deadline': ['deadline_extension_request', 'deadline'],
    'extension': ['deadline_extension_request', 'revaluation_extension_request', 'extension'],
}
for noun, patterns in _EXTRA_NOUN_PATTERNS.items():
    if noun not in _NOUN_TO_SQL_PATTERNS:
        _NOUN_TO_SQL_PATTERNS[noun] = patterns


def _check_noun_coverage(nl: str, sql: str, result: AuditResult) -> None:
    """
    L1: Verify NL key nouns appear somewhere in the SQL.

    For each domain noun in the NL question, check if the SQL references
    at least one of the expected table/column identifiers. If a mapped
    noun is completely absent from the SQL, it may indicate the LLM
    answered a different question than what was asked.

    Penalty: 0.05 (lowest — highest false-positive risk).
    """
    nl_nouns = _extract_nl_nouns(nl)
    sql_low = _sql_lower(sql)

    # Check each NL noun against its expected SQL patterns
    missing_nouns: list[str] = []
    for noun in nl_nouns:
        patterns = _NOUN_TO_SQL_PATTERNS.get(noun)
        if patterns is None:
            continue  # Not a mapped noun — skip (e.g. "latest", "approved")
        # At least one pattern must appear in the SQL
        if not any(p in sql_low for p in patterns):
            missing_nouns.append(noun)

    # Cap at 5 nouns in the warning to keep it readable
    if missing_nouns:
        result.add_warning(
            "L1",
            f"NL mentions [{', '.join(missing_nouns[:5])}] but SQL doesn't reference "
            f"the expected tables/columns. This may indicate a semantic mismatch.",
            penalty=0.05,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Check L2: GROUP BY Alignment with "per/by" Nouns
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Q114 — "Faculty_course_mapping entries per academic year" but SQL
#          grouped by au.id (academic_unit PK) instead of fcm.academic_year.
#          The "per academic year" noun didn't appear in GROUP BY.
#
# RISK LEVEL: Medium penalty (0.08–0.10). Only fires for aggregation/comparison
#   intents to avoid false positives on lookup queries.

def _check_group_by_alignment(nl: str, sql: str, intent: str, result: AuditResult) -> None:
    """
    L2: If NL says 'per X' or 'by X', verify GROUP BY references X.

    Strategy:
      1. Only check aggregation/comparison intents (skip lookups)
      2. Extract noun(s) after "per", "by", or "for each" in the NL
      3. Convert to snake_case and check if it appears in GROUP BY
      4. Flag if GROUP BY exists but doesn't reference the expected noun

    Penalty: 0.10 if GROUP BY is missing entirely, 0.08 if GROUP BY exists
    but references the wrong column.
    """
    sql_low = _sql_lower(sql)

    # Only relevant for aggregation-type queries — "per" in a lookup query
    # like "find student by name" is not a GROUP BY signal
    if intent not in ('aggregation', 'comparison'):
        return

    # Extract the noun after "per", "by", or "for each"
    # Captures up to 2 words: "per academic year", "by board", "for each exam"
    per_by_matches = re.findall(
        r'\b(?:per|by|for each)\s+([a-z_]+(?:\s+[a-z_]+)?)',
        nl.lower()
    )

    if not per_by_matches:
        return

    # First check: is there a GROUP BY at all?
    # Use regex to handle variable whitespace (GROUP  BY, GROUP\nBY)
    if 'group by' not in sql_low and 'group  by' not in sql_low:
        has_group = bool(re.search(r'group\s+by', sql_low))
        if not has_group:
            result.add_warning(
                "L2",
                f"NL mentions 'per/by {per_by_matches[0]}' but SQL has no GROUP BY clause.",
                penalty=0.10,
            )
            return

    # Second check: does the GROUP BY reference the right column?
    for noun in per_by_matches:
        # Convert NL noun to likely SQL column name
        # "academic year" → "academic_year"; "board" → "board"
        noun_parts = noun.strip().split()
        noun_snake = '_'.join(noun_parts)

        # Extract the GROUP BY column list from the SQL
        group_by_match = re.search(r'group\s+by\s+(.+?)(?:\s+order|\s+having|\s+limit|$)', sql_low)
        if group_by_match:
            group_by_cols = group_by_match.group(1)
            # Check if any part of the noun appears in GROUP BY
            # e.g., "academic_year" in "GROUP BY fcm.academic_year" → match
            if not any(part in group_by_cols for part in [noun_snake] + noun_parts):
                result.add_warning(
                    "L2",
                    f"NL says 'per/by {noun}' but GROUP BY ({group_by_cols[:60]}) "
                    f"doesn't reference '{noun_snake}'. The grouping may be wrong.",
                    penalty=0.08,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Check L3: Aggregation-Question Match
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Q8  — "Average number of annotations per frozen attempt" but SQL returned
#          a list of counts per attempt with no outer AVG() wrapper.
#   Q10 — "Percentage of revaluation requests..." but no * 100 or FILTER.
#
# RISK LEVEL: High penalty (0.08–0.10) — these are clear semantic mismatches.

def _check_aggregation_match(nl: str, sql: str, result: AuditResult) -> None:
    """
    L3: If NL asks for 'average', verify SQL has AVG(). Same for percentage.

    Checks three aggregation patterns:
      - "average"/"mean" → SQL must contain AVG()
      - "percentage"/"percent" → SQL must have * 100 with FILTER or division
      - "ratio" → SQL must have a division operator

    Penalty: 0.10 for average (most common failure), 0.08 for percentage/ratio.
    """
    nl_low = nl.lower()
    sql_low = _sql_lower(sql)

    # Check 1: "average" or "mean" in NL → expect AVG() in SQL
    if re.search(r'\b(average|avg|mean)\b', nl_low):
        if 'avg(' not in sql_low:
            result.add_warning(
                "L3",
                "NL asks for 'average/mean' but SQL has no AVG() function. "
                "The query may return a list instead of an aggregated average.",
                penalty=0.10,
            )

    # Check 2: "percentage" in NL → expect * 100 with FILTER or division
    # Our prompt rule says: COUNT(*) FILTER(WHERE cond) * 100.0 / NULLIF(COUNT(*), 0)
    if re.search(r'\b(percentage|percent|pct)\b', nl_low):
        has_pct = ('100' in sql_low and ('filter' in sql_low or '/' in sql_low))
        if not has_pct:
            result.add_warning(
                "L3",
                "NL asks for 'percentage' but SQL doesn't compute one "
                "(expected * 100 with FILTER or division).",
                penalty=0.08,
            )

    # Check 3: "ratio" in NL → expect division
    if re.search(r'\bratio\b', nl_low):
        if '/' not in sql_low and 'nullif' not in sql_low:
            result.add_warning(
                "L3",
                "NL asks for 'ratio' but SQL has no division operation.",
                penalty=0.08,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Check L4: Anti-Join Polarity
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Q60 — "Exam schedules with bundles but no scripts" but SQL filtered
#          NOT EXISTS(bundle) — the exact opposite of what was asked.
#          The SQL should have had NOT EXISTS(answer_script) instead.
#
# RISK LEVEL: High penalty (0.10). Anti-join polarity reversal produces
#   completely wrong results (includes instead of excludes).

def _check_anti_join_polarity(nl: str, sql: str, result: AuditResult) -> None:
    """
    L4: If NL says 'none'/'without'/'missing', SQL should have NOT EXISTS or IS NULL.

    Strategy:
      1. Detect negation words/phrases in the NL
      2. Check if the SQL has an anti-join pattern
      3. Flag if negation is present but no anti-join pattern found

    Note: This does NOT check polarity correctness (which table is negated).
    That would require deeper NL parsing. It only checks presence/absence
    of the anti-join pattern.

    Penalty: 0.10 (high — polarity errors produce completely wrong results).
    """
    nl_low = nl.lower()
    sql_low = _sql_lower(sql)

    # Negation patterns in the NL that imply an anti-join is needed
    negation_patterns = [
        r'\bno\s+\w+',       # "no scripts", "no coordinator"
        r'\bnone\b',          # "none of the..."
        r'\bwithout\b',       # "scripts without annotations"
        r'\bmissing\b',       # "missing rubrics"
        r'\bnever\b',         # "evaluators who never submitted"
        r'\bnot\s+(?:assigned|registered|created|started|approved)\b',  # "not assigned"
    ]

    has_negation = any(re.search(p, nl_low) for p in negation_patterns)
    if not has_negation:
        return

    # Valid anti-join SQL patterns: NOT EXISTS, IS NULL (LEFT JOIN), NOT IN
    has_anti_join = (
        'not exists' in sql_low or
        'is null' in sql_low or
        'not in' in sql_low or
        'left join' in sql_low  # LEFT JOIN + WHERE IS NULL pattern
    )

    if not has_anti_join:
        result.add_warning(
            "L4",
            "NL contains negation (none/without/missing/never) but SQL has no "
            "anti-join pattern (NOT EXISTS, IS NULL, LEFT JOIN). "
            "The query may return wrong results by including rather than excluding.",
            penalty=0.10,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Check L5: Tautological Aggregation
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Q143 — "Questions belonging to exactly one attempt rule group" produced
#          COUNT(DISTINCT q.id) GROUP BY q.id — every group has exactly 1 row,
#          so COUNT is always 1. The outer query then gets N rows of 1.
#   Q15  — COUNT(DISTINCT question_id) GROUP BY question_id — same pattern.
#
# WHY THIS HAPPENS:
#   The LLM confuses "count X per X" with "count X per Y". When X and Y are
#   the same column, the aggregation becomes a no-op.
#
# RISK LEVEL: High penalty (0.10). Tautological aggregation is always wrong.

def _check_tautological_aggregation(sql: str, result: AuditResult) -> None:
    """
    L5: Detect patterns that produce meaningless results.

    Currently detects:
      - COUNT(DISTINCT x) ... GROUP BY x → always returns 1 per group

    Future candidates (not yet implemented):
      - SUM(x) ... GROUP BY x (single-table) → always = x itself
      - AVG(page_number) → averaging ordinals, not counts
      - self-referencing subtraction (col_a - col_a)

    Penalty: 0.10 (high — always a semantic error).
    """
    sql_low = _sql_lower(sql)

    # Extract COUNT(DISTINCT col) expressions from the SQL
    # Matches both "count(distinct q.id)" and "count(distinct id)"
    count_distinct = re.findall(r'count\s*\(\s*distinct\s+(\w+\.\w+|\w+)\s*\)', sql_low)

    # Extract GROUP BY column list
    group_by_match = re.search(r'group\s+by\s+(.+?)(?:\s+order|\s+having|\s+limit|$)', sql_low)

    if count_distinct and group_by_match:
        group_cols = group_by_match.group(1)
        for col in count_distinct:
            # Strip table alias: "q.id" → "id" for matching against GROUP BY
            col_name = col.split('.')[-1] if '.' in col else col
            # If the COUNT(DISTINCT) column is also in GROUP BY → tautology
            if col_name in group_cols:
                result.add_warning(
                    "L5",
                    f"COUNT(DISTINCT {col}) with GROUP BY {col_name} always produces 1. "
                    f"This is likely a tautological aggregation.",
                    penalty=0.10,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Check L6: Constraint Coverage  (NEW — structural, NL-driven)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Fourteen batch-run-20260625 queries where the LLM kept the easy
#   filters and silently dropped the hard ones:
#     Q56 — NL says "expired KEKs", SQL has no expired filter
#     Q76 — NL says "active scanner devices", SQL has no active filter
#     Q92 — NL says "CUSTODIAN role", SQL is fine, but missing
#           entity_type='answer_script' filter
#     Q147 — NL says "students marked absent", SQL counts ALL students
#     Q166 — NL says "WITHHELD eligibility", SQL has no WITHHELD literal
#     Q189 — NL says "CSE department", SQL uses unit_type='COURSE'
#
# WHY IT IS NOT A BANDAID:
#   The earlier L1 noun-coverage check matched bag-of-words.  This
#   check operates on STRUCTURED REQUIREMENTS — each constraint carries
#   the SQL token it expects to find AND the kind of position that
#   token should occupy.  Adding a new constraint kind benefits this
#   check automatically; no new SQL-anti-pattern detector is needed.

def _check_constraint_coverage(
    requirements,                      # validation.nl_requirements.QuestionRequirements
    sql: str,
    result: AuditResult,
) -> tuple[int, int]:
    """
    L6: For each filter constraint extracted from the NL, verify at
    least one of its expected tokens appears in the SQL.

    Returns:
        (satisfied_count, total_count) so the caller can roll this
        into a coverage score.

    Penalty: 0.10 per missed ENUM/STATUS/ROLE constraint (high-
    confidence misses), 0.05 per missed TIME_RANGE/NUMERIC (these
    have more SQL-shape variability and warrant softer flagging).
    """
    if not requirements.filter_constraints:
        return (0, 0)

    sql_low = _sql_lower(sql)
    satisfied = 0
    total = 0

    for c in requirements.filter_constraints:
        total += 1
        if not c.tokens:
            # No specific token expected — count as satisfied to avoid
            # false flag on under-specified constraints.
            satisfied += 1
            continue

        # A constraint is satisfied if ANY of its tokens appears in
        # the SQL.  This is intentionally permissive — the audit job
        # is to catch silent DROPS, not to enforce one particular
        # phrasing.
        if any(t.lower() in sql_low for t in c.tokens):
            satisfied += 1
            continue

        # Constraint was DROPPED.  Penalty depends on kind.
        from validation.semantic.nl_requirements import ConstraintKind   # local import to avoid cycle
        if c.kind in (ConstraintKind.ENUM, ConstraintKind.STATUS,
                       ConstraintKind.ROLE, ConstraintKind.BOOLEAN,
                       ConstraintKind.SCOPE):
            penalty = 0.10
        else:
            # TIME_RANGE, NUMERIC, TEXT_LIKE — softer flagging because
            # the SQL form is variable (CURRENT_DATE - INTERVAL, BETWEEN,
            # DATE_TRUNC, EXTRACT, ...).
            penalty = 0.05

        result.add_warning(
            "L6",
            f"NL constraint '{c.raw}' (kind={c.kind.value}) is missing "
            f"from the SQL — expected one of {c.tokens} to appear.",
            penalty=penalty,
        )
        result.coverage_misses.append(
            f"constraint:{c.kind.value}={c.raw}"
        )

    return (satisfied, total)


# ═══════════════════════════════════════════════════════════════════════════════
# Check L7: Output Coverage   (NEW — structural, NL-driven)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Three queries where the NL explicitly named output columns the
#   SQL never projected:
#     Q32 — "show student name, course code, ..." — SQL projects URN
#           only, no student name, no course code
#     Q35 — "show their faculty name, department, ..." — SQL omits
#           department
#     Q39 — "broken down by evaluator role" — SQL group-by mangled
#
# IMPLEMENTATION NOTE:
#   We do NOT need an SQL parser for this — checking whether the
#   noun appears in the SQL's projected text is enough.  False
#   positives are reduced by only firing when the NL uses an
#   explicit projection header (show/list/display/including).

def _check_output_coverage(
    requirements,
    sql: str,
    result: AuditResult,
) -> tuple[int, int]:
    """
    L7: When the NL has an explicit output list, verify each named
    item appears in the SQL's SELECT region.

    Returns (satisfied, total).

    Only fires when the NL contains a REAL projection list — i.e.
    multiple comma-separated items.  Single-item "outputs" like
    "list all question papers" are too easily confused with a bare
    table reference; the LLM may legitimately project all columns
    via SELECT * or selected columns, and we'd false-flag.
    """
    if not requirements.output_columns:
        return (0, 0)

    # Filter out items that begin with "all"/"every"/"any" — these are
    # NL quantifiers over a table, not specific output columns.
    # Also drop bare single-word items (too noisy).
    filtered = []
    for col in requirements.output_columns:
        first = col.split()[0] if col.split() else ""
        if first in ("all", "every", "any", "the", "those"):
            continue
        # Skip if the item is the SQL table name verbatim (e.g.
        # "question papers" when the question is just listing the
        # question_paper table).
        if len(col.split()) == 1:
            continue
        filtered.append(col)

    # Require at least 2 items in the list — otherwise this isn't a
    # multi-column projection but a single-noun reference.
    if len(filtered) < 2:
        return (0, 0)

    sql_low = _sql_lower(sql)
    m = re.search(r"\bselect\b\s+(.*?)\s+\bfrom\b", sql_low, re.DOTALL)
    select_region = m.group(1) if m else sql_low

    satisfied = 0
    for col in filtered:
        words = col.split()
        head = words[-1] if words else ""
        underscored = "_".join(words)
        compact = "".join(words)

        if (head and head in select_region) or \
           (underscored and underscored in select_region) or \
           (compact and compact in select_region):
            satisfied += 1
        else:
            result.add_warning(
                "L7",
                f"NL requests output column '{col}' but the SQL's "
                f"SELECT list doesn't appear to project it.",
                penalty=0.08,
            )
            result.coverage_misses.append(f"output:{col}")

    return (satisfied, len(filtered))


# ═══════════════════════════════════════════════════════════════════════════════
# Check L8: LEFT JOIN with Filter in ON  (NEW — structural)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Five queries where a filter on the right-hand side of a LEFT JOIN
#   was placed in the ON clause, silently making the filter inert
#   because LEFT JOIN keeps left-side rows regardless:
#     Q2  — LEFT JOIN question_section ... AND qs.name = 'Section A'
#           AND qs.qp_id = (SELECT ...)
#     Q76 — LEFT JOIN academic_unit ... AND au.code = 'MAIN'
#     Q91 — LEFT JOIN app_user ... AND au.display_name ILIKE '%COE Office%'
#     Q104 — LEFT JOIN exam_schedule_cache ... AND esc.academic_year = '2025-2026'
#     Q130 — LEFT JOIN exam_schedule_cache ... AND esc.academic_year = '2025-12'
#
# WHY THE EXISTING L4 / SEMANTIC CHECK 19 DOES NOT CATCH IT:
#   Check 19 catches the MIRROR pattern: LEFT JOIN followed by
#   WHERE on the optional-side column, which nullifies the LEFT JOIN.
#   That pattern is "filter applied AFTER the join — too aggressive".
#   THIS pattern is "filter applied IN the join — too permissive".
#   The two patterns are opposites; the existing check is blind to
#   this one.
#
# DETECTION:
#   Pattern-match `LEFT JOIN <table> <alias> ON <conditions>`, parse
#   the conditions into equality joins vs. filter predicates, and
#   flag when a non-join filter exists in ON without a corresponding
#   `<alias>.<col> IS NOT NULL` in the WHERE clause.

# Capture the alias and the ON clause body.  The body extends until
# the next JOIN/WHERE/GROUP/ORDER/LIMIT/UNION keyword or end of SQL.
_LEFT_JOIN_RE = re.compile(
    r"\bleft\s+(?:outer\s+)?join\s+"
    r"(?:\w+\.)?(\w+)"                            # table name
    r"(?:\s+(?:as\s+)?(\w+))?"                    # optional alias
    r"\s+on\s+"
    r"(.+?)"                                       # ON body (lazy)
    r"(?=\b(?:left|right|inner|cross|full|join|where|group|order|"
    r"having|limit|union|except|intersect)\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _check_left_join_on_filter(sql: str, result: AuditResult) -> None:
    """
    L8: Detect LEFT JOIN whose ON clause contains a value filter
    on the right-hand-side table that is not paired with a
    `IS NOT NULL` in the WHERE clause.
    """
    sql_low = _sql_lower(sql)

    # Find the WHERE region (or empty string).  We test for
    # `IS NOT NULL` checks inside it.
    where_m = re.search(
        r"\bwhere\b(.*?)(?=\bgroup\s+by\b|\border\s+by\b|"
        r"\bhaving\b|\blimit\b|\bunion\b|$)",
        sql_low, re.DOTALL,
    )
    where_body = where_m.group(1) if where_m else ""

    for jm in _LEFT_JOIN_RE.finditer(sql_low):
        table = jm.group(1)
        alias = jm.group(2) or table
        on_body = jm.group(3) or ""

        # Split the ON body on AND/OR.  Each clause is either an
        # equality JOIN (`a.col = b.col`) or a value filter
        # (`a.col = 'literal'`, `a.col >= ...`).
        # We're permissive on whitespace; sqlglot would be more
        # robust but we keep this regex-only to stay light.
        clauses = re.split(r"\s+\b(?:and|or)\b\s+", on_body)
        suspicious_filters: list[str] = []

        for cl in clauses:
            cl = cl.strip().rstrip(")").lstrip("(")
            if not cl:
                continue
            # Is this an equality between two columns (true join)?
            # `<ident>.<ident> = <ident>.<ident>` or just `<ident> = <ident>`.
            if re.match(
                r"^\(?\s*(?:\w+\.)?\w+\s*=\s*(?:\w+\.)?\w+\s*\)?$",
                cl,
            ):
                continue
            # Is this a NULL check?
            if "is null" in cl or "is not null" in cl:
                continue
            # Does this filter mention the right-hand-side alias and
            # also a literal (number, quoted string, current_date,
            # interval expression)?
            mentions_alias = (
                f" {alias}." in f" {cl}" or
                cl.startswith(f"{alias}.")
            )
            mentions_literal = (
                "'" in cl or '"' in cl or
                bool(re.search(r"\b\d", cl)) or
                "current_date" in cl or
                "current_timestamp" in cl or
                "now(" in cl or
                "interval" in cl
            )
            if mentions_alias and mentions_literal:
                suspicious_filters.append(cl)

        if not suspicious_filters:
            continue

        # If the WHERE clause has `<alias>.<some-col> IS NOT NULL`
        # then the LEFT JOIN is being deliberately used as a semi-
        # join (e.g. anti-pattern is intentional).  Don't flag.
        not_null_in_where = bool(re.search(
            rf"\b{re.escape(alias)}\.\w+\s+is\s+not\s+null\b",
            where_body,
        ))
        if not_null_in_where:
            continue

        # ANTI-JOIN PATTERN: if WHERE has `<alias>.<col> IS NULL`
        # the LEFT JOIN is the standard SQL anti-join idiom
        # ("rows in left side with NO match in right side").  The
        # ON-clause filter is then SCOPING the anti-join — entirely
        # correct usage.  Catches Q33 false positive.
        is_null_in_where = bool(re.search(
            rf"\b{re.escape(alias)}\.\w+\s+is\s+null\b",
            where_body,
        ))
        if is_null_in_where:
            continue

        # AGGREGATION PATTERN: if the SQL aggregates over the joined
        # alias (`COUNT(<alias>.id)`, `SUM(<alias>.amount)`,
        # `MAX(<alias>.created_at)`, etc.) then the LEFT JOIN is
        # being used to count-or-aggregate matches per left-side
        # row INCLUDING rows with zero matches.  The ON-clause
        # filter then scopes which matches qualify.  Standard SQL
        # idiom.  Catches Q11 false positive.
        agg_over_alias = bool(re.search(
            rf"\b(?:count|sum|avg|min|max|array_agg|string_agg|"
            rf"bool_or|bool_and|jsonb_agg|json_agg)\s*\("
            rf"\s*(?:distinct\s+)?{re.escape(alias)}\.",
            sql_low,
        ))
        # Also catch `COUNT(<alias>.id) FILTER (WHERE ...)` and
        # `COUNT(CASE WHEN <alias>.col ...)` patterns.
        if not agg_over_alias:
            agg_over_alias = bool(re.search(
                rf"\bcount\s*\([^)]*case\s+when\s+{re.escape(alias)}\.",
                sql_low,
            ))
        if agg_over_alias:
            continue

        # Genuine miss.  Report it.
        result.add_warning(
            "L8",
            f"LEFT JOIN {table} {alias!r} has filter predicate(s) "
            f"{suspicious_filters} inside the ON clause without a "
            f"matching `{alias}.<col> IS NOT NULL` in WHERE. "
            f"The filter is silently dropped because LEFT JOIN keeps "
            f"left-side rows regardless.  Move the filter into the "
            f"WHERE clause or use INNER JOIN.",
            penalty=0.10,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Check L9: Entity-Type Hint Mismatch  (NEW — structural)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IT CATCHES:
#   Q189 — NL says "CSE department", SQL filters unit_type='COURSE'
#   Any future query where the NL pairs an identifier with an entity
#   type ("CS101 program", "Main campus") and the SQL filters by the
#   identifier with the WRONG unit_type or relationship side.

def _check_entity_type_hints(
    requirements,
    sql: str,
    result: AuditResult,
) -> None:
    """
    L9: When the NL pairs an identifier with an entity type
    ("CSE department"), verify the SQL uses the matching unit_type.

    This is a STRICT check — we only flag when the SQL clearly uses
    a DIFFERENT unit_type literal for the same identifier.
    """
    if not requirements.entity_type_hints:
        return

    sql_low = _sql_lower(sql)
    for ident, expected_type in requirements.entity_type_hints:
        # If the identifier appears in the SQL and a unit_type / type
        # filter also appears, verify they agree.
        if ident.lower() not in sql_low:
            continue

        # Extract every unit_type literal the SQL uses.
        used_types = set(re.findall(
            r"unit_type\s*=\s*'([A-Z_]+)'", sql_low.upper(),
        ))
        # Also catch the lower-case variant just in case
        used_types |= set(t.upper() for t in re.findall(
            r"unit_type\s*=\s*'([a-z_]+)'", sql_low,
        ))

        if used_types and expected_type not in used_types:
            wrong = sorted(used_types - {expected_type})
            result.add_warning(
                "L9",
                f"NL refers to '{ident} {expected_type.lower()}' but the "
                f"SQL filters by unit_type IN {wrong}.  Likely the wrong "
                f"academic-unit category.",
                penalty=0.12,
            )
            result.coverage_misses.append(
                f"entity_type:{ident}={expected_type}->{wrong}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_logical_audit(
    nl_query: str,
    sql: str,
    intent: str = "",
    tables_used: Sequence[str] | None = None,
) -> AuditResult:
    """
    Run all logical audit checks on a validated SQL query.

    Args:
        nl_query: The user's natural language question.
        sql: The validated SQL query.
        intent: The query intent (e.g., 'aggregation', 'lookup').
        tables_used: Tables referenced by the SQL.

    Returns:
        AuditResult with any warnings and confidence penalty.

    Check sequence:
        L1–L5  existing SQL-pattern checks (noun coverage, GROUP BY,
               aggregation match, anti-join polarity, tautology).
        L6     NL→SQL constraint coverage (NEW).
        L7     NL→SQL output coverage (NEW).
        L8     LEFT JOIN with filter in ON (NEW).
        L9     Entity-type hint mismatch (NEW).

    The NEW checks operate on a typed QuestionRequirements value
    parsed from the NL.  Together with the coverage score they
    provide an objective (non-LLM) confidence signal.
    """
    result = AuditResult()

    if not nl_query or not sql:
        return result

    # Existing SQL-pattern checks (unchanged) ─────────────────────────
    _check_noun_coverage(nl_query, sql, result)           # L1
    _check_group_by_alignment(nl_query, sql, intent, result)  # L2
    _check_aggregation_match(nl_query, sql, result)       # L3
    _check_anti_join_polarity(nl_query, sql, result)      # L4
    _check_tautological_aggregation(sql, result)          # L5

    # NEW: parse the NL once, then run requirement-driven checks ─────
    # Import is local to keep validation.nl_requirements optional —
    # if the module is somehow missing, the existing audit still runs.
    try:
        from validation.semantic.nl_requirements import parse_question
        requirements = parse_question(nl_query)
    except Exception:
        # Defensive: never let the new pipeline break the old one.
        return result

    c_satisfied, c_total = _check_constraint_coverage(   # L6
        requirements, sql, result,
    )
    o_satisfied, o_total = _check_output_coverage(       # L7
        requirements, sql, result,
    )
    _check_left_join_on_filter(sql, result)              # L8
    _check_entity_type_hints(requirements, sql, result)  # L9

    # Compute non-LLM coverage score.  None when no requirements
    # were extracted — avoids dragging confidence down on queries
    # the parser has nothing to say about.
    total_reqs = c_total + o_total
    if total_reqs > 0:
        result.requirement_coverage = round(
            (c_satisfied + o_satisfied) / total_reqs, 3,
        )

    return result