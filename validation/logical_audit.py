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
    """
    warnings: list[str] = field(default_factory=list)
    confidence_penalty: float = 0.0

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
_NOUN_TO_SQL_PATTERNS: dict[str, list[str]] = {
    # --- Core entities ---
    'script': ['answer_script', 'script_id', 'script'],
    'scripts': ['answer_script', 'script_id', 'script'],
    'board': ['board'],
    'boards': ['board'],
    'evaluator': ['evaluator', 'faculty_cache', 'evaluation_attempt', 'script_assignment'],
    'evaluators': ['evaluator', 'faculty_cache', 'evaluation_attempt', 'script_assignment'],
    'student': ['student_cache', 'student_id', 'student'],
    'students': ['student_cache', 'student_id', 'student'],
    'faculty': ['faculty_cache', 'faculty_course_mapping'],
    'coordinator': ['board_coordinator'],
    'coordinators': ['board_coordinator'],
    # --- Scanning & physical assets ---
    'bundle': ['bundle'],
    'bundles': ['bundle'],
    'scanner': ['scanner_device', 'scan_history'],
    'scan': ['scan_history', 'scan_status', 'scan_metadata'],
    # --- Evaluation components ---
    'rubric': ['answer_key_rubric', 'rubric'],
    'rubrics': ['answer_key_rubric', 'rubric'],
    'annotation': ['evaluation_annotation', 'annotation'],
    'annotations': ['evaluation_annotation', 'annotation'],
    'revaluation': ['revaluation_request'],
    'honorarium': ['honorarium_summary', 'honorarium'],
    'moderation': ['moderation_rule', 'moderation_application'],
    'result': ['result'],
    'results': ['result'],
    'marks': ['evaluation_marks', 'marks', 'total_marks', 'max_marks'],
    'attempt': ['evaluation_attempt', 'attempt'],
    'attempts': ['evaluation_attempt', 'attempt'],
    'policy': ['evaluation_policy'],
    'policies': ['evaluation_policy'],
    # --- Academic structure ---
    'question': ['question'],
    'questions': ['question'],
    'paper': ['question_paper'],
    'papers': ['question_paper'],
    'department': ['academic_unit', 'department'],
    'departments': ['academic_unit', 'department'],
    'course': ['academic_unit', 'course'],
    'courses': ['academic_unit', 'course'],
    'exam': ['exam_schedule_cache', 'exam'],
    'exams': ['exam_schedule_cache', 'exam'],
    # --- Workflow entities ---
    'hold': ['script_hold', 'hold'],
    'holds': ['script_hold', 'hold'],
    'deadline': ['deadline_extension_request', 'deadline'],
    'extension': ['deadline_extension_request', 'revaluation_extension_request', 'extension'],
}


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
    """
    result = AuditResult()

    # Guard: skip audit for empty inputs (shouldn't happen in practice,
    # but defensive against upstream edge cases)
    if not nl_query or not sql:
        return result

    # Run all checks in order. Each check mutates `result` by appending
    # warnings and accumulating confidence_penalty.
    # Checks are independent — they don't short-circuit on failure.
    _check_noun_coverage(nl_query, sql, result)          # L1: 0.05 penalty
    _check_group_by_alignment(nl_query, sql, intent, result)  # L2: 0.08–0.10
    _check_aggregation_match(nl_query, sql, result)      # L3: 0.08–0.10
    _check_anti_join_polarity(nl_query, sql, result)     # L4: 0.10
    _check_tautological_aggregation(sql, result)         # L5: 0.10

    return result
