"""
validation/semantic/semantic_checks.py
──────────────────────────────
Semantic heuristic checks and hardcoded-literal detection for SQL validation.

Contains Step 7 (12 semantic checks) and Step 8 (hardcoded literal IDs).
These are pure analysis — no DB or LLM calls — and form the fastest-changing
part of the validator.  New checks are added with each batch evaluation audit.

Separated from sql_validator.py so that new checks can be added without
touching the core structural validation pipeline.
"""

from __future__ import annotations

import re

import sqlglot
import sqlglot.expressions as exp

from models.schema import ValidationResult
from utils.logging_config import get_logger
from utils.heuristics import HEURISTICS
from ..core.base import BaseValidationStep
from ..core.context import ValidationContext

logger = get_logger(__name__)


def _predicate_has_literal_counterpart(col: "exp.Column") -> bool:
    """
    True when `col` participates in a comparison whose OTHER operand is a
    literal (col = 'X', col IN ('A','B'), col > 5, col LIKE '%x%'). False when
    the counterpart is another column (a join key / correlated reference), which
    does NOT nullify an outer join.
    """
    parent = col.parent
    if parent is None:
        return False
    # IN (...) list of literals
    if isinstance(parent, exp.In):
        exprs = parent.args.get("expressions") or []
        return any(isinstance(e, exp.Literal) for e in exprs) and not any(
            isinstance(e, exp.Column) for e in exprs
        )
    # Binary comparisons expose .left / .right
    left = getattr(parent, "left", None)
    right = getattr(parent, "right", None)
    if left is None and right is None:
        return False
    sibling = right if left is col else left
    if sibling is None:
        return False
    if isinstance(sibling, exp.Column):
        return False
    # Literal directly, or an expression whose leaves are literals (no columns)
    if isinstance(sibling, exp.Literal):
        return True
    if hasattr(sibling, "find_all"):
        has_lit = any(True for _ in sibling.find_all(exp.Literal))
        has_col = any(True for _ in sibling.find_all(exp.Column))
        return has_lit and not has_col
    return False


# ── Advisory (non-fatal) semantic checks ─────────────────────────────────────
# These NL↔SQL heuristics are synonym/encoding-blind: they compare surface
# keywords in the question against surface tokens in the SQL, so they reject
# CORRECT queries whenever the model uses a synonym or a structural encoding
# rather than the literal keyword. Run-3 evidence:
#   * semantic_unprompted_enum_filter  — Q56: "already expired" correctly encoded
#     as status = 'EXPIRED', but the check only looks for the word "status".
#   * semantic_missing_scope_filter    — Q148: "global scope" correctly encoded as
#     course_id IS NULL AND board_id IS NULL, but the check demands the word "global".
#   * semantic_unprompted_filter / semantic_noun_missing — same failure mode.
# Demoted to advisory: they still log (useful signal for corpus review) but do NOT
# fail the query. To restore any as fatal, remove its event name from this set.
#
# Run-4 evidence (batch run 20260812, Mistral small, 191 questions). Four more
# checks in this family rejected correct SQL for the same reason — they compare
# NL surface keywords against SQL surface tokens:
#
#   * semantic_wrong_entity_count   — Q53: "the number of pages recorded in scan
#     metadata" vs "the number of script_page rows" is a COMPARISON of two
#     counts. The check read "How many pages" and demanded COUNT(answer_script);
#     the query counted script_page rows on purpose.
#   * semantic_per_by_missing_group — Q153: "average records held per purge job".
#     purge_job_log holds exactly one row per job, so AVG(records_held) with no
#     GROUP BY is the correct reading. "per X" named the metric, not a grouping.
#   * semantic_avg_ordinal          — Q157: "average display order value" asks
#     literally for AVG(display_order). The ordinal-column rule assumed the user
#     must have meant an average count.
#   * semantic_per_entity_inner_join — Q20: fired 18 times across the run. It is
#     a real modelling concern, but it cannot distinguish "per X, include zeroes"
#     from "per X, among those that have any", and the retry it forces is the
#     step that converted Q20's correct LEFT JOIN query into an INNER JOIN one.
#
# All four still log, so the signal remains available for corpus review; none
# fails the query. Rule of thumb established by this run: a check derived from
# the DDL or the AST may hard-fail; a check that compares question words to SQL
# words is advisory.
#
# Run-6 (batch run 20260814_102207, Mistral small, 191 questions). ONE check
# leaves this set: semantic_unprompted_enum_filter, promoted back to fatal.
#
#   The Run-3 justification for demoting it was Q56 — "already expired"
#   correctly encoded as status = 'EXPIRED', flagged because the check only
#   looked for the word "status". That justification is now STALE: the check
#   was subsequently rewritten to test the VALUE, not the column name, and
#   'expired' is a substring of "already expired", so Q56 is justified and
#   silent under the current code. The comment outlived the defect.
#
#   Meanwhile the check fired 59 times in this run and was right about the
#   ones that mattered: Q39 ("broken down by evaluator role" -> WHERE
#   role_in_board = 'EVALUATOR', collapsing a group-by into a filter),
#   Q138 ("distribution across entity types" -> silent is_critical = TRUE),
#   Q42 ("not yet frozen" -> status = 'ASSIGNED', dropping IN_PROGRESS).
#   Each shipped as Success with a wrong population.
#
#   To keep the false-positive rate at the level that justified the original
#   demotion, _enum_value_justified() below adds morphological tolerance
#   (SCANNED/scanning, APPROVED/approve) on top of the existing surface-form
#   match. A value the question gestures at in ANY of those forms is accepted.
#
# semantic_per_entity_inner_join deliberately STAYS advisory: it cannot
# distinguish "per X, include zeroes" from "per X, among those that have any",
# and Run-4 showed the retry it forces makes correct queries worse.
_ADVISORY_SEMANTIC_EVENTS: set[str] = {
    "semantic_noun_missing",
    "semantic_unprompted_filter",
    "semantic_missing_scope_filter",
    "semantic_wrong_entity_count",
    "semantic_per_by_missing_group",
    "semantic_avg_ordinal",
    "semantic_per_entity_inner_join",
    # Run-5 evidence (batch run 20260813_144312, Mistral small, 191 questions).
    # semantic_antijoin_mismatch — Q31: "in SCANNING status for more than 24
    # hours without reaching COMPLETE". A bundle's status is a single column;
    # "without reaching COMPLETE" is correctly encoded as status = 'SCANNING',
    # not as an anti-join. The check read the word "without" in the question and
    # the substring "join" in the SQL and rejected a correct query. Same
    # question-words-vs-SQL-words shape as the eight above.
    "semantic_antijoin_mismatch",
    # Run-7: see HardcodedLiteralValidator. Moving a literal out of a LEFT
    # JOIN's ON clause changes the result set by design, so this can never
    # be asserted from the question wording alone.
    "semantic_left_join_on_literal",
}


_ANTI_JOIN_TABLES = ("script_assignment", "evaluation_attempt")


def _inner_join_tables(select_node: exp.Select) -> set[str]:
    """
    Table names reached by a genuine INNER join in THIS query block.

    FIX-A2. The previous test was `'join' in sql_lower`, which is also true of
    `LEFT JOIN` — the variable was named has_inner_join and could not tell the
    two apart. sqlglot records the outer-ness on the Join node, so ask it.
    """
    inner: set[str] = set()
    for join in select_node.args.get("joins", []) or []:
        side = (getattr(join, "side", "") or "").upper()
        kind = (getattr(join, "kind", "") or "").upper()
        if side in ("LEFT", "RIGHT", "FULL") or kind == "CROSS":
            continue
        tbl = join.this if isinstance(join.this, exp.Table) else None
        if tbl is not None and tbl.name:
            inner.add(tbl.name.lower())
    return inner


def _has_anti_join_pattern(ast: exp.Expression) -> bool:
    """True if the statement expresses absence structurally, anywhere."""
    for node in ast.walk():
        if isinstance(node, exp.Not) and node.find(exp.Exists):
            return True
        if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
            return True
        if isinstance(node, exp.Except):
            return True
        if isinstance(node, exp.Join):
            side = (getattr(node, "side", "") or "").upper()
            if side in ("LEFT", "RIGHT", "FULL"):
                return True
    return False


def _or_guarded_by_is_null(col: exp.Expression, alias: str) -> bool:
    """
    True when `col` sits inside an OR whose other branch is an IS NULL test on
    the same table alias.

    FIX-N1. `LEFT JOIN x ... WHERE (x.c IS NULL OR x.c <> 'V')` is the explicit
    outer-join-preserving idiom; the rows with no match survive the filter, so
    the LEFT JOIN is not silently an INNER JOIN.
    """
    node = col.parent
    while node is not None:
        if isinstance(node, exp.Or):
            for is_node in node.find_all(exp.Is):
                if not isinstance(is_node.expression, exp.Null):
                    continue
                target = is_node.this
                if isinstance(target, exp.Column) and (target.table or '').lower() == alias:
                    return True
        node = node.parent
    return False


def _enum_value_justified(value: str, nl_lower: str, nl_under_to_space: str) -> bool:
    """
    Does the question gesture at this enum value in ANY reasonable surface form?

    Exact and separator variants were already tested by the caller. This adds
    the morphological cases that made the check feel unsafe: an English question
    says "scanning" where the enum says SCANNED, "approve" where it says
    APPROVED, "hold" where it says HELD. Stem comparison rather than a synonym
    table, because the enum values in this schema are overwhelmingly regular
    past participles.

    Deliberately generous: a false NEGATIVE here blocks a correct query, which
    is the failure mode that got this check demoted in the first place.
    """
    haystacks = (nl_lower, nl_under_to_space)
    for token in str(value).lower().replace("-", "_").split("_"):
        if len(token) < 4:
            continue
        stems = {token}
        for suffix in ("ed", "d", "s", "ing", "en"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                stems.add(token[: -len(suffix)])
        # Re-inflect each stem so SCANNED also matches "scanning"/"scans".
        for stem in list(stems):
            base = stem.rstrip("n") if stem.endswith("nn") else stem
            stems.update({base + "ing", base + "s", base + "e", base + "ed"})
        for stem in stems:
            if len(stem) < 4:
                continue
            if any(stem in hay for hay in haystacks):
                return True
    return False


# ── Free-text literal grounding (Run-7) ─────────────────────────────────────
# Distinct from _enum_value_justified above, which governs CHECK-constrained
# columns. This one governs the opposite case: a literal compared to a column
# with NO enum constraint -- a name, a code, a free-form action string. There
# is no schema list to validate such a value against, so the only available
# evidence that the model did not invent it is the question itself.
#
# Batch run 20260818_085111, Q175: "Which users have auxiliary roles granted
# by the Custodian Admin?" produced
#     WHERE au_granted.display_name = 'COE Office'
# and shipped as Success. 'COE Office' appears nowhere in the question; the
# model substituted a different, plausible-looking principal. The query
# returns rows -- the WRONG person's grants -- with full confidence.
#
# The bar is deliberately the lowest one that catches fabrication: ZERO
# lexical support. A single shared word of four or more characters, in any
# separator or case form, is enough to pass. That keeps every legitimate
# encoding in the corpus silent -- 'BOARD_CLOSURE' for "board closure
# actions", 'DEK_REWRAP' for "the DEK was re-wrapped", 'CROSS_LISTING' for
# "cross-listing relationships" -- while a value with no anchor at all in
# the question cannot be justified by any reading of it.
#
# Scope limits that keep the false-positive rate at zero on the current
# corpus, and that are the reason this may block rather than advise:
#   * Equality only. An IN list enumerates a domain rather than naming one
#     value, and its members are routinely absent from the question.
#   * Columns with allowed_values are skipped entirely -- the enum check
#     owns those, with its own morphological tolerance.
#   * Numeric literals are skipped -- HardcodedLiteralValidator owns those.
#   * Pattern operands (LIKE/ILIKE) are skipped: a wildcard fragment is a
#     search term, and the model is explicitly instructed to build them.
#   * Any literal shorter than four characters is skipped, because a short
#     token carries too little signal to call fabricated.
_GROUNDING_MIN_TOKEN = 4


def _normalise_for_grounding(text: str) -> str:
    """Lowercase, strip separators. 're-wrapped' and 'REWRAP' converge."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _literal_is_grounded(value: str, nl_question: str) -> bool:
    """
    Does any word-sized piece of this literal appear in the question?

    Compared on the separator-stripped forms of both sides, so 'DEK_REWRAP'
    matches "re-wrapped" and 'CROSS_LISTING' matches "cross-listing". A
    single hit anywhere is sufficient -- this is a fabrication test, not a
    paraphrase test.
    """
    haystack = _normalise_for_grounding(nl_question)
    if not haystack:
        # No question to check against: cannot call anything fabricated.
        return True
    whole = _normalise_for_grounding(value)
    if whole and whole in haystack:
        return True
    for token in re.split(r"[^A-Za-z0-9]+", str(value)):
        piece = _normalise_for_grounding(token)
        if len(piece) < _GROUNDING_MIN_TOKEN:
            continue
        if piece in haystack:
            return True
        # Tolerate regular English inflection in either direction, the same
        # way _enum_value_justified does for CHECK-constrained values.
        for suffix in ("ed", "s", "ing", "e"):
            if piece.endswith(suffix) and len(piece) - len(suffix) >= 3:
                if piece[: -len(suffix)] in haystack:
                    return True
        for suffix in ("ed", "s", "ing"):
            if piece + suffix in haystack:
                return True
    return False


def _resolve_unqualified(name: str, alias_map: dict, schema_map: dict) -> str | None:
    """
    Base table owning an UNQUALIFIED column, when exactly one table in
    scope has it. Ambiguity returns None -- guessing which table a bare
    column belongs to is how a validator starts reading the wrong
    constraint and rejecting correct SQL.
    """
    target = (name or "").lower()
    if not target:
        return None
    owners = {
        table for table in set(alias_map.values())
        if any(
            col.lower() == target
            for col in (getattr((schema_map or {}).get(table), "columns", {}) or {})
        )
    }
    return owners.pop() if len(owners) == 1 else None


def _column_is_controlled_vocabulary(
    column: "exp.Column", table: str, schema_map: dict,
) -> bool:
    """
    True when the column's values come from a vocabulary the schema
    defines, rather than being free text.

    Three sources, all equally disqualifying for a fabrication test:
      * a CHECK (... IN (...)) list       -- allowed_values
      * values seen in the DDL's own seed -- observed_values
      * a foreign key into a catalog table

    The FK case is the one that is easy to miss. `academic_unit.unit_type`
    carries no CHECK -- it is an FK to unit_type_config -- but 'COURSE' is
    still a schema-supplied value the model can legitimately reach from a
    retrieved chunk without the question naming it. Q177 of run
    20260818_085111 is exactly that shape, and treating it as invented
    would reject correct SQL.
    """
    inv = (schema_map or {}).get(table)
    if inv is None:
        return False
    name = (column.name or "").lower()
    for col_name, col in (getattr(inv, "columns", {}) or {}).items():
        if col_name.lower() != name:
            continue
        if getattr(col, "allowed_values", None) or getattr(col, "observed_values", None):
            return True
    for fk in getattr(inv, "foreign_keys", []) or []:
        if (getattr(fk, "from_col", "") or "").lower() == name:
            return True
    return False


def _column_has_enum(column: "exp.Column", alias_map: dict, schema_map: dict) -> bool:
    """True when the resolved column carries a CHECK(... IN ...) value list."""
    table = alias_map.get((column.table or "").lower()) or (column.table or "").lower()
    inv = (schema_map or {}).get(table)
    if inv is None:
        return False
    col = (getattr(inv, "columns", {}) or {}).get(column.name)
    if col is None:
        for name, candidate in (getattr(inv, "columns", {}) or {}).items():
            if name.lower() == (column.name or "").lower():
                col = candidate
                break
    return bool(col is not None and getattr(col, "allowed_values", None))


def _advisory_or_fail(event: str, message: str, sql: str) -> ValidationResult | None:
    """
    Return a failing ValidationResult for `event`, or None when the check is
    configured advisory-only (log-and-continue). Centralises the strict/advisory
    policy so it is one edit to change, not a scattered set of behaviours.
    """
    if event in _ADVISORY_SEMANTIC_EVENTS:
        logger.info(
            component="sql_validator",
            event=f"{event}_advisory",
            note="synonym/encoding-blind check demoted to advisory; not failing",
            detail=(message or "")[:100],
        )
        return None
    return ValidationResult(passed=False, step="semantic", message=message, sql=sql)


# ── Constants ─────────────────────────────────────────────────────────────────

# Ordinal/sequential columns that should not be averaged directly.
# AVG(page_number) averages page *numbers* (meaningless), not page *count*.
_ORDINAL_COLUMNS = frozenset(HEURISTICS.get('ordinal_columns', []))

# Known phantom string values the LLM invents as placeholder IDs.
_PHANTOM_STRING_PATTERNS = frozenset(HEURISTICS.get('phantom_patterns', []))

# Known safe integer literals that appear frequently in valid SQL.
SAFE_LITERALS = frozenset(str(x) for x in HEURISTICS.get('safe_literals', []))


# ── Step 7: Semantic heuristic checks ─────────────────────────────────────────

# ══ PURE CONTRACT CHECKS  (FIX-R4 — shared by serve validation AND corpus gate) ══
# These two checks caused a dominant post-fine-tune failure class because the
# TRAINING CORPUS contained gold SQL that violates them (per/for-each rows with
# INNER-join-only gold). Fine-tuning on SQL your own validator rejects trains
# failures in. Extracted as pure (question, sql) -> error|None so that
# fine_tuning/preprocess/quality.gate() enforces the identical contract on the
# corpus. NEVER fork this logic — the class checks below call these.

def per_entity_left_join_error(question: str, sql: str) -> str | None:
    """Check 16 core: per-entity aggregate served by INNER JOIN drops
    zero-count entities. Returns the failure message or None."""
    query_lower = question.lower()
    sql_lower   = sql.lower()
    per_x_triggers     = [' per ', ' for each ']
    zero_group_exclude = [' with ', ' that have ', ' assigned to ']
    padded_query = f" {query_lower} "
    asks_per_x = (any(p in padded_query for p in per_x_triggers)
                  or 'including those with none' in query_lower)
    has_exclusion = any(p in padded_query for p in zero_group_exclude)
    sql_has_aggregate = bool(re.search(r'\b(count|sum|avg|min|max)\s*\(', sql_lower))
    if not (asks_per_x and not has_exclusion and sql_has_aggregate):
        return None
    has_left_join = bool(re.search(r'\b(?:left|right|full)\s+(?:outer\s+)?join\b', sql_lower))
    has_any_join  = bool(re.search(r'\bjoin\b', sql_lower))
    if not has_left_join and has_any_join:
        return (
            "The question asks for a metric 'per' or 'for each' entity. "
            "Using an INNER JOIN drops entities that have a count of zero. "
            "You MUST use a LEFT JOIN to ensure entities with zero "
            "associated records are included in the results."
        )
    return None


def avg_duration_epoch_error(question: str, sql: str) -> str | None:
    """Check 8 core: 'average duration/time' must use EXTRACT(EPOCH ...).
    Returns the failure message or None."""
    query_lower = question.lower()
    sql_lower   = sql.lower()
    avg_time_patterns = ['average duration', 'average time', 'avg duration', 'avg time']
    asks_avg_time = any(p in query_lower for p in avg_time_patterns)
    uses_epoch = 'extract(epoch' in sql_lower or 'extract (epoch' in sql_lower
    if asks_avg_time and not uses_epoch:
        return (
            "The question asks for 'average duration' or 'average time'. "
            "You MUST use EXTRACT(EPOCH FROM (end_ts - start_ts)) / 86400 to "
            "compute durations before averaging. Do NOT use "
            "AVG(timestamp - timestamp)."
        )
    return None


class SemanticValidator(BaseValidationStep):
    name = "SemanticValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 7: Lightweight heuristic logic checks.
        """
        sql = ctx.working_sql or ctx.sql
        original_query = ctx.original_query
        schema_map = ctx.schema_map
        
        sql_lower = sql.lower()
        query_lower = original_query.lower()

        # ── Check 1: Anti-join mismatch ────────────────────────────────────
        # Detect: question has negation words AND the SQL expresses no absence.
        #
        # FIX-A2 (false positive, Q31 of batch run 20260813_144312). Two defects,
        # both from matching substrings instead of reading the AST:
        #   * `has_inner_join = 'join' in sql_lower` is True for a query built
        #     entirely from LEFT JOINs. The variable name asserted a distinction
        #     the code never made.
        #   * `'is null' in sql_lower` / `'not exists' in sql_lower` scan the
        #     whole statement as text, so a guard inside an unrelated CTE counts
        #     while a genuine one written as `NOT EXISTS(...)` across a line
        #     break does not.
        # Now AST-driven, and demoted to advisory: the trigger is a word in the
        # question, which is the family this file already rules must not fail a
        # query (see _ADVISORY_SEMANTIC_EVENTS above).
        negation_phrases = HEURISTICS.get('anti_join_negation_phrases', [])
        has_negation = any(phrase in query_lower for phrase in negation_phrases)

        if has_negation:
            _anti_ok = True
            try:
                _ast = sqlglot.parse_one(sql, dialect="postgres")
                if _ast is not None:
                    _inner_joined = set()
                    for _sel in _ast.find_all(exp.Select):
                        _inner_joined |= _inner_join_tables(_sel)
                    _anti_ok = _has_anti_join_pattern(_ast) or not _inner_joined
            except Exception:
                _anti_ok = True

            if not _anti_ok:
                logger.warning(
                    component="sql_validator",
                    event="semantic_antijoin_mismatch",
                    query_preview=original_query[:60],
                )
                _adv = _advisory_or_fail(
                    "semantic_antijoin_mismatch",
                    "The question asks for items that are missing/without/have no "
                    "related records, but the SQL uses INNER JOIN which silently "
                    "excludes those items. Use LEFT JOIN ... WHERE ... IS NULL "
                    "or WHERE NOT EXISTS (...) to find items WITHOUT matches.",
                    sql,
                )
                if _adv is not None:
                    return _adv

        # ── Check 2: Percentage without * 100 ──────────────────────────────
        # Only trigger when question explicitly mentions "percent" or "%"
        # and SQL has no multiplication by 100.
        asks_percentage = ('percent' in query_lower or '%' in original_query)
        has_100_mult = ('100' in sql_lower)
        if asks_percentage and not has_100_mult:
            # Don't trigger if the SQL already uses a percentage function
            # or if the question is about "percentile" (different concept).
            if 'percentile' not in query_lower:
                logger.warning(
                    component="sql_validator",
                    event="semantic_percentage_missing",
                    query_preview=original_query[:60],
                )
                return ValidationResult(
                    passed=False, step="semantic",
                    message=(
                        "The question asks for a percentage but the SQL does not "
                        "multiply by 100. Use the FILTER pattern: "
                        "COUNT(*) FILTER (WHERE <condition>) * 100.0 / NULLIF(COUNT(*), 0) "
                        "to compute a percentage (0-100 range)."
                    ),
                    sql=sql,
                )

        # ── Check 3: "Average per" without AVG wrapper ─────────────────────
        # Pattern: question says "average X per Y" but SQL has no AVG().
        # The correct pattern is:
        #   SELECT AVG(cnt) FROM (SELECT y_col, COUNT(*) AS cnt ... GROUP BY y_col) sub
        avg_per_patterns = [
            'average per ', 'avg per ', 'average number per ',
            'average count per ', 'mean per ', 'average per-',
        ]
        asks_avg_per = any(p in query_lower for p in avg_per_patterns)
        has_avg = 'avg(' in sql_lower or 'avg (' in sql_lower
        if asks_avg_per and not has_avg:
            logger.warning(
                component="sql_validator",
                event="semantic_avg_per_missing",
                query_preview=original_query[:60],
            )
            return ValidationResult(
                passed=False, step="semantic",
                message=(
                    "The question asks for 'average per' but the SQL has no AVG() "
                    "function. For 'average X per Y', use a subquery: "
                    "SELECT AVG(cnt) FROM (SELECT y_col, COUNT(*) AS cnt ... "
                    "GROUP BY y_col) sub. Do NOT nest aggregate functions "
                    "like AVG(COUNT(*))."
                ),
                sql=sql,
            )

        # ── Check 4: NOT_ASSIGNED + INNER JOIN script_assignment/attempt ──
        # The schema guarantees this one: answer_script carries
        #   CHECK (block_status IN ('NONE','MALPRACTICE') OR evaluation_status = 'NOT_ASSIGNED')
        # and a script with evaluation_status = 'NOT_ASSIGNED' has no
        # script_assignment or evaluation_attempt row yet, so an INNER JOIN to
        # either returns zero rows. Being DDL-derived, this check may hard-fail
        # — but only when it has actually established the pattern.
        #
        # FIX-A3 (false positive, Q43 of batch run 20260813_144312). The old
        # implementation was three substring tests over the whole statement:
        #   has_not_assigned      = "'not_assigned'" in sql_lower
        #   joins_assignment_tables = 'evaluation_attempt' in sql_lower
        #   uses_not_exists_pattern = 'not exists' in sql_lower
        # so it fired on Q43, where the literal appears inside
        #   s.evaluation_status IN ('NOT_ASSIGNED','ASSIGNED','IN_PROGRESS')
        # — a membership test that explicitly admits assigned scripts — and the
        # only reference to evaluation_attempt was a LEFT JOIN in a different
        # CTE. Its own comment said "Check that it's actually an INNER JOIN" and
        # then did not. Three conditions must now hold in the SAME query block:
        # an equality (not IN) filter on the literal, and an INNER (not LEFT)
        # join to one of the assignment tables.
        try:
            _ast = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            _ast = None

        if _ast is not None:
            for _sel in _ast.find_all(exp.Select):
                _where = _sel.args.get("where")
                if _where is None:
                    continue
                _eq_not_assigned = False
                for _eq in _where.find_all(exp.EQ):
                    _col = _eq.this if isinstance(_eq.this, exp.Column) else None
                    _lit = _eq.expression
                    if _col is None or not isinstance(_lit, exp.Literal):
                        continue
                    if (_col.name or "").lower() != "evaluation_status":
                        continue
                    if str(_lit.this).upper() == "NOT_ASSIGNED":
                        _eq_not_assigned = True
                        break
                if not _eq_not_assigned:
                    continue

                _inner = _inner_join_tables(_sel)
                _conflict = _inner & set(_ANTI_JOIN_TABLES)
                if not _conflict:
                    continue
                if _sel.find(exp.Exists) is not None:
                    continue

                logger.warning(
                    component="sql_validator",
                    event="semantic_not_assigned_join_conflict",
                    query_preview=original_query[:60],
                    joined=sorted(_conflict),
                )
                return ValidationResult(
                    passed=False, step="semantic",
                    message=(
                        f"The SQL filters evaluation_status = 'NOT_ASSIGNED' and "
                        f"INNER JOINs {', '.join(sorted(_conflict))} in the same "
                        f"query block. A NOT_ASSIGNED script has no row in that "
                        f"table, so the join returns zero rows. Filter on "
                        f"answer_script.evaluation_status alone, use a LEFT JOIN, "
                        f"or use NOT EXISTS to assert the absence of assignments."
                    ),
                    sql=sql,
                )

        # ── Check 5: Tautology Check ───────────────────────────────────────
        # Fixes Pattern B: Semantic Misunderstanding of Aggregations
        # specifically tautologies like Q12 where COUNT(*) / COUNT(*) = 100%
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                for eq in ast.find_all(exp.EQ):
                    if eq.left.sql(dialect="postgres").lower() == eq.right.sql(dialect="postgres").lower():
                        logger.warning(
                            component="sql_validator",
                            event="semantic_tautology_detected",
                            query_preview=original_query[:60],
                            tautology=eq.sql(dialect="postgres")
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                f"SQL contains a tautology: '{eq.sql(dialect='postgres')}'. "
                                "This indicates a logic error where a column is compared to itself "
                                "instead of to a literal or a joined column. Please review the JOIN or WHERE conditions."
                            ),
                            sql=sql,
                        )
        except Exception:
            pass

        # ── Check 6: Noun Matching (Lightweight Heuristic) ───────────────────
        stop_words = {'what', 'is', 'the', 'how', 'many', 'show', 'me', 'list', 'all', 'of', 'in', 'for', 'to', 'a', 'an', 'and', 'or', 'with', 'who', 'which', 'that', 'are', 'were', 'was', 'on', 'at', 'by', 'from', 'get'}
        words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        key_words = words - stop_words

        domain_synonyms = {
            'evaluator': ['faculty', 'evaluator', 'pool', 'assignment', 'attempt'],
            'course': ['course', 'board', 'academic_unit', 'program'],
            'board': ['board'],
            'student': ['student', 'result', 'script'],
            'script': ['script', 'answer', 'scan_history', 'scan', 'device'],
            'revaluation': ['reval'],
            'coordinator': ['coordinator'],
            'moderation': ['moderation'],
        }

        for noun, mapped_terms in domain_synonyms.items():
            if noun in key_words:
                if not any(term in sql_lower for term in mapped_terms):
                    logger.warning(
                        component="sql_validator",
                        event="semantic_noun_missing",
                        missing_noun=noun,
                        query_preview=original_query[:60],
                    )
                    _adv = _advisory_or_fail(
                        "semantic_noun_missing",
                        f"The question mentions '{noun}', but the SQL does not seem to query "
                        f"any related tables or columns. Ensure you are joining the correct tables.",
                        sql,
                    )
                    if _adv is not None:
                        return _adv
                    continue

        # ── Check 7: "per/by" without GROUP BY ───────────────────────────────
        per_by_patterns = [' per ', ' by ']
        asks_per_by = any(p in query_lower for p in per_by_patterns)
        has_group_by = bool(re.search(r'group\s+by', sql_lower))

        ignore_by_phrases = [
            'order by', 'filter by', 'approved by', 'evaluated by',
            'created by', 'published by', 'scanned by', 'superseded by',
            'replaced by', 'grouped by', 'measured by', 'encrypted by',
            'placed by', 'started by', 'applied by', 'granted by',
            'coordinated by', 'initiated by', 'marked by',
            ' by more than', ' by less than', ' by at least',
            ' by exactly', ' by a ', ' by the ',
        ]
        is_ignored_by = any(p in query_lower for p in ignore_by_phrases)

        if asks_per_by and not has_group_by and not is_ignored_by:
            logger.warning(
                component="sql_validator",
                event="semantic_per_by_missing_group",
                query_preview=original_query[:60],
            )
            _res = _advisory_or_fail(
                "semantic_per_by_missing_group",
                "The question asks for an aggregation 'per' or 'by' something, "
                "but the SQL lacks a GROUP BY clause. You MUST include a GROUP BY "
                "clause when grouping data.",
                sql,
            )
            if _res:
                return _res

        # ── Check 8: "average duration/time" uses EXTRACT(EPOCH) ─────────────
        # FIX-R4: logic lives in avg_duration_epoch_error() at module top so
        # the fine-tuning corpus gate enforces the identical contract.
        _avg_err = avg_duration_epoch_error(original_query, sql)
        if _avg_err:
            logger.warning(
                component="sql_validator",
                event="semantic_avg_time_no_epoch",
                query_preview=original_query[:60],
            )
            return ValidationResult(
                passed=False, step="semantic",
                message=_avg_err,
                sql=sql,
            )

        # ── Check 9: Self-referencing subtraction ─────────────────────────────
        # Catches: column_a - column_a (always 0). Observed in Q18 where the
        # model computed esc.synced_at - esc.synced_at instead of
        # b.created_at - esc.synced_at. A subtraction of a column from itself
        # is almost certainly a bug — the model confused which column to use.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                for sub_node in ast.find_all(exp.Sub):
                    left_sql = sub_node.left.sql(dialect="postgres").lower().strip()
                    right_sql = sub_node.right.sql(dialect="postgres").lower().strip()
                    if left_sql and right_sql and left_sql == right_sql:
                        logger.warning(
                            component="sql_validator",
                            event="semantic_self_ref_subtraction",
                            expression=sub_node.sql(dialect="postgres"),
                            query_preview=original_query[:60],
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                f"SQL subtracts a column from itself: "
                                f"'{sub_node.sql(dialect='postgres')}'. "
                                f"This always evaluates to zero. One side of the "
                                f"subtraction must reference a different column or table."
                            ),
                            sql=sql,
                        )
        except Exception:
            pass

        # ── Check 10: AVG of ordinal/sequential columns ──────────────────────
        # Catches: AVG(page_number), AVG(display_order), AVG(version),
        # AVG(depth). These columns are ordinals/sequence numbers, not
        # measures. AVG(page_number) gives the average page number (meaningless),
        # not the average page count. Observed in Q127.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                for avg_node in ast.find_all(exp.Avg):
                    for col in avg_node.find_all(exp.Column):
                        if col.name and col.name.lower() in _ORDINAL_COLUMNS:
                            logger.warning(
                                component="sql_validator",
                                event="semantic_avg_ordinal",
                                column=col.name,
                                query_preview=original_query[:60],
                            )
                            _res = _advisory_or_fail(
                                "semantic_avg_ordinal",
                                (
                                    f"SQL computes AVG({col.name}), but '{col.name}' is "
                                    f"an ordinal/sequence column (not a measure). "
                                    f"To compute an average count, use a subquery: "
                                    f"SELECT AVG(cnt) FROM (SELECT ..., COUNT(*) AS cnt "
                                    f"... GROUP BY ...) sub."
                                ),
                                sql=sql,
                            )
                            if _res:
                                return _res
        except Exception:
            pass

        # ── Check 11: Hardcoded phantom string IDs ───────────────────────────
        # Catches: WHERE script_id = '12345' or WHERE column = '123' when the
        # question never mentions that value. These are phantom values the LLM
        # invents. Step 8 already catches integer IDs; this catches string
        # literals used against obvious ID columns. Observed in Q80, Q103.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                for eq in ast.find_all(exp.EQ):
                    col_node = None
                    val_node = None
                    if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Literal):
                        col_node, val_node = eq.left, eq.right
                    elif isinstance(eq.right, exp.Column) and isinstance(eq.left, exp.Literal):
                        col_node, val_node = eq.right, eq.left

                    if col_node and val_node and val_node.is_string:
                        val_str = val_node.this
                        col_name = (col_node.name or "").lower()
                        # Check if it's a known phantom value and not mentioned in the question
                        if val_str in _PHANTOM_STRING_PATTERNS and val_str not in original_query:
                            logger.warning(
                                component="sql_validator",
                                event="semantic_phantom_string_id",
                                column=col_name,
                                value=val_str,
                            )
                            return ValidationResult(
                                passed=False, step="semantic",
                                message=(
                                    f"SQL contains a suspicious hardcoded string literal "
                                    f"'{val_str}' in condition {col_name} = '{val_str}'. "
                                    f"This value does not appear in the question. Do NOT "
                                    f"invent IDs. Remove this filter or replace it with "
                                    f"the correct value from the question."
                                ),
                                sql=sql,
                            )
        except Exception:
            pass

        # ── Check 12: Tautological aggregation on GROUP BY key ───────────────
        # Catches: GROUP BY qp.id + SELECT SUM(qp.total_marks) or
        #          GROUP BY q.id + SELECT COUNT(DISTINCT q.id).
        # When the GROUP BY key is the same as the aggregated column, the
        # aggregation is over exactly one distinct value per group:
        #   SUM(x) where x is the group key = x itself
        #   COUNT(DISTINCT x) = 1 for every group
        # This is almost always a logic error. Observed in Q126, Q143.
        #
        # 2026-06-25 FIX: Removed bare-name matching that caused false positives
        # on Q8/Q20/Q27/Q59 etc.  Previously, GROUP BY b.id + COUNT(sa.id) tripped
        # the check because both columns ended in '.id' under the bare-name index.
        # Now matches only on fully-qualified table.column signatures.  The narrow
        # edge case lost (qualified GROUP BY + bare aggregate column from the same
        # table) is rare and the underlying SQL is usually correct anyway since
        # PostgreSQL resolves the bare column via the FROM clause.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                select_node = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
                if select_node:
                    group_by = select_node.args.get("group")
                    if group_by:
                        # Collect grouped column signatures (table.column only)
                        grouped_cols = set()
                        for gc in group_by.find_all(exp.Column):
                            sig = f"{(gc.table or '').lower()}.{(gc.name or '').lower()}"
                            grouped_cols.add(sig)

                        # Check aggregated columns against grouped columns
                        for agg in select_node.find_all(exp.AggFunc):
                            for col in agg.find_all(exp.Column):
                                col_sig = f"{(col.table or '').lower()}.{(col.name or '').lower()}"
                                if col_sig in grouped_cols:
                                    agg_name = type(agg).__name__.upper()
                                    # COUNT(DISTINCT key) = always 1; SUM(key) = key itself
                                    if agg_name in ('COUNT', 'SUM'):
                                        col_display = col.sql(dialect="postgres")
                                        logger.warning(
                                            component="sql_validator",
                                            event="semantic_tautological_agg",
                                            agg=agg_name,
                                            column=col_display,
                                        )
                                        return ValidationResult(
                                            passed=False, step="semantic",
                                            message=(
                                                f"SQL applies {agg_name}({col_display}) but "
                                                f"'{col_display}' is also a GROUP BY key. "
                                                f"{agg_name} over a GROUP BY key is tautological: "
                                                f"SUM returns the value itself, COUNT(DISTINCT) "
                                                f"always returns 1. Use a different column for "
                                                f"the aggregation, or restructure the query."
                                            ),
                                            sql=sql,
                                        )
        except Exception:
            pass

        # ── Check 13: Window Function + GROUP BY Clash ───────────────
        # A query with a window function (e.g. LAG OVER) and a GROUP BY in the same
        # query block is almost always logically incorrect in our domain,
        # as GROUP BY collapses the rows that the window function is meant to traverse.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                for select_node in ast.find_all(exp.Select):
                    group_by = select_node.args.get("group")
                    has_window = False
                    for expr in select_node.expressions:
                        if list(expr.find_all(exp.Window)):
                            has_window = True
                            break
                    if group_by and has_window:
                        logger.warning(
                            component="sql_validator",
                            event="semantic_window_groupby_clash"
                        )
                        return ValidationResult(
                            passed=False, step="semantic",
                            message=(
                                "SQL uses both a window function (e.g., LAG, LEAD, ROW_NUMBER) "
                                "and a GROUP BY clause in the same query block. This is usually "
                                "an error because GROUP BY collapses rows before the window "
                                "function can operate on them. If you need to aggregate and use "
                                "a window function, do the aggregation in a subquery or CTE first, "
                                "then apply the window function in the outer query."
                            ),
                            sql=sql,
                        )
        except Exception:
            pass

        # ── Check 14: Cartesian Product of SIBLING Child Tables ───────────
        # A genuine fan-out is two or more child fact tables joined to the SAME
        # parent alias in the SAME query block, with a non-DISTINCT aggregate
        # over one of them: each child multiplies the other's rows.
        #
        # FIX-F1 (false positive, Q43 / Q49 of batch run 20260813). The previous
        # implementation counted child tables per exp.Select without regard to
        # what they joined TO, and `ast.find_all(exp.Select)` descends into CTE
        # bodies. Two consequences:
        #   1. A CHAIN (answer_script -> evaluation_attempt -> evaluation_marks)
        #      was read as a fan-out. It is not: each hop is parent -> child, so
        #      no measure is duplicated.
        #   2. The correct remediation — pre-aggregating each child in its own
        #      CTE — was itself rejected, because the check fired inside the CTE.
        #      The correction prompt then asked the model to do what it had
        #      already done, and the retry loop stalled.
        # The check now inspects the ON clause of each joined child to find its
        # anchor alias, and fires only when two children share one anchor.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                child_tables = {
                    'evaluation_marks', 'honorarium_summary', 'script_assignment',
                    'evaluation_attempt', 'script_page',
                }
                for select_node in ast.find_all(exp.Select):
                    # alias -> table name for every source in THIS block
                    alias_to_table: dict[str, str] = {}
                    # sqlglot renamed this arg from "from" to "from_" in 30.x;
                    # the codebase targets >=23.0, so accept both spellings.
                    from_node = (select_node.args.get("from")
                                 or select_node.args.get("from_"))
                    root_aliases: list[str] = []
                    if from_node:
                        for tbl in from_node.find_all(exp.Table):
                            if not tbl.name:
                                continue
                            alias = (tbl.alias or tbl.name).lower()
                            alias_to_table[alias] = tbl.name.lower()
                            root_aliases.append(alias)
                    joins = select_node.args.get("joins", []) or []
                    for join in joins:
                        for tbl in join.find_all(exp.Table):
                            if not tbl.name:
                                continue
                            alias = (tbl.alias or tbl.name).lower()
                            alias_to_table.setdefault(alias, tbl.name.lower())

                    # anchor alias -> child aliases hanging off it
                    anchors: dict[str, set[str]] = {}
                    for join in joins:
                        tbl = join.this if isinstance(join.this, exp.Table) else None
                        if tbl is None or not tbl.name:
                            continue
                        joined_alias = (tbl.alias or tbl.name).lower()
                        if alias_to_table.get(joined_alias) not in child_tables:
                            continue
                        on_clause = join.args.get("on")
                        if on_clause is None:
                            continue
                        referenced = {
                            (c.table or '').lower()
                            for c in on_clause.find_all(exp.Column)
                            if c.table
                        }
                        for anchor in referenced - {joined_alias}:
                            if anchor in alias_to_table:
                                anchors.setdefault(anchor, set()).add(joined_alias)

                    siblings = {a: kids for a, kids in anchors.items() if len(kids) >= 2}
                    if not siblings:
                        continue

                    # Only a non-DISTINCT aggregate is inflated by the fan-out.
                    # COUNT(DISTINCT x) is duplicate-safe by construction.
                    inflatable = False
                    for expr in select_node.expressions:
                        for agg in expr.find_all(exp.AggFunc):
                            # COUNT(DISTINCT x) is duplicate-safe. sqlglot models
                            # the DISTINCT either as an arg or as an exp.Distinct
                            # child depending on version — test both.
                            if agg.args.get("distinct") or agg.find(exp.Distinct):
                                continue
                            inflatable = True
                            break
                        if inflatable:
                            break
                    if not inflatable:
                        continue

                    anchor_alias, kid_aliases = next(iter(siblings.items()))
                    kid_names = sorted(
                        alias_to_table.get(k, k) for k in kid_aliases
                    )
                    logger.warning(
                        component="sql_validator",
                        event="semantic_cartesian_explosion",
                        anchor=anchor_alias,
                        children=kid_names,
                    )
                    return ValidationResult(
                        passed=False, step="semantic",
                        message=(
                            f"SQL joins sibling child fact tables "
                            f"({', '.join(kid_names)}) to the same parent alias "
                            f"'{anchor_alias}' in one query block. Each child "
                            f"multiplies the other's rows, so the non-DISTINCT "
                            f"aggregates are inflated. Pre-aggregate each child "
                            f"in its own CTE keyed on the parent, then LEFT JOIN "
                            f"those CTEs to the parent."
                        ),
                        sql=sql,
                    )
        except Exception:
            pass

        # ── Check 15: Average of Surrogate Key ───────────────
        # Catch queries that do AVG(faculty_cache_id) instead of AVGing a count.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                for select_node in ast.find_all(exp.Select):
                    for agg in select_node.find_all(exp.Avg):
                        for col in agg.find_all(exp.Column):
                            col_name = (col.name or "").lower()
                            if col_name.endswith("_id") or col_name == "id":
                                col_display = col.sql(dialect="postgres")
                                logger.warning(
                                    component="sql_validator",
                                    event="semantic_avg_surrogate_key",
                                    column=col_display
                                )
                                return ValidationResult(
                                    passed=False, step="semantic",
                                    message=(
                                        f"SQL applies AVG({col_display}). Averaging a surrogate key "
                                        f"(an ID column) is meaningless. If the question asks for 'average X per Y', "
                                        f"you must first compute COUNT(X) grouped by Y in a subquery or CTE, "
                                        f"and then apply AVG() to that count in the outer query."
                                    ),
                                    sql=sql,
                                )
        except Exception:
            pass

        # ── Check 16: Zero-Group / LEFT JOIN ───────────────
        # FIX (Q32/Q37 over-fire): the previous trigger list included bare
        # ' every ' and ' all ', so plain detail-listing queries ("show ...
        # for every script", "... for each script that received moderation")
        # were forced to LEFT JOIN even though they do no per-entity counting.
        # Now this only fires when BOTH hold:
        #   (a) the phrasing is a per-entity metric ('per' / 'for each'), and
        #   (b) the SQL actually contains an aggregate (COUNT/SUM/AVG/...),
        #       i.e. a zero-count row could genuinely be dropped by INNER JOIN.
        # FIX-R4: logic lives in per_entity_left_join_error() at module top so
        # the fine-tuning corpus gate enforces the identical contract.
        _per_err = per_entity_left_join_error(original_query, sql)
        if _per_err:
            logger.warning(
                component="sql_validator",
                event="semantic_per_entity_inner_join"
            )
            _res = _advisory_or_fail("semantic_per_entity_inner_join", _per_err, sql)
            if _res:
                return _res

        # ── Check 17: Subject Resolution ───────────────
        # 2026-06-25: Expanded to match 'count of', 'total', 'number of',
        # 'distribution of' patterns in addition to 'how many'.
        entity_count_patterns = [
            r'how many ([a-z_]+s?)\b',
            r'(?:count of|total|number of|distribution of)\s+([a-z_]+s?)\b',
            r'([a-z_]+s?)\s+count\b',
        ]
        match = None
        for pattern in entity_count_patterns:
            match = re.search(pattern, query_lower)
            if match:
                break

        if match:
            noun = match.group(1)
            noun_map = HEURISTICS.get('entity_synonyms', {})
            target_tbl = None
            for k, v in noun_map.items():
                if noun.endswith(k) or k in query_lower:
                    target_tbl = v
                    break

            if target_tbl:
                try:
                    ast = sqlglot.parse_one(sql, dialect="postgres")
                    if ast:
                        for agg in ast.find_all(exp.Count):
                            if getattr(agg, "this", None) and isinstance(agg.this, exp.Column):
                                tbl_name = (agg.this.table or "").lower()
                                if tbl_name:
                                    alias_map = {}
                                    for select_node in ast.find_all(exp.Select):
                                        for t in select_node.find_all(exp.Table):
                                            alias = (t.alias or t.name).lower()
                                            alias_map[alias] = t.name.lower()
                                    resolved_tbl = alias_map.get(tbl_name, tbl_name)
                                    if resolved_tbl != target_tbl and resolved_tbl not in ['id', 'urn', '']:
                                        logger.warning(
                                            component="sql_validator",
                                            event="semantic_wrong_entity_count"
                                        )
                                        _res = _advisory_or_fail(
                                            "semantic_wrong_entity_count",
                                            (
                                                f"The question asks 'How many {noun}...', implying you should count the '{target_tbl}' table. "
                                                f"However, your query counts a column from '{resolved_tbl}'. "
                                                f"Make sure you are counting the correct entity (e.g. COUNT({target_tbl}.id))."
                                            ),
                                            sql,
                                        )
                                        if _res:
                                            return _res
                except Exception:
                    pass

        # ── Check 18: Over-Filtering Guard ───────────────
        # 2026-06-25 fix: expanded lifecycle_status keyword list to cover
        # evaluation-flow vocabulary.  When a question mentions evaluators,
        # marking, or assignment, filtering by lifecycle_status='ATTEMPTED'
        # is a legitimate precondition (only ATTEMPTED scripts enter the
        # evaluation flow), not an unprompted defensive filter.  Original
        # narrow keyword list flagged Q42 as a false positive.
        defensive_filters = {
            'is_final': ['final', 'latest', 'completed'],
            'approval_status': ['approved', 'rejected', 'pending'],
            'lifecycle_status': [
                'status', 'attempted', 'submitted', 'frozen', 'evaluated',
                # Evaluation-context vocabulary — only ATTEMPTED scripts can be
                # in any of the following states, so filtering on the lifecycle
                # is a domain precondition rather than a defensive over-filter.
                'assigned', 'assignment', 'evaluator', 'evaluation', 'marking',
                'reviewed', 'review', 'rescan', 'rescanned',
            ],
            'status': [
                'status', 'approved', 'rejected', 'pending',
                # Status enum values across the schema (evaluation_attempt,
                # board, bundle, answer_script, etc.).  When the question
                # uses one of these terms, filtering on a status column is
                # justified by the question's vocabulary.
                'frozen', 'assigned', 'submitted', 'open', 'closed',
                'scanning', 'scanned', 'in progress', 'in_progress',
                'eval', 'evaluation', 'evaluating', 'evaluated',
                'partially_frozen', 'partially frozen',
                'not_assigned', 'not assigned', 'unassigned',
                'archived', 'inactive', 'active',
            ]
        }

        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                for where in ast.find_all(exp.Where):
                    for eq in where.find_all((exp.EQ, exp.In)):
                        if isinstance(eq.left, exp.Column):
                            col_name = (eq.left.name or "").lower()
                            if col_name in defensive_filters:
                                justified = False
                                for kw in defensive_filters[col_name]:
                                    if kw in query_lower:
                                        justified = True
                                        break
                                if not justified:
                                    logger.warning(
                                        component="sql_validator",
                                        event="semantic_unprompted_filter",
                                        column=col_name
                                    )
                                    _adv = _advisory_or_fail(
                                        "semantic_unprompted_filter",
                                        f"SQL filters on '{col_name}' but the question does not mention it. "
                                        f"Do not apply defensive filters like '{col_name}' unless explicitly "
                                        f"justified by the question (e.g., asking for 'final', 'approved', etc.).",
                                        sql,
                                    )
                                    if _adv is not None:
                                        return _adv
        except Exception:
            pass

        # ── Check 18b: Schema-driven defensive-filter check (PHASE-1 FIX) ──
        #
        # WHY THIS EXISTS
        # ───────────────
        # The original Check 18 has a hard-coded `defensive_filters` dict
        # keyed by column name (is_final, approval_status, lifecycle_status,
        # status).  It misses defensive filters on ANY OTHER enum column.
        # In the 27-Jun batch, 3 hidden bugs in the "Success" set fell
        # into this gap:
        #
        #   Q67  added `ak.status = 'APPROVED'` to a question that just
        #        asked "who prepared the answer key" — dropping DRAFT keys.
        #        (status IS in Check 18's dict, but the value 'APPROVED'
        #         WAS in its keyword list — false-negative on the keyword
        #         path, because the NL just didn't mention it.)
        #
        #   Q91  added `au.user_type = 'ADMIN_STAFF'` to "show all bulk
        #        operations initiated by the COE Office".  user_type isn't
        #        in Check 18's defensive_filters dict at all.
        #
        #   Q162 added `ar.rule_type = 'PICK_N'` to "show THE attempt rule
        #        configuration for question 2".  rule_type isn't in
        #        Check 18's defensive_filters dict at all.
        #
        # WHAT THIS DOES
        # ──────────────
        # When the validator's schema_map is available, scan every WHERE
        # equality of the form `T.col = 'literal'`.  If `col` has a
        # CHECK (col IN (...)) constraint AND `'literal'` is in that
        # allowed-values set AND `'literal'` doesn't appear in the NL
        # (in either UPPER form, lower form, or with underscores replaced
        # by spaces) — flag it.
        #
        # We additionally skip the answer_script 4-status columns because
        # they participate in the documented "active-marking filter"
        # pattern (lifecycle_status='ATTEMPTED' is intentionally added by
        # the LLM per the system prompt's status model block).  Those are
        # exactly the legitimate cases that the keyword-driven Check 18
        # already accepted; the new check honours the same exemption.
        if schema_map:
            _AS_STATUS_EXEMPT = {
                ("answer_script", "lifecycle_status"),
                ("answer_script", "scan_status"),
                ("answer_script", "evaluation_status"),
                ("answer_script", "block_status"),
            }
            try:
                ast = sqlglot.parse_one(sql, dialect="postgres")
            except Exception:
                ast = None

            if ast is not None:
                # Build alias_map for this SQL
                cte_names: set[str] = set()
                for cte in ast.find_all(exp.CTE):
                    if cte.alias:
                        cte_names.add(cte.alias.lower())

                alias_to_table: dict[str, str] = {}
                for tbl in ast.find_all(exp.Table):
                    tname = (tbl.name or "").lower()
                    if not tname or tname in cte_names:
                        continue
                    alias = (tbl.alias or "").lower()
                    if alias:
                        alias_to_table[alias] = tname
                    alias_to_table[tname] = tname

                nl_lower = query_lower
                nl_under_to_space = nl_lower.replace('_', ' ')

                # Collect every EQ comparison we want to scan:
                #   * Anywhere in a WHERE clause
                #   * Anywhere in an INNER JOIN's ON clause
                # SKIP equalities inside a LEFT JOIN's ON clause: those are
                # legitimate join-scoping that don't drop rows from the main
                # query (Check 19 handles the LEFT-JOIN-with-WHERE-filter
                # nullification case separately).  For Q67-class bugs, the
                # defensive filter is often in an INNER JOIN ON — where the
                # filter behaves exactly like a WHERE.
                scan_eqs: list[exp.EQ] = []

                for where in ast.find_all(exp.Where):
                    for eq in where.find_all(exp.EQ):
                        scan_eqs.append(eq)

                for join in ast.find_all(exp.Join):
                    side = (getattr(join, 'side', None) or '')
                    if isinstance(side, str) and side.upper() == 'LEFT':
                        continue  # LEFT JOIN ON: skip
                    on_clause = join.args.get('on')
                    if on_clause is None:
                        continue
                    for eq in on_clause.find_all(exp.EQ):
                        scan_eqs.append(eq)

                # Scan all WHERE equalities + INNER JOIN ON equalities.  We
                # already trust Check 18 for the 4 hard-coded columns and
                # don't want to double-fire when it accepted them.
                checked_18_cols = {
                    'is_final', 'approval_status', 'lifecycle_status', 'status'
                }
                for eq in scan_eqs:
                    lhs, rhs = eq.left, eq.right
                    if not (isinstance(lhs, exp.Column) and isinstance(rhs, exp.Literal)
                            and rhs.is_string):
                        continue
                    col_name = (lhs.name or "").lower()

                    # For the 4 hard-coded columns Check 18 owns, only second-
                    # guess Check 18 when the column reference is in a JOIN ON
                    # (Check 18 only walks WHERE, so it missed it).  When
                    # Check 18 already had a chance and let it through, don't
                    # override it from here — its keyword whitelists for
                    # those columns are deliberately broader than ours.
                    in_where = False
                    p = eq.parent
                    while p is not None:
                        if isinstance(p, exp.Where):
                            in_where = True
                            break
                        p = p.parent
                    if col_name in checked_18_cols and in_where:
                        continue

                    tbl_part = (lhs.table or "").lower()
                    target = alias_to_table.get(tbl_part)
                    if target is None and tbl_part in schema_map:
                        target = tbl_part
                    if target is None:
                        continue

                    if (target, col_name) in _AS_STATUS_EXEMPT:
                        continue

                    inv = schema_map.get(target)
                    if inv is None:
                        continue
                    col_info = inv.columns.get(col_name) if hasattr(inv, 'columns') else None
                    if col_info is None:
                        continue
                    allowed = getattr(col_info, 'allowed_values', None)
                    if not allowed:
                        # No CHECK enum — not a defensive-filter candidate
                        continue

                    val_raw = str(rhs.this)
                    if val_raw not in allowed:
                        # Already handled by the literal/enum-membership
                        # check inside _validate_column_types.
                        continue

                    val_lo  = val_raw.lower()
                    val_spc = val_lo.replace('_', ' ')
                    val_hyp = val_lo.replace('_', '-')

                    # Justification: does the NL mention this value in any
                    # of its surface forms?  Substring match is intentional:
                    # "approved" matches "approved", "Approved", "APPROVED",
                    # "approved keys", etc.
                    justified = (
                        val_lo  in nl_lower
                        or val_spc in nl_lower
                        or val_hyp in nl_lower
                        or val_lo  in nl_under_to_space
                        or val_spc in nl_under_to_space
                    )
                    if not justified:
                        justified = _enum_value_justified(
                            val_raw, nl_lower, nl_under_to_space,
                        )
                    if justified:
                        continue

                    logger.warning(
                        component="sql_validator",
                        event="semantic_unprompted_enum_filter",
                        table=target,
                        column=col_name,
                        value=val_raw,
                    )
                    _adv = _advisory_or_fail(
                        "semantic_unprompted_enum_filter",
                        f"SQL filters {target}.{col_name} = '{val_raw}' "
                        f"but the question does not mention '{val_raw}' "
                        f"(or any close paraphrase).  This is a "
                        f"defensive filter not supported by the user's "
                        f"intent — remove it, or surface it as a "
                        f"clarification.  Allowed values for "
                        f"{target}.{col_name}: {sorted(allowed)}.",
                        sql,
                    )
                    if _adv is not None:
                        return _adv
                    continue

        # ── Check 19: LEFT JOIN + WHERE nullification ─────────────────
        # Catches: LEFT JOIN ... WHERE right_alias.col = value
        # This silently converts LEFT JOIN to INNER JOIN because NULL
        # rows from the LEFT side are eliminated by the WHERE filter.
        # The filter should be in the ON clause instead.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if ast:
                # FIX-N2 (false positive, Q51 of batch run 20260813_192547). A
                # LEFT JOIN alias and the WHERE clause it's paired against must
                # come from the SAME select block. The previous implementation
                # collected left_join_aliases as one set across every Select in
                # the statement -- including the bodies of CTEs -- and then
                # scanned every top-level WHERE the same way. A query joining
                # `LEFT JOIN bundles_opened b ON ...` in the outer SELECT, where
                # a CTE named bundles_opened *itself* does
                # `FROM bundle b WHERE b.created_at >= ...`, has two unrelated
                # aliases that both happen to be spelled "b". The global sets
                # matched them across scopes and flagged a LEFT JOIN that was
                # never nullified -- the WHERE belonged to the CTE, not to the
                # query that joins it. Iterate per select_node and only check a
                # WHERE against the LEFT JOIN aliases declared in that same
                # select_node's own joins.
                for select_node in ast.find_all(exp.Select):
                    left_join_aliases: set[str] = set()
                    for join in select_node.args.get("joins", []):
                        side = getattr(join, 'side', None) or ''
                        if isinstance(side, str) and side.upper() == 'LEFT':
                            tbl = join.this
                            if isinstance(tbl, exp.Table):
                                alias = (tbl.alias or tbl.name or '').lower()
                                if alias:
                                    left_join_aliases.add(alias)
                            elif hasattr(tbl, 'find'):
                                for t in tbl.find_all(exp.Table):
                                    alias = (t.alias or t.name or '').lower()
                                    if alias:
                                        left_join_aliases.add(alias)

                    if not left_join_aliases:
                        continue

                    where = select_node.args.get("where")
                    if where is None:
                        continue
                    for col in where.find_all(exp.Column):
                        col_table = (col.table or '').lower()
                        if col_table in left_join_aliases:
                            # Skip IS NULL pattern (anti-join is correct)
                            parent = col.parent
                            if isinstance(parent, exp.Is):
                                continue
                            # Skip NOT (col IS NULL) too
                            grandparent = getattr(parent, 'parent', None)
                            if isinstance(grandparent, exp.Not) and isinstance(parent, exp.Is):
                                continue
                            # FIX-N1 (false positive, Q42 of batch run
                            # 20260813). A predicate that is OR-ed with an
                            # IS NULL test on the same alias preserves the
                            # unmatched rows:
                            #   WHERE (ea.status IS NULL OR ea.status != 'X')
                            # The LEFT JOIN is explicitly guarded, not
                            # nullified. Walk up to the enclosing OR and
                            # look for the guard before failing.
                            if _or_guarded_by_is_null(col, col_table):
                                continue
                            # FIX (Q126): a LEFT JOIN is only nullified when the
                            # left column is filtered against a LITERAL value
                            # (e.g. WHERE hs.status = 'PENDING'). When the column
                            # is compared to ANOTHER COLUMN -- e.g. a correlated
                            # join key inside NOT EXISTS (WHERE q2.parent = q.id) --
                            # NULL rows are not eliminated and the pattern is fine.
                            if not _predicate_has_literal_counterpart(col):
                                continue
                            col_name = (col.name or '').lower()
                            logger.warning(
                                component="sql_validator",
                                event="semantic_left_join_nullified",
                                alias=col_table,
                                column=col_name,
                            )
                            return ValidationResult(
                                passed=False, step="semantic",
                                message=(
                                    f"LEFT JOIN on alias '{col_table}' is nullified by "
                                    f"'WHERE {col_table}.{col_name} = ...' which eliminates "
                                    f"NULL rows. Move the filter into the ON clause: "
                                    f"LEFT JOIN ... ON ... AND {col_table}.{col_name} = ..., "
                                    f"or change to INNER JOIN if zero-match rows should be excluded."
                                ),
                                sql=sql,
                            )
        except Exception:
            pass

        # ── Check 20: Missing AVG() for "average" questions ───────────────
        # Broader than Check 3 ("average per"). Triggers on any question
        # containing "average" or "avg" where the outermost SELECT has no AVG().
        avg_patterns = ['average ', ' avg ', 'average(', 'avg(']
        asks_average = any(p in f" {query_lower} " for p in avg_patterns)
        if asks_average and 'avg(' not in sql_lower and 'avg (' not in sql_lower:
            # Guard: skip if "average" appears as part of a name
            # (e.g., "Average Joe" — unlikely in our domain but safe)
            if 'averag' not in sql_lower:  # no AVG alias either
                logger.warning(
                    component="sql_validator",
                    event="semantic_avg_missing",
                    query_preview=original_query[:60],
                )
                return ValidationResult(
                    passed=False, step="semantic",
                    message=(
                        "The question asks for an 'average' but the outermost SELECT "
                        "has no AVG() function. Use AVG(column) for simple averages, "
                        "or SELECT AVG(sub.cnt) FROM (...) sub for averages of counts."
                    ),
                    sql=sql,
                )

        # ── Check 21: Missing scope filter ─────────────────────────────────
        # When the question uses a keyword that implies a specific filter,
        # verify the SQL contains that filter. Conservative design:
        # only triggers when keyword + matching table + no skip phrase.
        _keyword_filter_map = {
            'exported': {
                'sql_check': "'exported'",
                'tables': ['honorarium_summary', 'honorarium'],
                'skip_if': ['all', 'distribution', 'breakdown', 'each'],
            },
            'global': {
                'sql_check': "'global'",
                'tables': ['configuration', 'evaluation_policy'],
                'skip_if': ['all', 'every', 'each scope', 'by scope'],
            },
        }
        for keyword, spec in _keyword_filter_map.items():
            if keyword in query_lower:
                if any(skip in query_lower for skip in spec['skip_if']):
                    continue
                if any(t in sql_lower for t in spec['tables']):
                    if spec['sql_check'] not in sql_lower:
                        logger.warning(
                            component="sql_validator",
                            event="semantic_missing_scope_filter",
                            keyword=keyword,
                        )
                        _adv = _advisory_or_fail(
                            "semantic_missing_scope_filter",
                            f"The question mentions '{keyword}' but the SQL has no "
                            f"corresponding filter (expected {spec['sql_check']} in the query). "
                            f"Add the appropriate WHERE clause to scope the results.",
                            sql,
                        )
                        if _adv is not None:
                            return _adv
                        continue

        # ── Check 22: HAVING on "list all" questions ───────────────────────
        # When the question says "list all" or "show all", a HAVING COUNT > 1
        # inappropriately filters out entities. The user wants ALL entities.
        all_patterns = ['list all', 'show all', 'find all', 'get all']
        asks_all = any(p in query_lower for p in all_patterns)
        if asks_all:
            try:
                ast = sqlglot.parse_one(sql, dialect="postgres")
                if ast:
                    having = ast.find(exp.Having)
                    if having:
                        for gt in having.find_all(exp.GT):
                            if gt.find(exp.Count) and isinstance(gt.right, exp.Literal):
                                threshold = gt.right.this
                                if threshold in ('1', '2'):
                                    logger.warning(
                                        component="sql_validator",
                                        event="semantic_having_on_list_all",
                                        threshold=threshold,
                                    )
                                    return ValidationResult(
                                        passed=False, step="semantic",
                                        message=(
                                            f"The question asks to 'list all' but the HAVING clause "
                                            f"filters out entities with count <= {threshold}. "
                                            f"Remove the HAVING clause or adjust it to match the "
                                            f"question's intent of listing ALL entities."
                                        ),
                                        sql=sql,
                                    )
            except Exception:
                pass

        return ValidationResult(passed=True, step="semantic", sql=sql)


    # ── Step 8: Hardcoded literal detection ───────────────────────────────────────

class HardcodedLiteralValidator(BaseValidationStep):
    name = "HardcodedLiteralValidator"

    def run(self, ctx: ValidationContext) -> ValidationResult:
        """
        Step 8: Detect suspiciously hardcoded integer literal IDs via AST.
        """
        sql = ctx.working_sql or ctx.sql
        original_query = ctx.original_query
        safe_literals = SAFE_LITERALS
        # FIX-R7a. This method used to carry a second, function-local
        # `import re` inside the integer-literal loop below. A name bound
        # ANYWHERE in a function body is local for the WHOLE body, so the
        # module-level `re` became invisible here and the very first use
        # of it -- the has_aggregate probe a few lines down -- raised
        # UnboundLocalError. The blanket `except Exception: pass` at the
        # bottom swallowed it and returned passed=True.
        #
        # Effect: this validator was a silent no-op for every query with
        # NO aggregate function. Aggregate queries survived only by
        # accident -- `bool(ast.find(exp.AggFunc))` short-circuits the
        # `or` before re.search is ever evaluated. Neither the LEFT-JOIN
        # ON-literal check nor the invented-integer-ID check has run on a
        # non-aggregate query since the inner import was introduced.
        # Nothing logged, because the failure path logs nothing.
        #
        # The inner import is deleted; `re` is imported at module scope
        # (line 16) and always was.
        try:
            ast = sqlglot.parse_one(sql, dialect="postgres")
            if not ast:
                return ValidationResult(passed=True, step="hardcoded", sql=sql)
            
            # [NEW] L8 Literal Detector Enhancement: Block hardcoded filters in outer joins
            # Hardcoding a *filter* in a LEFT JOIN's ON clause (e.g. ON a.id = b.id AND b.status = 'ACTIVE')
            # breaks the outer join semantics, effectively turning it into an INNER JOIN.
            #
            # FIX (Q45 false positive): a literal in a LEFT JOIN's ON clause is
            # NOT always wrong -- it is the CORRECT, required pattern for a
            # conditional outer join (keep unmatched left rows, e.g. "was the DEK
            # re-wrapped" -> LEFT JOIN audit_log ON ... AND action = 'DEK_REWRAP').
            # Moving such a literal to WHERE would wrongly drop the unmatched rows.
            # Following the same NL-aware design as Check 23, only fail when the
            # literal value actually appears in the user's question -- i.e. the
            # user asked to FILTER on it, so it belongs in WHERE. Literals that do
            # not appear in the NL are treated as legitimate join-scoping predicates.
            nl_lower = (original_query or "").lower()
            # FIX (Q27 regression): a literal in a LEFT JOIN's ON clause is the
            # REQUIRED idiom for conditional aggregation -- "count the APPROVED
            # answer keys per paper, including papers with zero" must read
            # LEFT JOIN answer_key ak ON ak.qp_id = qp.id AND ak.status = 'APPROVED'
            # so that zero-match parents survive. Moving 'APPROVED' to WHERE would
            # drop those parents. The earlier NL-aware rule wrongly fired here
            # because 'approved' appears in the question. When the statement
            # contains an aggregate (COUNT/SUM/AVG/MIN/MAX), treat every
            # LEFT-JOIN-ON literal as legitimate conditional-aggregation scope and
            # skip this check entirely.
            has_aggregate = bool(ast.find(exp.AggFunc)) or bool(
                re.search(r'\b(count|sum|avg|min|max)\s*\(', (sql or '').lower())
            )
            for join in ([] if has_aggregate else ast.find_all(exp.Join)):
                side = getattr(join, 'side', None) or ''
                if isinstance(side, str) and side.upper() == 'LEFT':
                    on_clause = join.args.get('on')
                    if on_clause:
                        for literal in on_clause.find_all(exp.Literal):
                            lit_val = str(literal.this).strip().lower()
                            if not lit_val:
                                continue
                            if lit_val not in nl_lower:
                                # Legitimate conditional-join predicate -- keep it.
                                logger.info(
                                    component="sql_validator",
                                    event="left_join_on_literal_allowed",
                                    literal=literal.this,
                                    note="literal not present in NL question; treated as conditional join scope",
                                )
                                continue
                            logger.warning(
                                component="sql_validator",
                                event="hardcoded_literal_in_left_join",
                                literal=literal.this,
                                query_preview=nl_lower[:60],
                            )
                            # FIX-R7a (continued). Un-shadowing `re` above
                            # woke this rule from a silent no-op, and on
                            # its first real exposure -- the 187 passing
                            # queries of run 20260818_085111 -- it was
                            # wrong every single time it fired: Q33, Q48
                            # and Q99 are textbook anti-joins (LEFT JOIN
                            # ... ON <literal> ... WHERE x.id IS NULL) and
                            # Q56 is a conditional outer join. In all four
                            # the literal MUST stay in the ON clause;
                            # moving it to WHERE is precisely the bug the
                            # message tells the model to introduce.
                            #
                            # The premise is unsound, not merely
                            # mis-tuned. "The value appears in the
                            # question, therefore it belongs in WHERE" is
                            # a question-words-vs-SQL-words inference --
                            # the family Runs 3-5 demoted -- and a literal
                            # in a LEFT JOIN's ON clause is never
                            # equivalent to the same literal in WHERE.
                            # Advisory, via the existing mechanism.
                            _adv = _advisory_or_fail(
                                "semantic_left_join_on_literal",
                                (f"Literal '{literal.this}' sits in a LEFT "
                                 f"JOIN ON clause and also appears in the "
                                 f"question. If it is meant to restrict the "
                                 f"result rather than scope the join, it "
                                 f"belongs in WHERE."),
                                sql,
                            )
                            if _adv is not None:
                                return _adv
        
            # ── Free-text literal grounding (Run-7) ──────────────────────
            # See _literal_is_grounded above. A string literal compared to a
            # column with no CHECK value list has nothing in the schema to
            # validate it against; the question is the only evidence that it
            # was not invented.
            alias_map = {}
            cte_names = {(c.alias or '').lower() for c in ast.find_all(exp.CTE)}
            for tbl in ast.find_all(exp.Table):
                tname = (tbl.name or '').lower()
                if tname and tname not in cte_names:
                    alias_map[(tbl.alias or tname).lower()] = tname

            for eq in ast.find_all(exp.EQ):
                # A literal inside FILTER (...) or CASE WHEN ... is a
                # BRANCH of a conditional aggregate, not a restriction on
                # the population. Q117 of run 20260818_085111 counts
                # requested-vs-approved with one FILTER per approval_status
                # value; enumerating a domain that way is correct even
                # though the question names only two of the branches.
                if (eq.find_ancestor(exp.Filter) is not None
                        or eq.find_ancestor(exp.Case) is not None):
                    continue
                col_node = val_node = None
                if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Literal):
                    col_node, val_node = eq.left, eq.right
                elif isinstance(eq.right, exp.Column) and isinstance(eq.left, exp.Literal):
                    col_node, val_node = eq.right, eq.left
                if col_node is None or not val_node.is_string:
                    continue
                value = str(val_node.this)
                if len(_normalise_for_grounding(value)) < _GROUNDING_MIN_TOKEN:
                    continue
                resolved = alias_map.get((col_node.table or '').lower())
                if resolved is None:
                    resolved = _resolve_unqualified(col_node.name, alias_map, ctx.schema_map)
                if resolved is None:
                    # Cannot tie the column to a base table, so cannot tell
                    # a CHECK-constrained column from a free-text one. Fail
                    # open: this check may only speak when it is certain.
                    continue
                if _column_is_controlled_vocabulary(col_node, resolved, ctx.schema_map):
                    # A schema-supplied value. If it is the WRONG one,
                    # semantic_unprompted_enum_filter owns that call --
                    # this check only speaks to values with no schema
                    # provenance at all.
                    continue
                if _literal_is_grounded(value, original_query):
                    continue
                target = f"{resolved}.{col_node.name}"
                logger.warning(
                    component="sql_validator",
                    event="ungrounded_literal_filter",
                    column=target,
                    literal=value,
                    query_preview=(original_query or "")[:80],
                )
                return ValidationResult(
                    passed=False, step="hardcoded_literals",
                    message=(
                        f"The SQL filters `{target} = '{value}'`, but "
                        f"'{value}' does not appear anywhere in the question "
                        f"and `{target}` has no CHECK constraint that could "
                        f"supply it. This is an invented value: the query will "
                        f"return rows, but they will be the wrong rows. Filter "
                        f"on the value the question actually names, or -- if "
                        f"the question names no value for this column -- drop "
                        f"the predicate entirely."
                    ),
                    sql=sql,
                )

            suspicious = []
            for eq in ast.find_all(exp.EQ):
                col_node = None
                val_node = None
            
                if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Literal):
                    col_node, val_node = eq.left, eq.right
                elif isinstance(eq.right, exp.Column) and isinstance(eq.left, exp.Literal):
                    col_node, val_node = eq.right, eq.left
                
                if col_node and val_node and not val_node.is_string:
                    col_name = (col_node.name or "").lower()
                    if col_name == "id" or col_name.endswith("_id"):
                        lit = val_node.this
                    
                        if lit in safe_literals:
                            continue
                        if len(lit) == 4 and lit.startswith(('19', '20')):
                            continue  # likely a year
                        if lit in original_query:
                            # User provided this number. However, if they didn't explicitly specify it's an ID,
                            # they likely meant a business code (e.g., "question 2" -> code = '2', not id = 2).
                            query_lower = original_query.lower()
                            if "id" not in query_lower and "identifier" not in query_lower:
                                # FIX-R7a (continued). Also revived by the
                                # `re` un-shadowing above. The entity noun
                                # and its number are routinely separated
                                # by a qualifier -- "attempt rule 1",
                                # "question paper 3" -- and the adjacent-
                                # only pattern read those as invented IDs.
                                # Allow up to two intervening words.
                                _named = r'(number|id|paper|script|attempt|question|rule|group|board)'
                                if not re.search(
                                    _named + r'(?:\s+\w+){0,2}\s+0*' + re.escape(lit),
                                    query_lower,
                                ):
                                    pass # still treat as suspicious
                                else:
                                    continue
                            else:
                                continue
                        
                        suspicious.append(lit)
                    
            if suspicious:
                logger.warning(
                    component="sql_validator",
                    event="hardcoded_literal_detected",
                    literals=suspicious[:3],
                    query_preview=original_query[:60],
                )
                return ValidationResult(
                    passed=False, step="hardcoded",
                    message=(
                        f"SQL contains hardcoded integer literal(s) "
                        f"{suspicious[:3]} used against an ID column. Do NOT invent numeric IDs. "
                        f"When the question references an entity by name, filter by the "
                        f"name/title/code column instead: "
                        f"WHERE table.name ILIKE '%value%'."
                    ),
                    sql=sql,
                )
        except Exception:
            pass

        return ValidationResult(passed=True, step="hardcoded", sql=sql)