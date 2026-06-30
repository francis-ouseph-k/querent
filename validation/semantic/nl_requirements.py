"""
validation/nl_requirements.py
─────────────────────────────
Parses a natural-language question into a structured set of REQUIREMENTS
that the generated SQL must satisfy.

WHY THIS EXISTS — THE STRUCTURAL GAP
────────────────────────────────────
Every check in semantic_checks.py and the L1–L5 checks in logical_audit.py
is a SQL-PATTERN DETECTOR: it walks the SQL alone, looking for a specific
anti-pattern (e.g. AVG of an ID column, INNER JOIN under a "per X" NL,
LEFT JOIN nullified by WHERE).  Each new failure mode requires a new
check, and the LLM keeps inventing new ways to be wrong that don't trip
any specific check.

The batch-run-20260625 review surfaced 39 queries that passed all
structural validators but were logically wrong.  The failures cluster
into a small number of shapes:

  A. CONSTRAINT SILENTLY DROPPED   (14 queries: Q24, Q40, Q56, Q76,
     Q92, Q104, Q116, Q128, Q130, Q141, Q144, Q147, Q166, Q189)
     NL contains a hard filter the SQL never expresses.

  B. LEFT-JOIN-FILTER-IN-ON        (5 queries: Q2, Q76, Q91, Q104, Q130)
     Filter conditions on the optional side are placed in the ON
     clause, so the LEFT JOIN silently keeps rows the user excluded.

  C. OUTPUT COLUMNS MISSING        (3 queries: Q32, Q35, Q39)
     NL says "show X, Y, Z" — SQL projects only X.

  D. AGGREGATION WRONG ENTITY      (7 queries)
     COUNT/SUM/AVG applied to the wrong side of a relationship.

  E. WRONG ENUM LITERAL            (3 queries)
     'BOARD_COORDINATOR' used where only PRIMARY/REVIEW/REVAL/THIRD
     are valid; date format '2025-2025' instead of '2025-2026'.

  F. WRONG RELATIONSHIP DIRECTION  (5 queries)
     programs as to_unit_id when they should be from_unit_id.

Every one of A, B, C and most of D, E, F is invisible to a SQL-only
audit because the SQL itself is internally consistent.  The thing
the SQL fails to satisfy is the NL CONTRACT — and the contract has
never been extracted as data.

WHAT THIS MODULE DOES
─────────────────────
It parses the NL question into a typed `QuestionRequirements` value:

  - output_columns        what the SELECT must project
  - filter_constraints    typed constraints that must appear in WHERE
                          (or in ON for anti-join polarity)
  - grouping_signals      "per X" / "for each X" / "by X" markers
  - quantifier_constraints  "more than N" / "exactly N" / "at least N"
  - aggregation_intent    avg / count / sum / max / min word in NL
  - polarity              positive | anti_join
  - entity_type_hints     "X department" → ('X', 'DEPARTMENT')

These requirements feed the new L6/L7/L8 logical-audit checks in
`logical_audit.py`, which verify the SQL satisfies each one in the
right STRUCTURAL POSITION (filter → WHERE, output → SELECT,
grouping → GROUP BY, etc).

WHY THIS IS NOT A BANDAID
─────────────────────────
Existing checks are linear: one check per failure pattern.  This
module converts the NL into a contract: any SQL that fails to
satisfy the contract is flagged, regardless of *how* the SQL is
wrong.  Adding support for a new constraint kind benefits every
downstream check at once.

DESIGN PRINCIPLES
─────────────────
  * Conservative parsing — when in doubt, don't emit a requirement.
    False positives degrade confidence on correct queries and erode
    trust in the audit signal.
  * Pure regex — no NER model, no LLM round-trip.  Cheap enough to
    run on every query.
  * Token-level expectations — each constraint carries the set of
    SQL tokens the audit can search for, not a deep AST predicate.
    Keeps the audit simple and resilient to SQL rewrites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
# Public dataclasses
# ─────────────────────────────────────────────────────────────────────────

class ConstraintKind(Enum):
    """
    Typology of NL filter constraints.

    Each kind tells the L6 audit check WHAT to look for in the SQL:

      STATUS / ENUM    a literal that must appear in WHERE (or CASE)
      TIME_RANGE       a date/timestamp range bound that must appear
                       in WHERE
      ROLE             a role/scope identifier (CUSTODIAN, EVALUATOR)
      NUMERIC          a numeric threshold (more than N, exactly N)
      BOOLEAN          an is_X column that should be filtered
      TEXT_LIKE        an ILIKE/LIKE substring constraint
      SCOPE            "in X department", "at Y campus" — typed entity
                       reference that must be present
    """
    STATUS      = "status"
    ENUM        = "enum"
    TIME_RANGE  = "time_range"
    ROLE        = "role"
    NUMERIC     = "numeric"
    BOOLEAN     = "boolean"
    TEXT_LIKE   = "text_like"
    SCOPE       = "scope"


@dataclass
class FilterConstraint:
    """
    A single typed filter constraint extracted from the NL.

    Fields:
        kind:    ConstraintKind — drives downstream check logic.
        raw:     The exact NL phrase, for diagnostics.
        tokens:  The set of SQL tokens we EXPECT to find in the SQL
                 (case-insensitive substring match).  An empty list
                 means "this constraint has no specific token to
                 search for" — the audit will use kind alone.
        must_be_in_where: True for nearly all kinds; False for
                 constraints that may legitimately appear in an ON
                 clause (e.g. anti-join shape).
    """
    kind: ConstraintKind
    raw: str
    tokens: list[str] = field(default_factory=list)
    must_be_in_where: bool = True


@dataclass
class QuestionRequirements:
    """
    The structured contract a SQL query must satisfy to correctly
    answer the NL question.

    Empty fields mean "no requirement detected" — they do NOT mean
    "no requirement exists".  The parser is intentionally conservative.
    """
    output_columns:        list[str]                  = field(default_factory=list)
    filter_constraints:    list[FilterConstraint]     = field(default_factory=list)
    grouping_signals:      list[str]                  = field(default_factory=list)
    quantifier_constraints: list[tuple[str, int]]     = field(default_factory=list)
    aggregation_intent:    Optional[str]              = None
    polarity:              str                        = "positive"   # or "anti_join"
    entity_type_hints:     list[tuple[str, str]]      = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Token sources
# ─────────────────────────────────────────────────────────────────────────

# Known status/enum value tokens that, when they appear in the NL,
# MUST also appear in the SQL.  These are upper-case in the DDL CHECK
# constraints and are searched case-insensitively in the SQL.
#
# Conservative list: only values that are unambiguous and high-signal.
# Adding a value here means "if the NL mentions this word, the SQL
# must include it as a literal".  False positives are expensive
# (degrade confidence on correct queries), so the list stays tight.
_ENUM_VALUE_TOKENS: dict[str, str] = {
    # answer_script.lifecycle_status
    "absent":         "ABSENT",
    "admitted":       "ADMITTED",
    "attempted":      "ATTEMPTED",
    "barred":         "BARRED",
    "eligible":       "ELIGIBLE",
    "withheld":       "WITHHELD",
    # bundle.status, board.status etc.
    "open":           "OPEN",
    "scanning":       "SCANNING",
    "complete":       "COMPLETE",
    "completed":      "COMPLETE",
    "reconciled":     "RECONCILED",
    # evaluation_attempt.status
    "assigned":       "ASSIGNED",
    "frozen":         "FROZEN",
    "submitted":      "SUBMITTED",
    "in progress":    "IN_PROGRESS",
    # evaluation_attempt.attempt_type
    # Note: 'primary'/'review'/'revaluation'/'third' as lowercase
    # words are NOT mapped here — they too often refer to the table
    # or action ("revaluation request", "review the script") rather
    # than the attempt_type enum value.  When the user wants to filter
    # by attempt_type, they typically write the upper-case form
    # ('PRIMARY attempt', 'a REVIEW evaluator'), which is caught by
    # the uppercase-pass below.
    "primary attempt":   "PRIMARY",
    "review attempt":    "REVIEW",
    "review evaluator":  "REVIEW",
    "reval":             "REVAL",
    "third attempt":     "THIRD",
    # result.computation_rule
    "second eval":    "SECOND_EVAL",
    "second_eval":    "SECOND_EVAL",
    # key_encryption_key.status / scope
    "active":         "ACTIVE",
    "expired":        "EXPIRED",
    "retired":        "RETIRED",
    "revoked":        "REVOKED",
    "cloud":          "CLOUD",
    "workstation":    "WORKSTATION",
    # honorarium_summary.approval_status, revaluation_request.approval_status
    "approved":       "APPROVED",
    "pending":        "PENDING",
    "rejected":       "REJECTED",
    "exported":       "EXPORTED",
    # board.deadline_enforcement
    "hard block":     "HARD_BLOCK",
    "hard deadline":  "HARD_BLOCK",
    "soft stop":      "SOFT_STOP",
    # script_hold lifecycle words
    "released":       "released_at",   # column reference, not enum
    "decommissioned": "archived_at",   # column reference
    # revaluation_request.requested_via
    "manual":         "MANUAL",
    "manually":       "MANUAL",
    "portal":         "PORTAL",
    # role-style values appearing in allowed_roles arrays
    "custodian":      "CUSTODIAN",
    "evaluator":      "EVALUATOR",
    "coordinator":    "COORDINATOR",
    "board coordinator": "BOARD_COORDINATOR",
    "external auditor":  "EXTERNAL_AUDITOR",
    "admin staff":    "ADMIN_STAFF",
    # ack/critical flags on audit_log
    "critical":       "is_critical",
    # source_type
    "manually entered": "MANUALLY_ENTERED",
    # archive_strategy
    "s3 glacier":     "S3_GLACIER",
    "keep":           "KEEP",
    # academic_unit unit_type hints
    "department":     "DEPARTMENT",
    "course":         "COURSE",
    "program":        "PROGRAM",
    "campus":         "CAMPUS",
    # relationship types
    "cross-listing":  "CROSS_LISTING",
    "cross listing":  "CROSS_LISTING",
    "prerequisite":   "PREREQUISITE",
}

# Aggregation intent keywords → canonical AGG name
_AGG_KEYWORDS: dict[str, str] = {
    "average":   "AVG",
    "avg":       "AVG",
    "mean":      "AVG",
    "total":     "SUM",
    "sum":       "SUM",
    "count":     "COUNT",
    "number of": "COUNT",
    "how many":  "COUNT",
    "maximum":   "MAX",
    "max ":      "MAX",
    "minimum":   "MIN",
    "percentage": "PCT",
    "percent":   "PCT",
    "ratio":     "PCT",
}

# Anti-join polarity markers — words that REQUIRE the SQL to use
# NOT EXISTS / IS NULL / EXCEPT / LEFT-JOIN-anti-pattern.
_ANTI_JOIN_MARKERS: tuple[str, ...] = (
    "without",
    " no ",
    "have no ",
    "have not ",
    "haven't ",
    "missing ",
    "never ",
    "but not ",
    "but no ",
    " lack ",
    "lacks ",
    " no corresponding ",
    " no matching ",
    "absent from ",
    " never been ",
)

# Common stop words to strip when extracting output column nouns
_OUTPUT_STOP = frozenset((
    "the", "a", "an", "of", "for", "to", "and", "or", "in",
    "with", "by", "per", "each", "their", "its", "his", "her",
    "these", "those", "this", "that",
))


# ─────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────────

def parse_question(nl: str) -> QuestionRequirements:
    """
    Parse an NL question into a QuestionRequirements contract.

    This function is intentionally conservative.  A requirement is
    only emitted when the NL signals it unambiguously.  When in
    doubt, the parser stays silent — the downstream audit will then
    not flag the SQL for a constraint that may not exist.
    """
    if not nl or not nl.strip():
        return QuestionRequirements()

    nl_lower = nl.lower().strip()
    req = QuestionRequirements()

    # ─── polarity ────────────────────────────────────────────────────
    # Detect anti-join shape from explicit "without/no/missing/never"
    # markers.  Order matters: we check explicit phrases first because
    # the bare " no " marker also matches inside benign words.
    for marker in _ANTI_JOIN_MARKERS:
        if marker in nl_lower:
            req.polarity = "anti_join"
            break

    # ─── aggregation intent ──────────────────────────────────────────
    # First match wins; PCT/COUNT keywords are checked before AVG
    # because "average percentage" should classify as PCT.
    for kw, agg in _AGG_KEYWORDS.items():
        if kw in nl_lower:
            req.aggregation_intent = agg
            break

    # ─── grouping signals ────────────────────────────────────────────
    # "per X" / "for each X" / "by X" patterns.  We deliberately
    # exclude the "by <Agent>" possessive form (e.g. "granted by the
    # Custodian Admin") by requiring the captured noun to NOT start
    # with a definite article.  Capture up to two words to handle
    # "per academic year", "for each scanner device".
    grouping_matches = re.findall(
        r"\b(?:per|for\s+each|grouped\s+by|broken\s+down\s+by)"
        r"\s+([a-z][a-z_]*(?:\s+[a-z][a-z_]*)?)",
        nl_lower,
    )
    # Note: bare "by X" is intentionally excluded — it's too noisy.
    # "by the user" and "approved by X" are not grouping signals.
    # Aggregation phrases ("count X by Y") use "by" but those queries
    # also use "per"/"for each" most of the time; missing edge cases
    # is cheaper than false positives.
    _NOT_GROUPING = {"hour", "day", "week", "year", "the", "a", "an"}
    for m in grouping_matches:
        first = m.strip().split()[0] if m.strip() else ""
        if first and first not in _NOT_GROUPING:
            req.grouping_signals.append(m.strip())

    # ─── output columns ──────────────────────────────────────────────
    # "show X, Y, and Z" / "display X, Y, Z" / "for each X, show A, B, C"
    # The list ends at the first "where"/"with"/"for"/"that"/"by" clause
    # or at sentence-end punctuation.
    out_cols = _extract_output_columns(nl_lower)
    req.output_columns = out_cols

    # ─── filter constraints ──────────────────────────────────────────
    req.filter_constraints = _extract_filter_constraints(nl, nl_lower)

    # ─── quantifier constraints ──────────────────────────────────────
    # "more than N", "less than N", "exactly N", "at least N", "at most N"
    for op_word, op_tag in (
        ("more than",   "gt"),
        ("greater than","gt"),
        ("less than",   "lt"),
        ("fewer than",  "lt"),
        ("at least",    "gte"),
        ("at most",     "lte"),
        ("exactly",     "eq"),
    ):
        m = re.search(rf"\b{op_word}\s+(?:one|1|2|3|4|5|6|7|8|9|\d+)\b", nl_lower)
        if m:
            num_word = m.group(0).split()[-1]
            n = {"one": 1, "1": 1, "2": 2, "3": 3, "4": 4,
                 "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}.get(num_word)
            if n is None:
                try:
                    n = int(num_word)
                except ValueError:
                    continue
            req.quantifier_constraints.append((op_tag, n))

    # ─── entity type hints ───────────────────────────────────────────
    # "X department" / "X program" / "X course" / "X campus"
    # These hints pair an identifier (e.g. "CSE") with an entity-type
    # word (e.g. "DEPARTMENT").  If the SQL uses a different unit_type
    # for that identifier, the audit can flag it (covers Q189).
    for type_word, canonical in (
        ("department", "DEPARTMENT"),
        ("program",    "PROGRAM"),
        ("course",     "COURSE"),
        ("campus",     "CAMPUS"),
    ):
        # Capture an UPPER-CASE code or capitalised word IMMEDIATELY
        # before the type word.  e.g. "CSE department" → ("CSE","DEPARTMENT")
        # Skip lowercase generic mentions like "the department".
        for m in re.finditer(
            rf"\b([A-Z][A-Z0-9]{{1,9}}|[A-Z][a-z]+)\s+{type_word}\b",
            nl,
        ):
            ident = m.group(1)
            if ident.lower() not in ("the", "a", "an", "this", "that"):
                req.entity_type_hints.append((ident, canonical))

    return req


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────

# Regex precompiles
_OUTPUT_HEADERS = re.compile(
    r"\b(?:show|list|display|return|give\s+(?:me\s+)?the|including)\b",
    re.IGNORECASE,
)
_OUTPUT_TERMINATORS = re.compile(
    r"\b(where|with|when|filtering|filter|that\s+have|having|"
    r"order(?:ed)?\s+by|grouped\s+by|sorted|broken\s+down)\b|[?.]",
    re.IGNORECASE,
)


def _extract_output_columns(nl_lower: str) -> list[str]:
    """
    Extract the comma-separated output-list following 'show'/'list'/
    'display'/'including'/'give me the' headers.

    Returns lower-cased noun phrases.  Conservative: returns an empty
    list when the NL doesn't use one of the explicit header verbs.
    The L7 audit check only runs when this list is non-empty, so
    silently returning [] effectively disables the check for queries
    that don't have an explicit projection list.
    """
    m = _OUTPUT_HEADERS.search(nl_lower)
    if not m:
        return []

    tail = nl_lower[m.end():]
    term = _OUTPUT_TERMINATORS.search(tail)
    if term:
        tail = tail[:term.start()]

    # Split on commas; the "and" before the last item is removed.
    items = [s.strip() for s in tail.split(",")]
    cleaned: list[str] = []
    for it in items:
        it = re.sub(r"^\s*and\s+", "", it).strip(" .;:?!")
        if not it:
            continue
        # Drop the leading article/possessive
        words = [w for w in it.split() if w not in _OUTPUT_STOP]
        if words:
            cleaned.append(" ".join(words))
    # If the very first item still has the verb attached (because
    # there was no comma after it), drop it.
    if cleaned and len(cleaned[0].split()) > 6:
        cleaned[0] = " ".join(cleaned[0].split()[-4:])
    return cleaned


def _extract_filter_constraints(
    nl: str,
    nl_lower: str,
) -> list[FilterConstraint]:
    """
    Extract typed filter constraints from the NL.

    Each detected pattern emits a `FilterConstraint` with the SQL
    tokens we expect the SQL to contain.  Patterns are independent
    and additive — a single NL can produce many constraints.
    """
    out: list[FilterConstraint] = []

    # ─ TIME_RANGE ────────────────────────────────────────────────────
    # "in the last N days/weeks/months/hours/years"
    for m in re.finditer(
        r"\b(?:in\s+the\s+last|in\s+last|last|over\s+the\s+last|past)\s+"
        r"(\d+)\s+(hour|day|week|month|year)s?\b",
        nl_lower,
    ):
        n, unit = m.group(1), m.group(2)
        out.append(FilterConstraint(
            kind=ConstraintKind.TIME_RANGE,
            raw=m.group(0),
            tokens=[f"{unit}s", unit, "interval", n],
        ))

    # "this week / month / quarter / year / semester"
    for unit in ("week", "month", "quarter", "year", "semester"):
        if re.search(rf"\bthis\s+{unit}\b", nl_lower):
            tokens = [unit]
            if unit == "month":
                tokens.append("date_trunc")
            if unit == "quarter":
                tokens.append("date_trunc")
            if unit == "year":
                tokens.append("current_date")
                tokens.append("date_trunc")
            out.append(FilterConstraint(
                kind=ConstraintKind.TIME_RANGE,
                raw=f"this {unit}",
                tokens=tokens,
            ))

    # "in <Month> <year>" or "in <year>" or "for <year>"
    # The exact-month form ("in January 2026") becomes a strict token
    # check against the month name and year in the SQL.
    months = ("january","february","march","april","may","june",
             "july","august","september","october","november","december")
    for m in re.finditer(
        rf"\b(?:in|for|during)\s+({'|'.join(months)})\s+(\d{{4}})\b",
        nl_lower,
    ):
        month, year = m.group(1), m.group(2)
        out.append(FilterConstraint(
            kind=ConstraintKind.TIME_RANGE,
            raw=m.group(0),
            tokens=[year, month[:3], month, "date", "between"],
        ))

    # ─ STATUS / ENUM literals ────────────────────────────────────────
    # Catch enum words ONLY when they have a clear filter-signal context:
    #   (a) the upper-case form appears in the ORIGINAL NL (strong literal
    #       intent — Q166 "WITHHELD", Q179 "ABSENT", Q33 "ELIGIBLE")
    #   (b) the word appears in adjective position immediately before a
    #       domain noun ("approved requests", "active devices",
    #       "expired KEKs", "absent students")
    #   (c) the word follows an explicit qualifier — "with status X",
    #       "where status is X", "in status X", "X role"
    #
    # Bare occurrence as part of an action verb or possessive entity
    # reference ("the evaluator has not started", "granted by the
    # Custodian Admin") must NOT trigger — these are not filter values.
    #
    # Sort by length DESC so multi-word phrases ("hard deadline") win
    # over their substrings ("hard").
    _DOMAIN_NOUNS = (
        "requests?", "scripts?", "attempts?", "devices?", "policies",
        "policy", "holds?", "boards?", "bundles?", "coordinators?",
        "evaluators?", "configurations?", "papers?", "questions?",
        "answer keys?", "rubrics?", "marks", "results?", "notifications?",
        "transitions?", "logs?", "entries", "rules?", "users?",
        "students?", "keks?", "deks?", "sessions?", "extensions?",
        "annotations?", "audit log entries", "audit logs?",
        "data retention policies", "purge jobs?", "operations?",
        "applications?", "moderations?", "eligibility",
    )
    _DOMAIN_NOUN_RE = "|".join(_DOMAIN_NOUNS)

    for nl_word in sorted(_ENUM_VALUE_TOKENS, key=len, reverse=True):
        canonical = _ENUM_VALUE_TOKENS[nl_word]
        if any(canonical in c.tokens for c in out):
            continue  # already emitted by a longer overlapping phrase

        triggered = False

        # (a) Upper-case form in original NL — strongest signal.
        # We require >= 3 chars to avoid 'IS', 'ON' etc.
        if len(canonical) >= 3 and re.search(
            rf"\b{re.escape(canonical)}\b", nl
        ):
            triggered = True

        # (b) Adjective-position before a domain noun.
        if not triggered and re.search(
            rf"\b{re.escape(nl_word)}\s+(?:{_DOMAIN_NOUN_RE})\b",
            nl_lower,
        ):
            triggered = True

        # (c) Explicit qualifier ("with status X", "X status",
        #     "role of X", "for the X role", "in X scope")
        if not triggered:
            qual_patterns = (
                rf"\b(?:with|in|where|of)\s+(?:the\s+)?(?:status|state|role|scope|type)\s+"
                rf"(?:is\s+|of\s+)?{re.escape(nl_word)}\b",
                rf"\b{re.escape(nl_word)}\s+(?:status|state|role|scope|type)\b",
                rf"\bfor\s+(?:the\s+)?{re.escape(nl_word)}\s+role\b",
                rf"\bmarked\s+{re.escape(nl_word)}\b",  # Q147 "marked absent"
            )
            for qp in qual_patterns:
                if re.search(qp, nl_lower):
                    triggered = True
                    break

        # (d) Predicative position — "X have/has/are/is/were/was/been
        # (already|currently|still|recently)? <word>".  Catches
        # Q56 "KEKs have already expired", Q76 "devices that are active",
        # Q183 "currently active KEK", Q14 "answer key has been approved".
        if not triggered and re.search(
            rf"\b(?:have|has|are|is|were|was|been|currently|still|already|"
            rf"recently|that\s+(?:are|have|is))\s+(?:already\s+|currently\s+|"
            rf"still\s+|recently\s+|been\s+)?{re.escape(nl_word)}\b",
            nl_lower,
        ):
            triggered = True

        if triggered:
            out.append(FilterConstraint(
                kind=ConstraintKind.ENUM,
                raw=nl_word,
                tokens=[canonical],
            ))

    # ─ BOOLEAN constraints (is_X = TRUE / FALSE pattern) ─────────────
    # "active boards" / "inactive policies" / "is_active = TRUE"
    if re.search(r"\binactive\b", nl_lower):
        out.append(FilterConstraint(
            kind=ConstraintKind.BOOLEAN,
            raw="inactive",
            tokens=["is_active", "false"],
        ))
    if re.search(r"\bdecommissioned\b", nl_lower):
        out.append(FilterConstraint(
            kind=ConstraintKind.BOOLEAN,
            raw="decommissioned",
            tokens=["archived_at", "is_active"],
        ))

    # ─ NUMERIC thresholds expressed as token-bearing constraints ─────
    # "more than 24 hours" — the *unit* and *number* should both
    # appear in the SQL.  This is already partially covered by
    # TIME_RANGE for the 'hours' case; here we cover the bare numeric
    # threshold for things like "more than 100", "below 0.5".
    for m in re.finditer(
        r"\b(?:more than|greater than|less than|fewer than|below|above|"
        r"at least|at most|under|over)\s+(\d+(?:\.\d+)?)\b",
        nl_lower,
    ):
        out.append(FilterConstraint(
            kind=ConstraintKind.NUMERIC,
            raw=m.group(0),
            tokens=[m.group(1)],
        ))

    # ─ TEXT_LIKE ─────────────────────────────────────────────────────
    # "containing X" / "mentions X" / "with X in the (name|reason|...)"
    for m in re.finditer(
        r"\b(?:containing|mentions|mentioning|with the word|contain)\s+"
        r"(?:the\s+(?:word|text)\s+)?([\"']?)(\w[\w\s]*?)\1\s+(?:in\b|$|[.,?])",
        nl_lower,
    ):
        kw = m.group(2).strip()
        if kw:
            out.append(FilterConstraint(
                kind=ConstraintKind.TEXT_LIKE,
                raw=m.group(0),
                tokens=[kw, "like", "ilike"],
            ))

    return out