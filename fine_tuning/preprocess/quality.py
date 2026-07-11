# -*- coding: utf-8 -*-
"""
fine_tuning/preprocess/quality.py
=================================
All validation and quality gates, in one place. Operates on the common pair
shape (see sources.py): {nl_query, sql, reasoning, source, category, line_no}.

Two gates:

  deleak(pairs, benchmark_questions)
      Train/test hygiene. Drops any training row whose QUESTION matches a
      benchmark question (exact after normalisation, or token-Jaccard >= thr).
      Without this, RESULT-2 is inflated by memorisation and the base-vs-tune
      delta is meaningless.

  gate(pairs)
      Correctness gate on the SQL. In order:
        1. drop non-SQL / prose rows (no SELECT|WITH)
        2. drop "garbage join" rows  (SELECT a.*, b.*  →  a.id = b.id cartesian
           templates that teach broken joins)
        3. PLACEHOLDER POLICY (approved): strip any boolean predicate that
           references a :placeholder IF the remaining SQL is still a valid
           SELECT (a vanished WHERE is fine); otherwise REJECT the row. Uses an
           AST (sqlglot) to decide + remove, then re-parses to validate. Only
           the stripped rows are reserialised; untouched rows pass through
           byte-identical.
        4. drop empty / too-short / too-long / not-SELECT rows

      Returns (kept_pairs, reject_counts_by_reason).

Consolidated from data_pipeline._quality_filter + deleak_train.deleak +
scrub/curate prototypes. This is the SINGLE gate both the curated and failures
paths flow through, so the bug where the curated path skipped quality filtering
cannot recur.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import sqlglot
from sqlglot import expressions as exp

from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── thresholds (ported from data_pipeline._quality_filter) ────────────────────
_MIN_NL_WORDS   = 4
_MIN_SQL_TOKENS = 10
_MAX_SQL_TOKENS = 800

_DIALECT = "postgres"
# "SELECT a.*, b.*" style templates — every occurrence in the corpus is a broken
# id=id cartesian example. These teach the join/hallucination errors seen at serve.
_GARBAGE_JOIN = re.compile(r"SELECT\s+\w+\.\*\s*,\s*\w+\.\*", re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
# DE-LEAK
# ══════════════════════════════════════════════════════════════════════════════
_NON_ALNUM  = re.compile(r"[^a-z0-9 ]+")
_MULTISPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _MULTISPACE.sub(" ", _NON_ALNUM.sub(" ", text.lower())).strip()


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deleak(
    pairs: list[dict[str, Any]],
    benchmark_questions: list[str],
    jaccard_threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove rows whose question matches a benchmark question. Returns (kept, removed)."""
    bench_norm: set[str] = set()
    bench_tok: list[frozenset[str]] = []
    for q in benchmark_questions:
        nq = _normalise(q)
        if nq:
            bench_norm.add(nq)
            bench_tok.append(frozenset(nq.split()))

    kept, removed = [], []
    for row in pairs:
        nq = _normalise(row.get("nl_query", ""))
        tq = frozenset(nq.split())
        matched = None
        if nq and nq in bench_norm:
            matched = "exact"
        elif jaccard_threshold < 1.0:
            best = max((_jaccard(tq, tb) for tb in bench_tok), default=0.0)
            if best >= jaccard_threshold:
                matched = f"jaccard={best:.3f}"
        if matched is None:
            kept.append(row)
        else:
            removed.append({"line_no": row.get("line_no"),
                            "question": row.get("nl_query", "")[:140],
                            "matched_by": matched})
    logger.info(component="preprocess.quality", event="deleak_complete",
                rows_in=len(pairs), kept=len(kept), removed=len(removed))
    return kept, removed


# ══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER SUBSTITUTION  (preferred) → STRIP (fallback) → REJECT
# ══════════════════════════════════════════════════════════════════════════════
# WHY SUBSTITUTE FIRST (FIX-F3): the old strip-only policy removed the WHERE
# predicate but KEPT the question that demands it ("…for board 12"). 153/378
# rows (40% of the v4 corpus) therefore taught "question names an entity →
# answer omits the filter". Post-FT symptoms observed in batch runs: dropped
# filters/grain (semantic 'per'/'for each' failures) and the model falling back
# to :qp_id/:board_id bind variables it was never allowed to resolve — a
# failure class the BASE model never produced. Substituting a plausible typed
# literal preserves the question↔filter association and matches serve Rule 3
# ("use literal values, never placeholders").
#
# Substitution is deliberately conservative — only when the literal's TYPE is
# unambiguous from the placeholder/column name:
#   *_id / id / *_no / limit / offset / count / marks / year → integer
#   *date* / *_at / *_on                                     → ISO date string
# Anything else (status enums, names, codes) falls through to the old strip
# logic: guessing 'FROZEN' vs 'SUBMITTED' wrong would teach invalid enum
# values, which is worse than a vanished filter.
_STOP = (exp.And, exp.Or, exp.Where, exp.Join, exp.Having, exp.Select)

_INT_HINT  = re.compile(r"(?:^|_)(id|no|num|limit|offset|count|marks|year|attempt)s?$", re.I)
_DATE_HINT = re.compile(r"date|_at$|_on$", re.I)

_SUBST_INT  = "42"
_SUBST_DATE = "'2026-01-01'"


def _placeholder_name(ph: exp.Placeholder) -> str:
    return str(ph.this or "")


def _context_column_name(ph: exp.Placeholder) -> str:
    """Name of the column the placeholder is compared against, if findable."""
    node = ph.parent
    for _ in range(4):                      # EQ/GT/Between/In are shallow
        if node is None:
            return ""
        col = node.find(exp.Column)
        if col is not None:
            return col.name or ""
        node = node.parent
    return ""


def _literal_for(ph: exp.Placeholder) -> str | None:
    """Typed literal SQL text, or None when the type cannot be inferred safely."""
    hint = _placeholder_name(ph) or _context_column_name(ph)
    if not hint:
        return None
    if _INT_HINT.search(hint):
        return _SUBST_INT
    if _DATE_HINT.search(hint):
        return _SUBST_DATE
    return None


def substitute_placeholders(sql: str) -> tuple[str, str]:
    """
    Return (sql, status). status ∈ {'clean', 'substituted', 'partial', 'reject:<why>'}.

    Replaces each :placeholder whose type is inferable with a typed literal.
    'partial' means some placeholders were substituted but others remain —
    caller should then run strip_placeholders() on the result.
    Untouched ('clean') rows are returned verbatim — no reserialisation.
    """
    if ":" not in sql:
        return sql, "clean"
    try:
        tree = sqlglot.parse_one(sql, read=_DIALECT)
    except Exception:
        return sql, "reject:parse_fail"
    if tree.find(exp.Placeholder) is None:
        return sql, "clean"                 # the ':' was a ::cast or a string literal

    replaced, remaining = 0, 0
    for ph in list(tree.find_all(exp.Placeholder)):
        lit = _literal_for(ph)
        if lit is None:
            remaining += 1
            continue
        ph.replace(sqlglot.parse_one(lit, read=_DIALECT))
        replaced += 1

    if replaced == 0:
        return sql, "partial"               # nothing inferable — strip fallback
    try:
        out = tree.sql(dialect=_DIALECT)
        sqlglot.parse_one(out, read=_DIALECT)
    except Exception:
        return sql, "reject:invalid_after_substitute"
    return out, ("substituted" if remaining == 0 else "partial")


# ── reasoning scrub (FIX-F3b) ─────────────────────────────────────────────────
# wrap_rows copies the curated `reasoning` field verbatim into the
# schema_reasoning/explanation LABELS. Three v4 rows discussed ':qp_id bind
# variables' in prose — training the exact token the serve validator rejects.
_BIND_MENTION = re.compile(r"(?<![:\w]):[a-zA-Z_][a-zA-Z0-9_]*\b")


def scrub_reasoning(text: str) -> str:
    """Remove :bind-variable mentions from label prose. Idempotent."""
    if not text or ":" not in text:
        return text
    return _BIND_MENTION.sub("a literal value", text)


def strip_placeholders(sql: str) -> tuple[str, str]:
    """
    Return (sql, status). status ∈ {'clean', 'stripped', 'reject:<why>'}.

    Removes each boolean predicate that references a :placeholder:
      - inside AND/OR  → replace the connector with its other operand
      - the whole WHERE → drop the WHERE clause  (allowed: filter vanishes)
      - the whole HAVING → drop the HAVING clause
      - sole JOIN ON condition → REJECT (removing it makes a cartesian join)
      - anywhere else (SELECT projection, LIMIT, function arg) → REJECT
    Then re-parses; if the result is not a valid SELECT with a FROM, REJECT.
    Untouched ('clean') rows are returned verbatim — no reserialisation.
    """
    if ":" not in sql:
        return sql, "clean"
    try:
        tree = sqlglot.parse_one(sql, read=_DIALECT)
    except Exception:
        return sql, "reject:parse_fail"
    if tree.find(exp.Placeholder) is None:
        return sql, "clean"            # the ':' was a ::cast or a string literal

    for _ in range(40):                # bounded: one predicate removed per pass
        ph = tree.find(exp.Placeholder)
        if ph is None:
            break
        pred = ph
        while pred.parent is not None and not isinstance(pred.parent, _STOP):
            pred = pred.parent
        parent = pred.parent
        if parent is None:
            return sql, "reject:no_parent"
        if isinstance(parent, (exp.And, exp.Or)):
            other = parent.right if parent.left is pred else parent.left
            parent.replace(other)
        elif isinstance(parent, exp.Where):
            parent.pop()
        elif isinstance(parent, exp.Having):
            parent.pop()
        elif isinstance(parent, exp.Join):
            return sql, "reject:sole_join_condition"
        else:                          # projection / limit / other → not a filter
            return sql, "reject:unremovable_position"
    else:
        return sql, "reject:too_many_placeholders"

    if tree.find(exp.Placeholder) is not None:
        return sql, "reject:residual_placeholder"
    try:
        out = tree.sql(dialect=_DIALECT)
        rt = sqlglot.parse_one(out, read=_DIALECT)
    except Exception:
        return sql, "reject:invalid_after_strip"
    sel = rt if isinstance(rt, exp.Select) else rt.find(exp.Select)
    if sel is None:
        return sql, "reject:not_select_after"
    if sel.find(exp.From) is None:
        return sql, "reject:no_from_after"
    return out, "stripped"


# ══════════════════════════════════════════════════════════════════════════════
# GATE
# ══════════════════════════════════════════════════════════════════════════════
def _is_sql(text: str) -> bool:
    return text.strip().lower().startswith(("select", "with"))


# FIX-F4: prompt-style whitelist. 118/378 rows of the v4 corpus (31%) were
# meta-prompts ("Fix this query…" with no query embedded) or conceptual
# categories whose question style never occurs at serve time. They dilute an
# already tiny corpus and teach answer shapes ("Two possible intents…") that
# break the single-JSON output contract. Only serve-shaped questions train.
DEFAULT_TRAIN_CATEGORIES: frozenset[str] = frozenset({
    "Business Question → SQL",
    "",                       # failure-log / few-shot / synthetic rows carry no category
})


def gate(
    pairs: list[dict[str, Any]],
    train_categories: frozenset[str] | None = DEFAULT_TRAIN_CATEGORIES,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Correctness gate. Returns (kept_pairs, reject_counts_by_reason)."""
    kept: list[dict[str, Any]] = []
    rej: Counter = Counter()
    n_stripped = 0
    n_substituted = 0

    for p in pairs:
        nl  = str(p.get("nl_query", "")).strip()
        sql = str(p.get("sql", "")).strip()

        # 0. FIX-F4: off-task prompt styles never reach the corpus
        if train_categories is not None and p.get("category", "") not in train_categories:
            rej["off_task_category"] += 1; continue
        # 1. prose / non-SQL
        if not sql:
            rej["empty_sql"] += 1; continue
        if not _is_sql(sql):
            rej["prose_no_sql"] += 1; continue
        # 2. garbage joins — BEFORE strip, so they're dropped not reserialised
        if _GARBAGE_JOIN.search(sql):
            rej["garbage_join"] += 1; continue
        # 3. placeholder policy — FIX-F3: substitute typed literal first;
        #    strip is the fallback only for uninferable placeholder types.
        new_sql, status = substitute_placeholders(sql)
        if status.startswith("reject"):
            rej[status] += 1; continue
        if status in ("substituted", "partial"):
            sql = new_sql
            if status == "substituted":
                n_substituted += 1
        if status == "partial":
            new_sql, status = strip_placeholders(sql)
            if status.startswith("reject"):
                rej[status] += 1; continue
            if status == "stripped":
                sql = new_sql
                n_stripped += 1
        # 4. length / question quality (on the possibly-stripped SQL)
        if not nl:
            rej["empty_nl"] += 1; continue
        if len(nl.split()) < _MIN_NL_WORDS:
            rej["short_nl"] += 1; continue
        toks = sql.split()
        if len(toks) < _MIN_SQL_TOKENS:
            rej["short_sql"] += 1; continue
        if len(toks) > _MAX_SQL_TOKENS:
            rej["long_sql"] += 1; continue
        if not sql.lower().startswith("select"):     # CTEs allowed in _is_sql but final SQL must resolve to SELECT
            # WITH ... SELECT is fine; only reject if truly not a query
            if not sql.lower().startswith("with"):
                rej["not_select"] += 1; continue

        p = dict(p, sql=sql)
        kept.append(p)

    rej["_stripped"]    = n_stripped      # informational, not a rejection
    rej["_substituted"] = n_substituted   # informational, not a rejection
    logger.info(component="preprocess.quality", event="gate_complete",
                rows_in=len(pairs), kept=len(kept),
                stripped=n_stripped, substituted=n_substituted, rejected=dict(rej))
    return kept, dict(rej)