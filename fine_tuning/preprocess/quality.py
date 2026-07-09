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
# PLACEHOLDER STRIP  (AST-based; strip-if-valid-else-reject)
# ══════════════════════════════════════════════════════════════════════════════
_STOP = (exp.And, exp.Or, exp.Where, exp.Join, exp.Having, exp.Select)


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


def gate(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Correctness gate. Returns (kept_pairs, reject_counts_by_reason)."""
    kept: list[dict[str, Any]] = []
    rej: Counter = Counter()
    n_stripped = 0

    for p in pairs:
        nl  = str(p.get("nl_query", "")).strip()
        sql = str(p.get("sql", "")).strip()

        # 1. prose / non-SQL
        if not sql:
            rej["empty_sql"] += 1; continue
        if not _is_sql(sql):
            rej["prose_no_sql"] += 1; continue
        # 2. garbage joins — BEFORE strip, so they're dropped not reserialised
        if _GARBAGE_JOIN.search(sql):
            rej["garbage_join"] += 1; continue
        # 3. placeholder policy
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

    rej["_stripped"] = n_stripped     # informational, not a rejection
    logger.info(component="preprocess.quality", event="gate_complete",
                rows_in=len(pairs), kept=len(kept),
                stripped=n_stripped, rejected=dict(rej))
    return kept, dict(rej)