#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_results.py — Phase 2 A/B: RESULT-1 (base) vs RESULT-2 (fine-tuned).

Both inputs are batch_run.py --dry-run outputs over the SAME benchmark
(data/inputs/benchmark-test-set.jsonl). Only the model differs.

Usage:
    python compare_results.py RESULT-1_base.jsonl RESULT-2_finetuned.jsonl

Reports:
  - overall validation-pass % for each, and the delta
  - "true" pass (excludes Success-but-confidence-0 rows — passed validation but
    the pipeline flagged them worthless)
  - per-difficulty (High/Medium/Low) pass %
  - REGRESSIONS  (pass -> fail)  — the ones that matter most; any is a red flag
  - GAINS        (fail -> pass)
  - error-taxonomy shift (schema / semantic / cost / placeholder / ambiguous)

Scope / limitation:
    Compares VALIDATION-PASS outcomes only (batch_run.py --dry-run). It never
    executes SQL and never sees result sets, so two queries that both pass
    validation but would return different rows / columns / aliases are treated
    as identical. Use full execution (not --dry-run) for semantic comparison.

Exit code:
    0 = RESULT-2 matched or improved over RESULT-1
    1 = RESULT-2 regressed — either the overall validation pass rate decreased,
        or a benchmark question present in RESULT-1 is missing from RESULT-2
"""
from __future__ import annotations

import collections
import json
import sys


def _load(path):
    """Load a benchmark result JSONL into a dict keyed by QNum."""
    rows = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[r["QNum"]] = r
    return rows


def _passed(r):
    """Pipeline validation succeeded."""
    return r.get("Result") == "Success"


def _true_passed(r):
    """
    Successful validation with a usable query.

    Confidence == 0 means the pipeline intentionally rejected the generated SQL,
    even though validation technically succeeded.
    """
    return _passed(r) and _confidence(r) > 0.0


# FIX #9b: Defensive confidence parsing. A malformed row (empty string, "N/A",
# non-numeric) is treated as 0.0, never a crash. An interrupted batch_run can
# leave a partially-written JSONL line; this guard prevents the entire compare
# script from failing on one bad row.
def _confidence(r):
    try:
        return float(r.get("Query Confidence", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _bucket(msg):
    """
    Classify failures into coarse buckets for before/after comparison.
    """
    # FIX #9c: Coerce to str() defensively. A non-string Error Message (dict,
    # list, None) would crash on .lower(). batch_run normally emits strings, but
    # a malformed or interrupted row can break the contract.
    m = str(msg or "").lower()

    # Validation-stage tags take priority. Some validation errors contain
    # words like "ambiguous", which should not be mistaken for the router's
    # ambiguous_query outcome.
    if "(cost)" in m:
        return "cost"
    if "(schema)" in m:
        return "schema"
    if "(semantic)" in m:
        return "semantic"
    if "placeholder" in m:
        return "placeholder"
    if "ambiguous_query" in m or m.strip() == "ambiguous":
        return "ambiguous"

    return "other"


def _rate(rows, pred):
    """Return (passed, total, percentage) for a given predicate."""
    n = len(rows)
    p = sum(1 for r in rows.values() if pred(r))
    return p, n, (100.0 * p / n if n else 0.0)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python compare_results.py RESULT-1.jsonl RESULT-2.jsonl"
        )

    # RESULT-1 = baseline model
    a = _load(sys.argv[1])

    # RESULT-2 = candidate fine-tuned model
    b = _load(sys.argv[2])

    # FIX #8: Compute common rows AND disjoint sets from the ORIGINAL dicts
    # before restricting. A question that disappears from RESULT-2 (pipeline
    # hang, crash, renumbering) is a real regression — silently dropping it
    # would shrink RESULT-2's denominator and hide the regression.
    common = sorted(set(a) & set(b))
    only_in_a = sorted(set(a) - set(b))   # in base, vanished from candidate
    only_in_b = sorted(set(b) - set(a))   # new / renumbered in candidate

    # FIX #10: Early guard against completely disjoint inputs (empty common set).
    # Prevents ZeroDivisionError later and gives a clear diagnostic instead of
    # a cryptic traceback.
    if not common:
        raise SystemExit(
            "No common QNums between the two runs — nothing to compare "
            f"(base={len(a)}, ft={len(b)})."
        )

    if only_in_a or only_in_b:
        print(
            f"⚠  QNum sets differ (base={len(a)}, ft={len(b)}, "
            f"common={len(common)})."
        )
        if only_in_a:
            # A question that disappears from RESULT-2 is a real regression —
            # not a silently dropped row. Counted as regressions in exit logic.
            print(
                f"   {len(only_in_a)} in RESULT-1 but MISSING in RESULT-2 "
                f"(counted as regressions): "
                + ", ".join(f"Q{q}" for q in only_in_a)
            )
        if only_in_b:
            print(
                f"   {len(only_in_b)} only in RESULT-2 (new/renumbered): "
                + ", ".join(f"Q{q}" for q in only_in_b)
            )
        print()

    # FIX #8 (continued): Restrict ALL rate math to the shared question set so
    # base and candidate share one denominator. In the normal case (identical
    # QNum sets) this is a no-op — common == every row. When rows differ, this
    # prevents the denominator-skew bug where a smaller RESULT-2 denominator
    # artificially inflates its pass rate.
    a = {q: a[q] for q in common}
    b = {q: b[q] for q in common}

    # Overall validation success.
    pa, na, ra = _rate(a, _passed)
    pb, nb, rb = _rate(b, _passed)

    # Success excluding confidence-zero results.
    ta = sum(1 for r in a.values() if _true_passed(r))
    tb = sum(1 for r in b.values() if _true_passed(r))

    print("=" * 64)
    print("PHASE 2 A/B — RESULT-1 (base) vs RESULT-2 (fine-tuned)")
    print("=" * 64)

    print(
        f"  validation pass   base {pa}/{na} ({ra:.1f}%)"
        f"  ->  ft {pb}/{nb} ({rb:.1f}%)   Δ {rb-ra:+.1f} pp"
    )

    print(
        f"  true pass (conf>0) base {ta} ({100*ta/na:.1f}%)"
        f"  ->  ft {tb} ({100*tb/nb:.1f}%)"
        f"   Δ {100*(tb/nb-ta/na):+.1f} pp"
    )

    print("\n  by difficulty:")
    for t in ("High", "Medium", "Low"):
        # FIX #11: Bucket BOTH runs by RESULT-1's difficulty label over the
        # common row set. This ensures the same QNums are compared in each tier
        # even if a label differs between runs (malformed or changed row).
        # Previously, each run used its own label, so a QNum with "High" in A
        # and "Medium" in B would be counted in different buckets, making the
        # delta comparison meaningless.
        sa = {q: a[q] for q in common if a[q].get("type") == t}
        sb = {q: b[q] for q in common if a[q].get("type") == t}
        _, _, xa = _rate(sa, _passed)
        _, _, xb = _rate(sb, _passed)
        print(f"    {t:6s}: {xa:5.1f}%  ->  {xb:5.1f}%   Δ {xb-xa:+.1f}")

    # These deserve manual review before accepting a new model.
    regressions = [q for q in common if _passed(a[q]) and not _passed(b[q])]

    # Newly fixed benchmark questions.
    gains = [q for q in common if not _passed(a[q]) and _passed(b[q])]

    # FIX #8 (continued): Include vanished questions in the regression header
    # and the red-flag warning. A missing question is as serious as a pass->fail.
    print(
        f"\n  REGRESSIONS (pass -> fail): {len(regressions)}"
        + (f"  (+{len(only_in_a)} missing in RESULT-2)" if only_in_a else "")
        + (
            "   ⚠ investigate before deploying"
            if (regressions or only_in_a)
            else "   ✓ none"
        )
    )

    for q in regressions:
        print(f"    Q{q}: {b[q].get('Question','')[:88]}")
        print(
            f"         -> {(_bucket(b[q].get('Error Message'))).upper()}: "
            f"{b[q].get('Error Message','')[:110]}"
        )

    print(f"\n  GAINS (fail -> pass): {len(gains)}")
    for q in gains:
        print(f"    Q{q}: {b[q].get('Question','')[:88]}")

    print("\n  error taxonomy (failures):")

    ea = collections.Counter(
        _bucket(r.get("Error Message"))
        for r in a.values()
        if not _passed(r)
    )

    eb = collections.Counter(
        _bucket(r.get("Error Message"))
        for r in b.values()
        if not _passed(r)
    )

    for k in sorted(set(ea) | set(eb)):
        print(
            f"    {k:12s}  base {ea.get(k,0):3d}"
            f"  ->  ft {eb.get(k,0):3d}"
            f"   Δ {eb.get(k,0)-ea.get(k,0):+d}"
        )

    print("=" * 64)

    # FIX #8 (continued): a vanished benchmark question is a regression
    # regardless of the pass-rate delta, so the NET label reflects it too — this
    # keeps the printed verdict consistent with the exit code below (previously a
    # flat/up pass rate with missing rows printed "FLAT"/"IMPROVED" yet exited 1).
    if only_in_a:
        net = "REGRESSED (missing rows)"
    elif rb > ra:
        net = "IMPROVED"
    elif rb == ra:
        net = "FLAT"
    else:
        net = "REGRESSED"

    print(f"  NET: {net}  ({ra:.1f}% -> {rb:.1f}%)")
    print("=" * 64)

    # FIX #8 (continued): CI exit code. A drop in validation pass rate OR a
    # benchmark question that vanished from RESULT-2 both count as regression.
    # Previously, only rb < ra triggered exit 1; a flat pass rate with missing
    # rows would incorrectly return 0 (success).
    sys.exit(1 if (rb < ra or only_in_a) else 0)


if __name__ == "__main__":
    main()