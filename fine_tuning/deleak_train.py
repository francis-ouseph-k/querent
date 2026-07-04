#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tuning/deleak_train.py
===========================

Strip benchmark-test questions out of the CURATED TRAINING corpus so that the
Phase-2 fine-tune never sees a question that Phase-1's benchmark will score.

Workflow this supports
----------------------
  1. batch_run.py on benchmark-test-set.jsonl with the BASE model      -> RESULT-1
  2. fine-tune on the (cleaned) training corpus ONLY
  3. batch_run.py on the SAME benchmark with the FINE-TUNED model       -> RESULT-2
  4. compare RESULT-1 vs RESULT-2

Step 2's corpus MUST NOT contain any benchmark question, or RESULT-2 is inflated
by memorisation and the RESULT-1 vs RESULT-2 delta is not real capability.

What this does
--------------
  - Loads the curated training corpus and the benchmark question set.
  - Removes every training row whose question matches a benchmark question:
        exact (after lower-case + punctuation normalisation), OR
        token-Jaccard >= --jaccard (default 0.85) near-duplicate.
  - Writes the cleaned corpus in the SAME format as the input (format-preserving —
    this stays your golden curated file, just leak-free).
  - Writes a report listing exactly which rows were removed and why.

What this deliberately does NOT do
----------------------------------
  - No train/val/test split. The trainer's own 10% dev split (for eval_loss) is a
    separate, in-training concern and is left untouched.
  - No gold-SQL harvesting / benchmark_gold emission. That was for evaluator.py's
    result-set metric, which this workflow does not use.
  - No reformatting to trainer's instruction/input/output schema. See NOTE below.

NOTE — trainer input format
---------------------------
trainer.py reads instruction/input/output records (built by data_pipeline.py).
The curated corpus uses instruction/schema_context/reasoning/output. The cleaned
file this script writes is still in the CURATED format. Convert it to trainer
format before training (data_pipeline formatting, or an equivalent formatter).
This script's single job is leak removal, nothing else.

Usage
-----
    python -m fine_tuning.deleak_train \
        --corpus     data/training/nl-sql_traning-dataset.jsonl \
        --benchmark  data/inputs/benchmark-test-set.jsonl \
        --out        data/training/nl-sql_traning-dataset.clean.jsonl

    # exact matches only (no near-dup removal):
    python -m fine_tuning.deleak_train --jaccard 1.0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# Near-duplicate threshold. Exact matches are ALWAYS removed. Rows whose
# token-Jaccard with any benchmark question is >= this are also removed.
# Set to 1.0 to remove only exact matches.
JACCARD_THRESHOLD = 0.85

_NON_ALNUM  = re.compile(r"[^a-z0-9 ]+")
_MULTISPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """lower-case, alphanumerics only, single-spaced — the matching canonical form."""
    return _MULTISPACE.sub(" ", _NON_ALNUM.sub(" ", text.lower())).strip()


def _tokens(norm_text: str) -> frozenset[str]:
    return frozenset(norm_text.split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    # Atomic write via .tmp + replace (matches data_pipeline.py's convention).
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def deleak(
    corpus_path:    Path,
    benchmark_path: Path,
    out_path:       Path,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    corpus_question_field:    str = "instruction",
    benchmark_question_field: str = "Question",
) -> dict[str, Any]:
    corpus    = _read_jsonl(corpus_path)
    benchmark = _read_jsonl(benchmark_path)

    # Benchmark questions: normalised set for exact match + token sets for near-dup.
    bench_norm: set[str] = set()
    bench_tok:  list[tuple[str, frozenset[str], Any]] = []
    for b in benchmark:
        q = str(b.get(benchmark_question_field, "")).strip()
        nq = _normalise(q)
        if nq:
            bench_norm.add(nq)
            bench_tok.append((q, _tokens(nq), b.get("QNum")))

    kept:    list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for row in corpus:
        q = str(row.get(corpus_question_field, "")).strip()
        nq = _normalise(q)
        tq = _tokens(nq)

        matched_by = None
        matched_qnum = None

        if nq in bench_norm:
            matched_by = "exact"
        else:
            best, bqnum = 0.0, None
            for _bq, tb, qnum in bench_tok:
                s = _jaccard(tq, tb)
                if s > best:
                    best, bqnum = s, qnum
            if best >= jaccard_threshold and jaccard_threshold < 1.0:
                matched_by = f"jaccard={best:.3f}"
                matched_qnum = bqnum

        if matched_by is None:
            kept.append(row)
        else:
            removed.append({"question": q[:140], "matched_by": matched_by,
                            "matched_qnum": matched_qnum})

    _write_jsonl(out_path, kept)

    report = {
        "corpus_path":       str(corpus_path),
        "benchmark_path":    str(benchmark_path),
        "jaccard_threshold": jaccard_threshold,
        "corpus_rows_in":    len(corpus),
        "rows_kept":         len(kept),
        "rows_removed":      len(removed),
        "benchmark_questions": len(benchmark),
        "clean_output":      str(out_path),
        "removed":           removed,   # full list — every removal is auditable
    }
    report_path = out_path.with_name("deleak_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _print_summary(r: dict[str, Any]) -> None:
    exact = sum(1 for x in r["removed"] if x["matched_by"] == "exact")
    near  = r["rows_removed"] - exact
    print("\n" + "=" * 62)
    print("DE-LEAK (training corpus vs benchmark)")
    print("=" * 62)
    print(f"  corpus rows in ...... {r['corpus_rows_in']}")
    print(f"  benchmark questions . {r['benchmark_questions']}")
    print(f"  removed ............. {r['rows_removed']}  (exact {exact}, near-dup {near})")
    print(f"  KEPT (train on) ..... {r['rows_kept']}")
    print("-" * 62)
    print(f"  clean  -> {r['clean_output']}")
    print(f"  report -> {r['report_path']}")
    print("=" * 62 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove benchmark questions from the training corpus.")
    ap.add_argument("--corpus", type=Path,
                    default=Path("data/training/nl-sql_traning-dataset.jsonl"))
    ap.add_argument("--benchmark", type=Path,
                    default=Path("data/inputs/benchmark-test-set.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/training/nl-sql_traning-dataset.clean.jsonl"))
    ap.add_argument("--jaccard", type=float, default=JACCARD_THRESHOLD,
                    help=f"Near-dup threshold (default {JACCARD_THRESHOLD}; 1.0 = exact only).")
    ap.add_argument("--corpus-field", default="instruction",
                    help="Field holding the NL question in the corpus (default: instruction).")
    ap.add_argument("--benchmark-field", default="Question",
                    help="Field holding the NL question in the benchmark (default: Question).")
    args = ap.parse_args()

    for p in (args.corpus, args.benchmark):
        if not p.exists():
            raise SystemExit(f"ERROR: input not found: {p}")

    report = deleak(args.corpus, args.benchmark, args.out, args.jaccard,
                    args.corpus_field, args.benchmark_field)
    _print_summary(report)


if __name__ == "__main__":
    main()