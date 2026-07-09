# -*- coding: utf-8 -*-
"""
fine_tuning/preprocess/sources.py
=================================
Source adapters. Every adapter returns the SAME "pair" shape so the rest of the
pipeline (quality → build) is source-agnostic:

    {
      "nl_query":  <the natural-language question>,
      "sql":       <the correct SQL — RAW, not yet JSON-wrapped>,
      "reasoning": <optional chain-of-thought / explanation>,
      "source":    <provenance tag>,
      "category":  <optional curated category, passed through for stats>,
      "line_no":   <1-based line in the source file, for audit>,
    }

Adapters
--------
  load_curated()   — the golden hand-authored corpus
                     (data/training/nl-sql_traning-dataset.jsonl):
                     instruction/schema_context/reasoning/output
  load_failures()  — Phase-1 failure logs with operator :correct SQL
  load_few_shots() — the curated few-shot pool (also used for retrieval)
  load_synthetic() — generate_synthetic.py output (bootstrap only)

Consolidated from the former data_pipeline.py loaders + build_train_from_curated
._load_curated so there is ONE place that knows each on-disk format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── Curated golden corpus ─────────────────────────────────────────────────────
def load_curated(
    path: Path,
    question_field: str = "instruction",
    sql_field: str = "output",
) -> list[dict[str, Any]]:
    """
    Load the curated corpus into pair shape. The curated file is
    instruction/schema_context/reasoning/output; we do NOT keep schema_context
    here because build.format() re-derives it from live retrieval so training
    context matches serve exactly.
    """
    pairs: list[dict[str, Any]] = []
    for i, row in enumerate(_read_jsonl(path), start=1):
        nl  = str(row.get(question_field, "")).strip()
        sql = str(row.get(sql_field, "")).strip()
        pairs.append({
            "nl_query":  nl,
            "sql":       sql,
            "reasoning": str(row.get("reasoning", "")),
            "source":    "curated",
            "category":  row.get("category", ""),
            "line_no":   i,
        })
    logger.info(component="preprocess.sources", event="curated_loaded",
                path=str(path), rows=len(pairs))
    return pairs


# ── Phase-1 failure logs (failures/*.json) ────────────────────────────────────
def load_failures(failure_dir: Path) -> list[dict[str, Any]]:
    """Only entries with a non-empty corrected_sql (operator :correct) are usable."""
    pairs: list[dict[str, Any]] = []
    skipped = 0
    if not failure_dir.exists():
        return pairs
    for path in sorted(failure_dir.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(component="preprocess.sources",
                           event="failure_file_unreadable", path=str(path), error=str(exc))
            continue
        corrected = (entry.get("corrected_sql") or "").strip()
        if not corrected:
            skipped += 1
            continue
        pairs.append({
            "nl_query":  str(entry.get("nl_query", "")).strip(),
            "sql":       corrected,
            "reasoning": entry.get("reasoning", ""),
            "source":    "failure_log",
            "category":  "",
            "line_no":   0,
        })
    logger.info(component="preprocess.sources", event="failures_loaded",
                usable=len(pairs), skipped_no_correction=skipped)
    return pairs


# ── Curated few-shot pool ─────────────────────────────────────────────────────
def load_few_shots(few_shot_path: Path) -> list[dict[str, Any]]:
    if not few_shot_path.exists():
        logger.warning(component="preprocess.sources", event="few_shot_missing",
                       path=str(few_shot_path))
        return []
    try:
        raw = json.loads(few_shot_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(component="preprocess.sources", event="few_shot_load_failed", error=str(exc))
        return []
    pairs = []
    for item in raw:
        nl  = (item.get("nl") or item.get("nl_question") or item.get("nl_query") or "").strip()
        sql = (item.get("expected_sql") or item.get("sql") or "").strip()
        if nl and sql:
            pairs.append({"nl_query": nl, "sql": sql,
                          "reasoning": item.get("reasoning", ""),
                          "source": "few_shot_curated", "category": "", "line_no": 0})
    logger.info(component="preprocess.sources", event="few_shots_loaded", count=len(pairs))
    return pairs


# ── Synthetic (bootstrap only) ────────────────────────────────────────────────
def load_synthetic(synthetic_path: Path) -> list[dict[str, Any]]:
    if not synthetic_path.exists():
        return []
    pairs = []
    for line in synthetic_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        pairs.append({"nl_query": item.get("nl_query", "").strip(),
                      "sql": item.get("sql", "").strip(),
                      "reasoning": item.get("reasoning", ""),
                      "source": "synthetic", "category": "", "line_no": 0})
    logger.info(component="preprocess.sources", event="synthetic_loaded", count=len(pairs))
    return pairs


def load_benchmark_questions(benchmark_path: Path, field: str = "Question") -> list[str]:
    """The NL questions the fine-tune must never see (leak set)."""
    if not benchmark_path.exists():
        return []
    return [str(b.get(field, "")).strip()
            for b in _read_jsonl(benchmark_path) if str(b.get(field, "")).strip()]