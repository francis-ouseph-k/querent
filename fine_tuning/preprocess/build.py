# -*- coding: utf-8 -*-
"""
fine_tuning/preprocess/build.py
===============================
Dataset building: format → wrap → fit. Runs on the box that has the retriever
(Qdrant + OpenSearch + FK graph) and the Qwen tokenizer. Operates on in-memory
row lists; the orchestrator (pipeline.py) owns all file I/O and caching.

Three stages (consolidated from build_train_from_curated.py, wrap_outputs_json.py,
fit_context.py):

  format_pairs(pairs, retriever, qu)
      For each clean pair, retrieve the SAME schema context the model sees at
      serve time and emit the ChatML training record:
        input       = system rules            (_SYSTEM_PROMPT)
        instruction = schema context + question (user turn)
        output      = raw correct SQL          (assistant turn)
      Retrieval parity is the whole point — train on what you serve.

  wrap_rows(rows, ddl_path)
      Replace each raw-SQL `output` with the JSON serve-contract envelope
      {schema_reasoning, sql, tables_used, confidence, explanation}. Any row
      whose output isn't SQL is dropped here as a backstop (quality.gate should
      already have removed all such rows).

  fit_rows(rows, model_dir, max_seq)
      Token-budget each row to <= max_seq WITHOUT right-truncating the question:
      system + question-block + output are reserved; the schema-context HEAD is
      trimmed from its tail. Uses the real Qwen tokenizer for exact counts.

CRITICAL: format parity with generation/prompt_builder.py. _SYSTEM_PROMPT and
_STATUS_MODEL_BLOCK are imported from there so training and inference cannot drift.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _fmt_hms(seconds: float) -> str:
    """Format a duration like batch_run: '1h 44m 11s' / '7m 39s' / '45s'."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

# Section headers that indicate retrieval actually injected context (for EMPTY-CTX flagging)
_RETRIEVED_HEADERS = (
    "=== SCHEMA CONTEXT ===", "=== WORKFLOW AND STATUS SEMANTICS ===",
    "=== DOMAIN TERMINOLOGY ===", "=== RELEVANT JOIN PATHS ===",
    "=== ADDITIONAL CONTEXT ===", "=== EXAMPLE QUERIES ===",
)
QUESTION_MARK = "=== QUESTION ==="


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════
def init_retriever():
    """Stand up the live Phase-1 retriever (matches inference). Raises on failure."""
    import networkx as nx
    from ingestion.ddl_parser import DDLParser
    from retrieval.orchestrator import RetrievalOrchestrator
    from indexing.qdrant_indexer import QdrantIndexer
    from indexing.opensearch_indexer import OpenSearchIndexer
    from generation.query_understanding import QueryUnderstanding

    DDLParser().parse_file(Path(settings.ddl_path))
    graph_data = json.loads(Path("data/fk_graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(graph_data)
    retriever = RetrievalOrchestrator(
        qdrant_indexer=QdrantIndexer(),
        opensearch_indexer=OpenSearchIndexer(),
        fk_graph=graph,
    )
    qu = QueryUnderstanding(settings.glossary_path)
    return retriever, qu


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT  (schema context + ChatML record) — ported from data_pipeline
# ══════════════════════════════════════════════════════════════════════════════
def _build_schema_context(nl_query: str, retriever, query_understanding) -> str:
    """Mirror generation/prompt_builder section order EXACTLY (train/serve parity)."""
    try:
        from models.schema import ChunkType
        from generation.prompt_builder import _STATUS_MODEL_BLOCK

        entity_tables: list[str] = []
        intent_value = "unknown"
        if query_understanding is not None:
            parsed = query_understanding.process(nl_query)
            entity_tables = parsed.entities
            intent_value = parsed.intent.value

        chunks, retrieval_meta = retriever.retrieve(
            query_text=nl_query, entity_tables=entity_tables, intent=intent_value,
        )
        few_shots = retriever.get_few_shot_examples(query_text=nl_query, top_k=3)
        join_paths: list[str] = retrieval_meta.get("join_paths", [])

        lines: list[str] = []
        table_chunks = [c for c in chunks if c.chunk_type in (ChunkType.TABLE, ChunkType.VIEW)]
        if table_chunks:
            lines.append("=== SCHEMA CONTEXT ===")
            for c in table_chunks:
                lines.append(c.text); lines.append("")

        wf_chunks = [c for c in chunks if c.chunk_type in (ChunkType.WORKFLOW, ChunkType.STATUS)]
        if wf_chunks:
            lines.append("=== WORKFLOW AND STATUS SEMANTICS ===")
            for c in wf_chunks:
                lines.append(c.text); lines.append("")

        lines.append(_STATUS_MODEL_BLOCK)   # fixed block, injected on every serve prompt

        glossary_chunks = [c for c in chunks if c.chunk_type == ChunkType.GLOSSARY]
        if glossary_chunks:
            lines.append("=== DOMAIN TERMINOLOGY ===")
            for c in glossary_chunks:
                lines.append(c.text); lines.append("")

        if join_paths:
            lines.append("=== RELEVANT JOIN PATHS ===")
            lines.append("\n".join(join_paths)); lines.append("")

        fk_chunks = [c for c in chunks if c.chunk_type in
                     (ChunkType.FK_MAP, ChunkType.INDEX, ChunkType.AUDIT, ChunkType.PARTITION)]
        if fk_chunks:
            lines.append("=== ADDITIONAL CONTEXT ===")
            for c in fk_chunks:
                lines.append(c.text); lines.append("")

        if few_shots:
            lines.append("=== EXAMPLE QUERIES ===")
            for i, ex in enumerate(few_shots, 1):
                lines.append(f"Example {i}:")
                lines.append(f"Question: {ex.nl_question}")
                lines.append(f"SQL: {ex.expected_sql}")
                lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning(component="preprocess.build", event="schema_retrieval_failed",
                       nl_query=nl_query[:80], error=str(exc))
        return ""


def format_pairs(pairs: list[dict[str, Any]], retriever, query_understanding) -> tuple[list[dict], list[int]]:
    """Enrich each pair with retrieved schema context → ChatML record. Returns (rows, empty_ctx_lines)."""
    # Train on the SHORT system prompt, not the full serve-time rulebook: the full
    # prompt overflows the training window and trains a dependence on the rulebook
    # fine-tuning should internalise. See prompt_builder._TRAIN_SYSTEM_PROMPT.
    from generation.prompt_builder import _TRAIN_SYSTEM_PROMPT as _SYSTEM_PROMPT

    rows: list[dict[str, Any]] = []
    empty_ctx: list[int] = []
    total      = len(pairs)
    loop_start = time.perf_counter()
    # Retrieval (Qdrant + OpenSearch + FK graph) runs once PER PAIR here — this is
    # the expensive stage of a --force rebuild. Trace like batch_run so a 1–2h run
    # shows Q#, %, elapsed and ETA instead of sitting silent.
    print(f"[preprocess] formatting {total} pairs (live retrieval per pair)…", flush=True)
    for i, p in enumerate(pairs):
        schema_context = (_build_schema_context(p["nl_query"], retriever, query_understanding)
                          if retriever is not None else "")
        user_parts: list[str] = []
        if schema_context:
            user_parts.append(schema_context)
        user_parts += [QUESTION_MARK, p["nl_query"], "",
                       "Respond with ONLY the JSON object as specified above:"]
        rec = {
            "instruction": "\n".join(user_parts),
            "input":       _SYSTEM_PROMPT,
            "output":      p["sql"],
            "reasoning":   p.get("reasoning", ""),
            "source":      p.get("source", "curated"),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }
        no_ctx = retriever is not None and not any(h in rec["instruction"] for h in _RETRIEVED_HEADERS)
        if no_ctx:
            empty_ctx.append(p.get("line_no", 0))
        rows.append(rec)

        # ── Progress line (batch_run-style: Q#, %, elapsed, ETA) ──────────────
        done    = i + 1
        elapsed = time.perf_counter() - loop_start
        eta     = (elapsed / done) * (total - done)
        pct     = 100.0 * done / total
        print(
            f"  Q{done:>3}/{total} ({pct:4.1f}%) | "
            f"Elapsed: {_fmt_hms(elapsed)} | ETA: {_fmt_hms(eta)} | "
            f"empty_ctx {len(empty_ctx)} | "
            f"{p['nl_query'][:60]}",
            flush=True,
        )
    logger.info(component="preprocess.build", event="format_complete",
                rows=len(rows), empty_context=len(empty_ctx))
    return rows, empty_ctx


# ══════════════════════════════════════════════════════════════════════════════
# WRAP  (JSON serve envelope) — ported from wrap_outputs_json
# ══════════════════════════════════════════════════════════════════════════════
GOLD_CONFIDENCE = 0.9


def _ddl_tables(ddl_path: Path) -> set[str]:
    txt = ddl_path.read_text(encoding="utf-8", errors="ignore").lower()
    return set(re.findall(r"create table (?:if not exists )?([a-z_][a-z0-9_]*)", txt))


def _tables_used(sql: str, real_tables: set[str]) -> list[str]:
    cand = re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_]*)", sql, re.I)
    seen: list[str] = []
    for t in cand:
        tl = t.lower()
        if tl in real_tables and tl not in seen:
            seen.append(tl)
    return seen


def _synth_reasoning(sql: str, tables: list[str]) -> str:
    if tables:
        return f"Query targets {', '.join(tables)}; joins/filters derived from the schema context above."
    return "Single-source query derived from the schema context above."


def _is_sql(text: str) -> bool:
    return text.strip().lower().startswith(("select", "with"))


def wrap_rows(rows: list[dict[str, Any]], ddl_path: Path) -> tuple[list[dict], int]:
    """Replace raw-SQL output with the JSON serve envelope. Returns (rows, dropped)."""
    real = _ddl_tables(ddl_path)
    if not real:
        raise RuntimeError(f"no tables parsed from DDL {ddl_path}")
    out_rows, dropped = [], 0
    for row in rows:
        raw = str(row.get("output", "")).strip()
        if not _is_sql(raw):
            dropped += 1
            continue
        tables = _tables_used(raw, real)
        reasoning = str(row.get("reasoning", "")).strip()
        envelope = {
            "schema_reasoning": reasoning or _synth_reasoning(raw, tables),
            "sql":              raw,
            "tables_used":      tables,
            "confidence":       GOLD_CONFIDENCE,
            "explanation":      reasoning or "Returns the rows described by the question.",
        }
        out_rows.append(dict(row, output=json.dumps(envelope, ensure_ascii=False)))
    logger.info(component="preprocess.build", event="wrap_complete",
                kept=len(out_rows), dropped=dropped)
    return out_rows, dropped


# ══════════════════════════════════════════════════════════════════════════════
# FIT  (token budget) — ported from fit_context
# ══════════════════════════════════════════════════════════════════════════════
TEMPLATE_OVERHEAD = 24


def _count(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def fit_row(tok, row: dict, max_seq: int) -> tuple[dict, bool]:
    instr = row["instruction"]; system = row.get("input", ""); output = row.get("output", "")
    idx = instr.find(QUESTION_MARK)
    if idx == -1:
        return row, False
    head, tail = instr[:idx], instr[idx:]
    reserve = _count(tok, system) + _count(tok, tail) + _count(tok, output) + TEMPLATE_OVERHEAD
    head_budget = max_seq - reserve
    if head_budget <= 0:
        row["instruction"] = tail
        return row, True
    head_ids = tok.encode(head, add_special_tokens=False)
    if len(head_ids) <= head_budget:
        return row, False
    kept = tok.decode(head_ids[:head_budget])
    nl = kept.rfind("\n")
    if nl > 0:
        kept = kept[: nl + 1]
    row["instruction"] = kept + tail
    return row, True


def fit_rows(rows: list[dict[str, Any]], model_dir: str, max_seq: int) -> tuple[list[dict], int, int]:
    """Trim schema head so each row fits max_seq. Returns (rows, trimmed, still_over)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    out, trimmed, over = [], 0, 0
    for row in rows:
        row, was = fit_row(tok, row, max_seq)
        if was:
            trimmed += 1
        full = (_count(tok, row.get("input", "")) + _count(tok, row["instruction"])
                + _count(tok, row.get("output", "")) + TEMPLATE_OVERHEAD)
        if full > max_seq:
            over += 1
        out.append(row)
    logger.info(component="preprocess.build", event="fit_complete",
                rows=len(out), trimmed=trimmed, still_over=over, max_seq=max_seq)
    return out, trimmed, over