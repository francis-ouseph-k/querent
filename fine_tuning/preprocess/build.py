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
def _build_schema_context(
    nl_query: str,
    retriever,
    query_understanding,
    gold_tables: list[str] | None = None,
) -> str:
    """
    Render the user-turn context via the SHARED train-format renderer
    (generation/prompt_builder.render_train_user_prompt) — one function for
    training and FT serving, so drift is impossible.

    FIX-F2a (gold pinning): retrieval is seeded with the tables the GOLD SQL
    actually uses, not just query_understanding's entity guesses. The
    orchestrator promotes entity_tables to mandatory chunks, so pinning here
    guarantees the answer's tables are IN the context the model learns from.
    Pre-fix measurement: 23% of rows (106/452 pre-fit) were missing at least
    one gold table from context purely due to retrieval misses — every such
    row trains the model to write SQL over tables it cannot see.
    """
    try:
        from generation.prompt_builder import render_train_user_prompt

        entity_tables: list[str] = []
        intent_value = "unknown"
        if query_understanding is not None:
            parsed = query_understanding.process(nl_query)
            entity_tables = parsed.entities
            intent_value = parsed.intent.value

        # gold pinning — union, gold first so mandatory promotion favours them
        seed = list(dict.fromkeys([*(gold_tables or []), *entity_tables]))

        chunks, retrieval_meta = retriever.retrieve(
            query_text=nl_query, entity_tables=seed, intent=intent_value,
        )
        few_shots = retriever.get_few_shot_examples(query_text=nl_query, top_k=3)
        join_paths: list[str] = retrieval_meta.get("join_paths", [])

        # render WITHOUT the question — format_pairs appends the question block
        rendered = render_train_user_prompt(
            schema_chunks=chunks, join_paths=join_paths,
            few_shots=few_shots, question="",
        )
        # strip the trailing question scaffold the renderer adds
        qidx = rendered.find(QUESTION_MARK)
        return rendered[:qidx].rstrip("\n") if qidx != -1 else rendered
    except Exception as exc:
        logger.warning(component="preprocess.build", event="schema_retrieval_failed",
                       nl_query=nl_query[:80], error=str(exc))
        return ""


def format_pairs(
    pairs: list[dict[str, Any]],
    retriever,
    query_understanding,
    real_tables: set[str] | None = None,
) -> tuple[list[dict], list[int]]:
    """Enrich each pair with retrieved schema context → ChatML record. Returns (rows, empty_ctx_lines)."""
    # Train on the SHORT system prompt, not the full serve-time rulebook: the full
    # prompt overflows the training window and trains a dependence on the rulebook
    # fine-tuning should internalise. See prompt_builder._TRAIN_SYSTEM_PROMPT.
    # PARITY NOTE (FIX-F1): the fine-tuned model MUST be served this same shape —
    # set LLM_PROMPT_PROFILE=ft so runner uses PromptBuilder.build_ft().
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
        # FIX-F2a: tables the gold SQL uses — pinned into retrieval, and carried
        # on the row so fit_row can protect their chunks from trimming.
        gold_tables = _tables_used(p["sql"], real_tables) if real_tables else []
        schema_context = (_build_schema_context(p["nl_query"], retriever,
                                                query_understanding, gold_tables)
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
            "gold_tables": gold_tables,     # consumed (and removed) by fit_rows
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
# FIX-F5: a CONSTANT confidence label (was 0.9 on all rows) trains the model
# to always emit the same number, destroying the signal CONFIDENCE_WARN_THRESHOLD
# gates on — post-FT distribution collapsed to {0.9, 0.85}. A deterministic
# complexity heuristic restores variance while staying honest: gold SQL is
# correct by definition, so simpler queries get higher confidence.
GOLD_CONFIDENCE = 0.9   # kept for backward reference; _gold_confidence() supersedes


def _gold_confidence(sql: str) -> float:
    s = sql.lower()
    joins = s.count(" join ")
    if s.strip().startswith("with") or joins >= 3:
        return 0.82
    if joins >= 1:
        return 0.88
    return 0.92


def _ddl_tables(ddl_path: Path) -> set[str]:
    """All queryable relations: tables AND views.

    FIX-R6: the previous regex matched only `create table`, so gold SQL that
    selected from a view (v10.5 has v_user_auxiliary_role_resolved and
    v_key_encryption_key_rotation_candidates) was never pinned — those rows
    trained with the view ABSENT from context.
    """
    txt = ddl_path.read_text(encoding="utf-8", errors="ignore").lower()
    tables = set(re.findall(r"create table (?:if not exists )?([a-z_][a-z0-9_]*)", txt))
    views  = set(re.findall(r"create (?:or replace )?view ([a-z_][a-z0-9_]*)", txt))
    return tables | views


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
        # FIX-F3b: reasoning goes verbatim into the schema_reasoning/explanation
        # LABELS — scrub :bind mentions so the model never learns the token.
        from fine_tuning.preprocess.quality import scrub_reasoning
        reasoning = scrub_reasoning(str(row.get("reasoning", "")).strip())
        envelope = {
            "schema_reasoning": reasoning or _synth_reasoning(raw, tables),
            "sql":              raw,
            "tables_used":      tables,
            "confidence":       _gold_confidence(raw),   # FIX-F5: complexity-derived
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


# FIX-R6: VIEW chunks ("VIEW: v_...") are first-class schema context too —
# the old TABLE-only regex made them invisible to gold protection.
_TABLE_CHUNK_RE = re.compile(r"^(?:TABLE|VIEW|AUDIT TABLE): (\w+)", re.M)


def _split_table_chunks(schema_body: str) -> tuple[str, list[tuple[str, str]]]:
    """Split '=== SCHEMA CONTEXT ===\\n<chunks>' into (header_line, [(table, chunk_text)])."""
    lines = schema_body.split("\n", 1)
    header = lines[0]
    body   = lines[1] if len(lines) > 1 else ""
    starts = [(m.start(), m.group(1).lower()) for m in _TABLE_CHUNK_RE.finditer(body)]
    chunks: list[tuple[str, str]] = []
    for n, (pos, tbl) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(body)
        chunks.append((tbl, body[pos:end]))
    return header, chunks


# ── FIX-R3: gold-chunk compression (salvage stage before reject) ─────────────
# Chunk text structure (ingestion/chunk_generator._table_chunk / _view_chunk):
#   TABLE: <name>            ← identity line (always kept)
#   Purpose: / Description:  ← prose        (level 1 drops)
#   Columns:                 ← essential    (level 2 strips ' — comment' tails)
#   JSONB Column Notes:      ← verbose      (level 1 drops)
#   Foreign Key References:  ← essential    (never dropped — join signal)
_CHUNK_BLOCK_HEADERS = (
    "Purpose:", "Description:", "Columns:",
    "JSONB Column Notes:", "Foreign Key References:",
)
_COL_COMMENT_RE = re.compile(r" — .*$")


def _compress_gold_chunk(text: str, level: int) -> str:
    """Shrink one TABLE/VIEW chunk, preserving identity, columns and FKs.
    level 1: drop Purpose/Description + JSONB Column Notes blocks.
    level 2: additionally strip per-column comment tails (' — ...')."""
    lines = text.split("\n")
    out: list[str] = []
    current_block = ""                       # '' = identity/preamble
    for ln in lines:
        stripped = ln.strip()
        if stripped in _CHUNK_BLOCK_HEADERS:
            current_block = stripped
        if current_block in ("Purpose:", "Description:", "JSONB Column Notes:"):
            continue                         # level >= 1 drops these blocks
        if level >= 2 and current_block == "Columns:" and stripped not in _CHUNK_BLOCK_HEADERS:
            ln = _COL_COMMENT_RE.sub("", ln)
        out.append(ln)
    return "\n".join(out)


def fit_row(tok, row: dict, max_seq: int) -> tuple[dict, str]:
    """
    Token-budget one row. Returns (row, status);
    status ∈ {'ok', 'trimmed', 'trimmed_gold_compressed',
              'reject:gold_over_budget', 'reject:no_question'}.

    FIX-F2b — the OLD implementation cut the schema head blindly from its tail.
    Result on the v4 corpus (manifest: schema_trimmed=378/378): rows whose gold
    SQL referenced context-absent tables jumped from 23% pre-fit to 54% post-fit
    — half the gradient taught the model to write SQL over tables it could not
    see, which is precisely the hallucinated-column failure class that appeared
    after fine-tuning. NEW policy, section-aware:

      1. Drop whole low-value sections first, in TRAIN_TRIM_ORDER
         (EXAMPLES → GLOSSARY → ADDITIONAL → JOIN PATHS → STATUS MODEL → WORKFLOW).
      2. Then drop non-gold TABLE chunks from the end of SCHEMA CONTEXT.
      3. NEVER drop a TABLE chunk for a table in row['gold_tables'].
      4. If gold chunks alone still exceed the budget → REJECT the row.
         A rejected row costs one example; a kept-but-poisoned row costs accuracy.
    """
    from generation.prompt_builder import TRAIN_TRIM_ORDER, split_train_sections

    instr  = row["instruction"]; system = row.get("input", ""); output = row.get("output", "")
    gold   = {t.lower() for t in row.get("gold_tables", []) or []}
    idx = instr.find(QUESTION_MARK)
    if idx == -1:
        return row, "reject:no_question"
    head, tail = instr[:idx], instr[idx:]
    reserve = _count(tok, system) + _count(tok, tail) + _count(tok, output) + TEMPLATE_OVERHEAD
    head_budget = max_seq - reserve

    def head_text(sections: list[tuple[str, str]]) -> str:
        return "".join(b for _, b in sections)

    sections = split_train_sections(head)
    if _count(tok, head_text(sections)) <= head_budget:
        return row, "ok"

    trimmed = False
    # 1. whole-section drops
    for victim in TRAIN_TRIM_ORDER:
        if victim == "=== SCHEMA CONTEXT ===":
            break
        new_sections = [(h, b) for h, b in sections if h != victim]
        if len(new_sections) != len(sections):
            sections, trimmed = new_sections, True
            if _count(tok, head_text(sections)) <= head_budget:
                row["instruction"] = head_text(sections) + tail
                return row, "trimmed"

    # 2. drop non-gold TABLE chunks, last first
    out_sections: list[tuple[str, str]] = []
    for h, b in sections:
        if h != "=== SCHEMA CONTEXT ===":
            out_sections.append((h, b))
            continue
        sc_header, chunks = _split_table_chunks(b)
        keep = list(chunks)
        def assemble() -> str:
            body = sc_header + "\n" + "".join(t for _, t in keep)
            return head_text(out_sections) + body + head_text(rest)
        rest = [(hh, bb) for hh, bb in sections[sections.index((h, b)) + 1:]]
        for n in range(len(chunks) - 1, -1, -1):
            if _count(tok, assemble()) <= head_budget:
                break
            if chunks[n][0] in gold:
                continue                      # protected — never dropped
            keep.remove(chunks[n]); trimmed = True
        body = sc_header + "\n" + "".join(t for _, t in keep)
        out_sections.append((h, body))
    sections = out_sections

    final_head = head_text(sections)
    if _count(tok, final_head) > head_budget:
        # 3. FIX-R3 SALVAGE — gold-only context over budget. The OLD policy
        #    rejected here, deleting multi-join High-tier rows wholesale.
        #    NEW: compress the gold chunks (drop prose, keep columns + FKs —
        #    level 1; also strip column comments — level 2) and only reject
        #    if even the compressed gold context cannot fit.
        for level in (1, 2):
            comp_sections: list[tuple[str, str]] = []
            for h, b in sections:
                if h != "=== SCHEMA CONTEXT ===":
                    comp_sections.append((h, b)); continue
                sc_header, chunks = _split_table_chunks(b)
                body = sc_header + "\n" + "".join(
                    _compress_gold_chunk(t, level) for _, t in chunks
                )
                comp_sections.append((h, body))
            final_head = head_text(comp_sections)
            if _count(tok, final_head) <= head_budget:
                row["instruction"] = final_head + tail
                return row, "trimmed_gold_compressed"
        # 4. even compressed gold-only context does not fit → reject, not poison
        return row, "reject:gold_over_budget"
    row["instruction"] = final_head + tail
    return row, ("trimmed" if trimmed else "ok")


def fit_rows(rows: list[dict[str, Any]], model_dir: str, max_seq: int) -> tuple[list[dict], int, int, dict]:
    """
    Trim schema head so each row fits max_seq.
    Returns (rows, trimmed, still_over, drop_counts).

    FIX-F2c — post-fit hard gate: any surviving row whose gold tables are not
    all present as TABLE chunks in the final context is DROPPED and counted
    (gold_ctx_missing). This closes both damage paths at once: retrieval
    misses (23% pre-fix) and fit truncation (54% post-fit pre-fix).
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    out, trimmed, over = [], 0, 0
    drops: dict[str, int] = {}
    for row in rows:
        gold = {t.lower() for t in row.pop("gold_tables", []) or []}
        row, status = fit_row(tok, dict(row, gold_tables=list(gold)), max_seq)
        row.pop("gold_tables", None)          # working field — not a training column
        if status.startswith("reject"):
            drops[status] = drops.get(status, 0) + 1
            continue
        if status.startswith("trimmed"):
            trimmed += 1
        if status == "trimmed_gold_compressed":
            # FIX-R3: informational — rows salvaged by gold-chunk compression
            # instead of being rejected. Surfaces in the manifest.
            drops["salvaged_gold_compressed"] = drops.get("salvaged_gold_compressed", 0) + 1
        # hard gate: gold ⊆ context tables
        ctx = {m.group(1).lower() for m in _TABLE_CHUNK_RE.finditer(row["instruction"])}
        if gold and not gold <= ctx:
            drops["reject:gold_ctx_missing"] = drops.get("reject:gold_ctx_missing", 0) + 1
            continue
        full = (_count(tok, row.get("input", "")) + _count(tok, row["instruction"])
                + _count(tok, row.get("output", "")) + TEMPLATE_OVERHEAD)
        if full > max_seq:
            over += 1
        out.append(row)
    logger.info(component="preprocess.build", event="fit_complete",
                rows=len(out), trimmed=trimmed, still_over=over,
                dropped=drops, max_seq=max_seq)
    return out, trimmed, over, drops