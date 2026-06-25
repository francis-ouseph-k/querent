"""
retrieval/orchestrator.py
──────────────────────────
Retrieval orchestrator — combines all retrieval signals into a final
ranked, budgeted list of SemanticChunk objects ready for the prompt.

Pipeline:
  1. Dense vector search (Qdrant)
  2. BM25 keyword search (OpenSearch)
  3. RRF fusion (custom Python)
  4. FK graph expansion (NetworkX bidirectional BFS)
  5. Cross-encoder reranker (optional, Phase 1+)
  6. Context budget manager (token-aware trimming with priority ordering)

The "lost in the middle" mitigation is applied here:
  highest-relevance chunks → start of list
  lower-priority chunks    → end of list
  never bury critical context in the middle

FIXES IN THIS VERSION
─────────────────────
M2  — _apply_context_budget() separated mandatory entity chunks from
      remaining chunks using `c not in entity_chunks`, which uses dataclass
      equality (compares full text strings) — O(n²) and fragile if two chunks
      have identical text.  Fix: separate by chunk_id set — O(1) per lookup,
      correct regardless of text content.

M3  — tiktoken cl100k_base tokenizes differently from Qwen2.5's tokenizer.
      Original fix applied a 15% safety margin.  After aligning the tokenizer
      to Qwen2.5 (utils/tokenizer.py), the safety factor was set to 1.0
      (no margin needed).  The budget passed to _apply_context_budget is now
      the full configured value.

FIX-O1 — meta["qdrant_dense_hits"] and meta["opensearch_bm25_hits"] stored
          full chunk ID lists. These were never logged (the log line filters
          for keys ending in _ms) but bloated the meta dict in memory on every
          query. Fix: store hit counts instead of ID lists. ID lists are still
          available in meta["rrf_top_k"] for the top-20 fused results.

FIX-O2 — get_few_shot_examples() had no error handling. A transient Qdrant
          outage propagated an unhandled exception to runner.py and killed the
          entire query. Fix: catch all exceptions, log a warning, return [].
          Few-shot examples are best-effort context — their absence should
          degrade quality, not crash the pipeline.
"""

from __future__ import annotations

import time
from typing import Any

import networkx as nx

from utils.tokenizer import count_tokens as _count_tokens
from config.settings import settings
from indexing.opensearch_indexer import OpenSearchIndexer
from indexing.qdrant_indexer import QdrantIndexer
from models.schema import ChunkType, SemanticChunk
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Priority order for context budget trimming (highest = most important)
_CHUNK_PRIORITY: dict[ChunkType, int] = {
    ChunkType.BUSINESS_RULE: 11,
    ChunkType.TABLE:     10,
    ChunkType.VIEW:      10,
    ChunkType.FK_MAP:     9,
    ChunkType.WORKFLOW:   8,
    ChunkType.STATUS:     7,
    ChunkType.GLOSSARY:   6,
    ChunkType.AUDIT:      5,
    ChunkType.INDEX:      4,
    ChunkType.PARTITION:  3,
    ChunkType.FEW_SHOT:   0,   # handled separately
}

# Safety margin to compensate for tokenizer mismatch.
# M-1 fix: aligned tokenizer to Qwen2.5, so 1.0 = no safety reduction needed.
_TOKEN_BUDGET_SAFETY_FACTOR = 1.0


class RetrievalOrchestrator:
    """
    Orchestrates the full hybrid retrieval pipeline.

    Usage:
        orch = RetrievalOrchestrator(qdrant, opensearch, graph)
        chunks, meta = orch.retrieve(query, entities=["board", "evaluation_attempt"])
    """

    def __init__(
        self,
        qdrant_indexer:     QdrantIndexer,
        opensearch_indexer: OpenSearchIndexer,
        fk_graph:           nx.DiGraph,
        reranker=None,       # optional CrossEncoderReranker instance
    ) -> None:
        self.qdrant     = qdrant_indexer
        self.opensearch = opensearch_indexer
        self.graph      = fk_graph
        self.reranker   = reranker
        from ingestion.graph_builder import GraphBuilder
        self._graph_builder = GraphBuilder()

    def retrieve(
        self,
        query_text:      str,
        entity_tables:   list[str]   = None,
        intent:          str         = "unknown",
        top_k_dense:     int         | None = None,
        top_k_bm25:      int         | None = None,
        budget_tokens:   int         | None = None,
    ) -> tuple[list[SemanticChunk], dict[str, Any]]:
        """
        Full retrieval pipeline. Returns (chunks, metadata_dict).

        chunks       — final ordered list ready for the prompt (highest priority first)
        metadata     — retrieval diagnostics for observability logging
        entity_tables must always be included in the output regardless of score.
        """
        entity_tables  = entity_tables or []
        top_k_dense    = top_k_dense  or settings.retrieval.dense_top_k
        top_k_bm25     = top_k_bm25   or settings.retrieval.bm25_top_k
        raw_budget     = budget_tokens or settings.retrieval.context_budget_tokens
        # M3: apply safety margin to compensate for tiktoken/Qwen tokenizer mismatch
        budget_tokens  = int(raw_budget * _TOKEN_BUDGET_SAFETY_FACTOR)

        t_start = time.time()
        meta: dict[str, Any] = {}

        # ── Step 1: Dense vector search ───────────────────────────────────
        t0          = time.time()
        dense_hits  = self.qdrant.search(
            query_text  = query_text,
            top_k       = top_k_dense,
            chunk_types = [ct for ct in ChunkType if ct != ChunkType.FEW_SHOT],
        )
        # FIX-O1: store count not ID list — IDs bloat meta dict, never logged
        meta["qdrant_dense_hits"] = len(dense_hits)
        meta["qdrant_dense_ms"]   = round((time.time() - t0) * 1000)
        # Debug: store full hit payloads (only when debug_mode active)
        if settings.debug_mode:
            debug_dense = []
            for hit in dense_hits:
                copy_hit = dict(hit)
                copy_hit["tokens"] = _count_tokens(hit.get("text", ""))
                debug_dense.append(copy_hit)
            meta["_debug_dense_hits"] = debug_dense

        # ── Step 2: BM25 keyword search ───────────────────────────────────
        t0         = time.time()
        bm25_hits  = self.opensearch.search(query_text=query_text, top_k=top_k_bm25)
        # FIX-O1: store count not ID list
        meta["opensearch_bm25_hits"] = len(bm25_hits)
        meta["opensearch_bm25_ms"]   = round((time.time() - t0) * 1000)
        # Debug: store full hit payloads
        if settings.debug_mode:
            debug_bm25 = []
            for hit in bm25_hits:
                copy_hit = dict(hit)
                copy_hit["tokens"] = _count_tokens(hit.get("text", ""))
                debug_bm25.append(copy_hit)
            meta["_debug_bm25_hits"] = debug_bm25

        # ── Step 3: RRF fusion ────────────────────────────────────────────
        t0    = time.time()
        fused = _rrf_merge(dense_hits, bm25_hits, k=settings.retrieval.rrf_k)
        meta["rrf_top_k"] = [{"id": cid, "score": round(score, 4)} for cid, score in fused[:20]]
        meta["rrf_ms"]    = round((time.time() - t0) * 1000)

        # Build chunk lookup from both result sets
        chunk_lookup: dict[str, dict[str, Any]] = {}
        for hit in dense_hits + bm25_hits:
            chunk_lookup[hit["chunk_id"]] = hit

        # ── Step 4: FK graph expansion ────────────────────────────────────
        t0 = time.time()
        graph_tables: set[str] = set(entity_tables)
        join_paths:   list[str] = []

        if entity_tables and self.graph:
            bfs_res      = self._graph_builder.find_join_paths(
                G           = self.graph,
                seed_tables = entity_tables,
            )
            graph_tables = bfs_res["connecting_tables"]
            join_paths   = bfs_res["path_descriptions"]

        meta["graph_tables"] = sorted(graph_tables)
        meta["join_paths"]   = join_paths
        meta["graph_ms"]     = round((time.time() - t0) * 1000)

        # ── Step 5: Materialise top chunks from RRF list ──────────────────
        ordered_chunks:        list[SemanticChunk] = []
        ordered_chunks_scores: dict[str, float]   = {}
        seen_ids:              set[str]            = set()

        for chunk_id, rrf_score in fused:
            if chunk_id in chunk_lookup and chunk_id not in seen_ids:
                chunk = SemanticChunk.from_payload(chunk_lookup[chunk_id])
                ordered_chunks.append(chunk)
                ordered_chunks_scores[chunk_id] = rrf_score
                seen_ids.add(chunk_id)

        # ── Step 6: Ensure entity TABLE + FK_MAP chunks are always present ──
        entity_mandatory = self._fetch_mandatory_chunks_for(entity_tables, chunk_lookup)
        for chunk in entity_mandatory:
            if chunk.chunk_id not in seen_ids:
                ordered_chunks.insert(0, chunk)
                ordered_chunks_scores[chunk.chunk_id] = 999.0
                seen_ids.add(chunk.chunk_id)

        # ── Step 7: Optional cross-encoder reranker ───────────────────────
        reranker_applied = False
        if self.reranker and settings.reranker.enabled:
            t0             = time.time()
            ordered_chunks = self.reranker.rerank(
                query  = query_text,
                chunks = ordered_chunks,
                top_k  = settings.reranker.top_k_output,
            )
            reranker_applied     = True
            meta["reranker_ms"] = round((time.time() - t0) * 1000)

        meta["reranker_applied"] = reranker_applied

        # ── Step 8: Context budget manager ───────────────────────────────
        t0 = time.time()
        final_chunks, tokens_used = _apply_context_budget(
            chunks        = ordered_chunks,
            entity_tables = set(entity_tables),
            rrf_scores    = ordered_chunks_scores,
            budget_tokens = budget_tokens,
        )
        meta["tokens_used"]        = tokens_used
        meta["budget_ms"]          = round((time.time() - t0) * 1000)
        meta["total_ms"]           = round((time.time() - t_start) * 1000)
        meta["effective_budget"]   = budget_tokens   # M3: log actual budget used

        # Debug: store final prompt-ready chunks with their RRF scores
        if settings.debug_mode:
            meta["_debug_final_chunks"] = [
                {
                    "chunk_id":   c.chunk_id,
                    "chunk_type": c.chunk_type.value,
                    "table_name": c.table_name,
                    "rrf_score":  round(ordered_chunks_scores.get(c.chunk_id, 0.0), 4),
                    "tokens":     _count_tokens(c.text),
                    "text":       c.text,
                }
                for c in final_chunks
            ]
            meta["_debug_rrf_scores"]    = ordered_chunks_scores
            meta["_debug_join_paths"]    = join_paths

        logger.info(
            component="retrieval_orchestrator",
            event="retrieve_complete",
            query=query_text[:80],
            intent=intent,
            chunks_returned=len(final_chunks),
            tokens_used=tokens_used,
            effective_budget=budget_tokens,
            **{k: v for k, v in meta.items() if k.endswith("_ms")},
        )

        return final_chunks, meta

    def get_few_shot_examples(self, query_text: str, top_k: int = 3) -> list[SemanticChunk]:
        """
        Retrieve FEW_SHOT examples from Qdrant by semantic similarity.

        FIX-O2: catches all exceptions and returns [] on failure.
        Few-shot examples are best-effort context — a Qdrant outage should
        degrade quality gracefully, not crash the entire query pipeline.
        """
        try:
            hits = self.qdrant.get_few_shot_examples(query_text=query_text, top_k=top_k)
            return [SemanticChunk.from_payload(h) for h in hits]
        except Exception as exc:
            logger.warning(
                component="retrieval_orchestrator",
                event="few_shot_retrieval_failed",
                error=str(exc),
                note="Continuing without few-shot examples",
            )
            return []

    def _fetch_mandatory_chunks_for(
        self,
        tables:       list[str],
        chunk_lookup: dict[str, dict[str, Any]],
    ) -> list[SemanticChunk]:
        """
        Fetch both TABLE and FK_MAP chunks for entity-extracted tables and
        return them as mandatory (inserted at the front of context regardless
        of RRF score).

        Fetch strategy:
          1. Check if already present in the RRF result set (chunk_lookup).
          2. If missing, do a targeted Qdrant lookup for each missing type.
        """
        chunks: list[SemanticChunk]       = []
        found:  dict[str, set[ChunkType]] = {}

        for payload in chunk_lookup.values():
            tname     = payload.get("table_name", "")
            ctype_str = payload.get("chunk_type", "")
            if tname not in tables:
                continue
            try:
                ctype = ChunkType(ctype_str)
            except ValueError:
                continue
            if ctype in (ChunkType.TABLE, ChunkType.FK_MAP):
                chunks.append(SemanticChunk.from_payload(payload))
                found.setdefault(tname, set()).add(ctype)

        for table in tables:
            for needed_type in (ChunkType.TABLE, ChunkType.FK_MAP):
                if needed_type not in found.get(table, set()):
                    hits = self.qdrant.search(
                        query_text     = table,
                        top_k          = 1,
                        chunk_types    = [needed_type],
                        filter_payload = {"table_name": table},
                    )
                    if hits:
                        chunks.append(SemanticChunk.from_payload(hits[0]))

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# RRF fusion — standalone function
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_merge(
    dense_hits: list[dict[str, Any]],
    bm25_hits:  list[dict[str, Any]],
    k:          int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    Returns list of (chunk_id, score) sorted descending by RRF score.
    k=60 is the standard constant from the original RRF paper.

    M-4 fix: deduplicates chunk_ids within each source to prevent
    double-counting when Qdrant/OpenSearch returns the same chunk_id
    twice (possible with overlapping filter conditions).
    """
    scores: dict[str, float] = {}

    # M-4 fix: track seen chunk_ids per source to prevent double-counting
    seen_dense: set[str] = set()
    for rank, hit in enumerate(dense_hits):
        cid = hit["chunk_id"]
        if cid not in seen_dense:
            seen_dense.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    seen_bm25: set[str] = set()
    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        if cid not in seen_bm25:
            seen_bm25.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Context budget manager
# ─────────────────────────────────────────────────────────────────────────────

def _apply_context_budget(
    chunks:        list[SemanticChunk],
    entity_tables: set[str],
    budget_tokens: int,
    rrf_scores:    dict[str, float] | None = None,
) -> tuple[list[SemanticChunk], int]:
    """
    Trim chunks to fit within the token budget using a quota-per-type system.

    Two-pass approach:
      Pass 1 — fill guaranteed quota slots for critical chunk types.
      Pass 2 — fill remaining budget with highest (priority, rrf_score) chunks
               from the leftover pool, iterating ALL remaining chunks rather
               than stopping at the first that doesn't fit.

    FIX-M2: entity chunk separation now uses chunk_id sets (O(1) lookup,
    correct regardless of text content) instead of dataclass equality
    comparison (O(n) text comparison, fragile).
    """
    rrf_scores = rrf_scores or {}

    # ── Step 1: Mandatory chunks — entity TABLE + entity FK_MAP ──────────
    # M2: use chunk_id set for O(1) membership test, not `c not in list`
    # which uses dataclass __eq__ (compares full text — O(n) and fragile).
    entity_ids: set[str] = set()
    for c in chunks:
        if c.table_name in entity_tables and c.chunk_type in (ChunkType.TABLE, ChunkType.FK_MAP):
            entity_ids.add(c.chunk_id)

    entity_chunks = []
    remaining = []
    for c in chunks:
        if c.chunk_id in entity_ids:
            entity_chunks.append(c)
        else:
            remaining.append(c)
    used_tokens   = sum(_count_tokens(c.text) for c in entity_chunks)
    final_chunks  = list(entity_chunks)

    if used_tokens >= budget_tokens:
        logger.warning(
            event="budget_entity_only",
            note="Entity TABLE + FK_MAP chunks alone fill the budget.",
            used=used_tokens,
            budget=budget_tokens,
        )
        return final_chunks, used_tokens

    available = budget_tokens - used_tokens

    # ── Step 2: Sort remaining by (priority DESC, rrf_score DESC) ────────
    def get_sort_key(chunk):
        # Multi-key sorting: primary key is chunk type priority, secondary is the rank fusion score.
        # Python sorts tuples element-by-element.
        priority = _CHUNK_PRIORITY.get(chunk.chunk_type, 0)
        score = rrf_scores.get(chunk.chunk_id, 0.0)
        return (priority, score)

    remaining.sort(key=get_sort_key, reverse=True)

    # ── Step 3: Quota-based selection ────────────────────────────────────
    tier_quotas: dict[str, int] = {
        "critical":   int(available * 0.40),   # FK_MAP + WORKFLOW — boosted for join accuracy
        "supporting": int(available * 0.35),   # TABLE (non-entity) + STATUS — boosted for column coverage
        "glossary":   int(available * 0.10),   # GLOSSARY + FEW_SHOT — reduced; column defs matter more
        "auxiliary":  available,               # INDEX + AUDIT + PARTITION
    }

    def _tier(chunk_type: ChunkType) -> str:
        if chunk_type in (ChunkType.BUSINESS_RULE, ChunkType.FK_MAP, ChunkType.WORKFLOW):
            return "critical"
        if chunk_type in (ChunkType.TABLE, ChunkType.VIEW, ChunkType.STATUS):
            return "supporting"
        if chunk_type in (ChunkType.GLOSSARY, ChunkType.FEW_SHOT):
            return "glossary"
        return "auxiliary"

    tier_used: dict[str, int] = {t: 0 for t in tier_quotas}
    selected:  list[SemanticChunk] = []
    deferred:  list[SemanticChunk] = []

    # Pass 1: fill quota slots
    for chunk in remaining:
        tok  = _count_tokens(chunk.text)
        tier = _tier(chunk.chunk_type)
        if tier_used[tier] + tok <= tier_quotas[tier]:
            selected.append(chunk)
            tier_used[tier] += tok
        else:
            deferred.append(chunk)

    # Pass 2: fill remaining budget — iterate ALL deferred, never break early
    pass1_tokens = sum(_count_tokens(c.text) for c in selected)
    leftover     = available - pass1_tokens

    for chunk in deferred:
        tok = _count_tokens(chunk.text)
        if tok <= leftover:
            selected.append(chunk)
            leftover -= tok

    final_chunks.extend(selected)
    used_tokens += sum(_count_tokens(c.text) for c in selected)

    # Warn if critical chunks were dropped
    retrieved_types  = {c.chunk_type for c in chunks}
    included_types   = {c.chunk_type for c in final_chunks}
    dropped_critical = (retrieved_types & {ChunkType.FK_MAP, ChunkType.WORKFLOW}) \
                       - included_types
    if dropped_critical:
        logger.warning(
            event="critical_chunks_dropped",
            dropped_types=[ct.value for ct in dropped_critical],
            note="FK_MAP or WORKFLOW chunks were retrieved but dropped by budget. "
                 "Consider increasing RETRIEVAL_CONTEXT_BUDGET_TOKENS.",
        )

    # "Lost in the middle" reorder — high priority at start
    high_priority = [c for c in final_chunks if _CHUNK_PRIORITY.get(c.chunk_type, 0) >= 8]
    low_priority  = [c for c in final_chunks if _CHUNK_PRIORITY.get(c.chunk_type, 0) < 8]

    return high_priority + low_priority, used_tokens