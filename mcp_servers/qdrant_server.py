"""
mcp_servers/qdrant_server.py
─────────────────────────────
MCP server wrapping the Qdrant dense-vector store for the NL→SQL pipeline.

FASTMCP VERSION: 3.x (tested on 3.4.2)
  Import : from fastmcp import FastMCP        (standalone package, NOT mcp.server.fastmcp)
  Startup: mcp.run(transport="http", ...)     (FastMCP calls uvicorn internally)

  FastMCP 3.x manages uvicorn internally via run_http_async().
  You do not call uvicorn.run() directly — FastMCP does it for you.
  To pass uvicorn tuning options use the uvicorn_config parameter (see entry point).

Exposes six tools:
  search_chunks          — dense vector search (called on every user query)
  get_few_shot_examples  — FEW_SHOT-only semantic similarity search
  upsert_chunks          — embed + upsert chunks into Qdrant (ingest-time)
  delete_chunks          — delete stale chunks on DDL change (ingest-time)
  ensure_collection      — create collection if absent (first run / --full)
  drop_collection        — drop collection entirely (--full re-ingestion)

WHY A SEPARATE PROCESS
  This server owns both the Qdrant connection and the BGE-small-en-v1.5
  embedding model (~90 MB RAM, ~1-2s load time).

  Benefits of isolating it here:
    1. The embedding model is loaded ONCE on first tool call and reused for
       every subsequent request — no per-request or per-worker reload.
    2. Multiple application workers share one model instance via MCP instead
       of each loading their own 90 MB copy.
    3. When Qdrant releases a breaking client API change, only this file
       needs updating — the main application is unaffected.
    4. Qdrant client version is pinned here; the main app has no qdrant-client
       dependency to manage.

TRANSPORT
  FastMCP 3.x Streamable HTTP (MCP 2025-03-26 spec, the current standard).
  Tools are called via: POST http://<host>:<port>/mcp
  If you need backward compatibility with older MCP clients use transport="sse".

STARTUP
  python mcp_servers/qdrant_server.py

CONFIG (.env — all optional, defaults shown)
  MCP_QDRANT_HOST=127.0.0.1
  MCP_QDRANT_PORT=5010
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# ── Add project root to sys.path so config/settings.py resolves correctly
# when this server is started from any working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastmcp import FastMCP                         # FastMCP 3.x standalone package
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── MCP server instance ────────────────────────────────────────────────────────
mcp = FastMCP(
    name        = "qdrant-schema-chunks",
    instructions = "Dense vector search and upsert for NL→SQL schema chunks",
)

# ── Module-level lazy singletons ───────────────────────────────────────────────
# Initialised on first tool call, reused for all subsequent requests.
# Module-level variables are safe here because FastMCP 3.x runs each server
# in a single process with a single asyncio event loop — no concurrent init races.
_qdrant_client: QdrantClient        | None = None
_embed_model:   SentenceTransformer | None = None


def _get_client() -> QdrantClient:
    """
    Lazy Qdrant client — connects on first call, reused for all subsequent calls.

    Reads host/port from settings.qdrant which maps QDRANT_HOST / QDRANT_PORT
    from .env. Default: localhost:6333.
    Timeout of 30s applies to all operations (search, upsert, delete).
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host    = settings.qdrant.host,
            port    = settings.qdrant.port,
            timeout = 30,
        )
        logger.info(
            component = "qdrant_mcp",
            event     = "client_connected",
            host      = settings.qdrant.host,
            port      = settings.qdrant.port,
        )
    return _qdrant_client


def _get_embedder() -> SentenceTransformer:
    """
    Lazy embedding model — loaded ONCE on first call, kept in RAM for the
    lifetime of this server process.

    Why lazy: loading BGE-small-en-v1.5 takes ~1-2 seconds and uses ~90 MB RAM.
    Loading at first call means the server starts instantly and pays the cost
    only when the first actual query arrives.

    Why in-process: the model stays loaded between requests. Per-request loading
    would add 1-2s latency to every query — unacceptable for a live CLI tool.

    Device is read from settings.embedding.device (EMBED_DEVICE in .env).
    Default is "cpu". The GPU is reserved for llama-server (the LLM inference
    engine). Only set EMBED_DEVICE=cuda if you have a second GPU or are running
    the LLM on CPU only.
    """
    global _embed_model
    if _embed_model is None:
        logger.info(
            component = "qdrant_mcp",
            event     = "loading_embedding_model",
            model     = settings.embedding.model_name,
            device    = settings.embedding.device,
            note      = "First call only — model stays in RAM after this.",
        )
        _embed_model = SentenceTransformer(
            settings.embedding.model_name,
            device = settings.embedding.device,
        )
        logger.info(component="qdrant_mcp", event="embedding_model_loaded")
    return _embed_model


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings into float vectors using BGE-small-en-v1.5.

    Runs in batches of settings.embedding.batch_size (default 32) to avoid
    memory pressure during large ingest runs. At query time texts=[one query],
    so batching has no effect — it only matters during ingest.

    normalize_embeddings=True is required: BGE-small vectors must be L2-normalised
    for cosine similarity to work correctly in Qdrant. Without normalisation,
    dot-product distance would be used instead and scores would be misleading.

    Args:
        texts: List of strings. Can be a single query string or many chunk texts.

    Returns:
        List of float vectors (384-dim for BGE-small), one per input text,
        in the same order as the input. Normalised to unit length.
    """
    model      = _get_embedder()
    batch_size = settings.embedding.batch_size
    vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(
            batch,
            normalize_embeddings = True,    # required for cosine similarity
            show_progress_bar    = False,
        ).tolist()
        vectors.extend(vecs)

    return vectors


def _build_filter(
    chunk_types:    list[str] | None,
    filter_payload: dict[str, str] | None,
) -> qmodels.Filter | None:
    """
    Build a Qdrant payload filter from optional chunk_types and key-value pairs.

    chunk_types → OR logic: any matching type is included.
      e.g. ["TABLE", "FK_MAP"] means (chunk_type=TABLE OR chunk_type=FK_MAP)

    filter_payload → AND logic: every key-value must match.
      e.g. {"table_name": "answer_script"} AND'd with the chunk_types filter.

    Combined example:
      chunk_types=["TABLE"], filter_payload={"table_name": "board"}
      → chunk_type = TABLE  AND  table_name = board

    Returns None when both arguments are empty — no filter, all chunks returned.
    """
    conditions: list[Any] = []

    if chunk_types:
        # OR: any of the listed chunk types qualifies
        conditions.append(
            qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key   = "chunk_type",
                        match = qmodels.MatchValue(value=ct),
                    )
                    for ct in chunk_types
                ]
            )
        )

    if filter_payload:
        # AND: every key-value pair must match exactly
        for key, value in filter_payload.items():
            conditions.append(
                qmodels.FieldCondition(
                    key   = key,
                    match = qmodels.MatchValue(value=value),
                )
            )

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return qmodels.Filter(must=conditions)


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_chunks(
    query_text:     str,
    top_k:          int               = 20,
    chunk_types:    list[str] | None  = None,
    filter_payload: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Dense vector search over schema chunks — the core retrieval operation.

    Called by retrieval/orchestrator.py on every user query. The query text
    is embedded with BGE-small-en-v1.5 (one forward pass, ~5-15ms on CPU)
    and the resulting vector is searched against all chunk vectors in Qdrant
    by cosine similarity.

    The top_k results are returned sorted by similarity score (highest first).
    These are then RRF-fused with OpenSearch BM25 results in the orchestrator.

    Args:
        query_text:     Natural language query — embedded at call time by
                        BGE-small-en-v1.5. No pre-processing needed.
        top_k:          Maximum results to return (default 20).
        chunk_types:    Optional ChunkType filter — OR logic.
                        e.g. ["TABLE", "FK_MAP", "WORKFLOW", "STATUS"]
                        None = all types returned (including FEW_SHOT).
        filter_payload: Optional exact key-value payload filters — AND logic.
                        e.g. {"table_name": "answer_script"}
                        Combined with chunk_types using AND.

    Returns:
        List of dicts, each containing all chunk payload fields plus:
          "chunk_id" : UUID string (content-addressed SHA-256)
          "score"    : cosine similarity (0.0–1.0, higher = more similar)
        Sorted by score descending.
    """
    t0       = time.time()
    col_name = settings.qdrant.collection_name

    # Embed the query — single string → single vector
    vector  = _embed([query_text])[0]
    qfilter = _build_filter(chunk_types, filter_payload)

    results = _get_client().query_points(
        collection_name = col_name,
        query           = vector,
        limit           = top_k,
        query_filter    = qfilter,
        with_payload    = True,    # return the full chunk payload, not just IDs
    )

    hits = [
        {"chunk_id": hit.id, "score": hit.score, **hit.payload}
        for hit in results.points
    ]

    logger.info(
        component  = "qdrant_mcp",
        event      = "search_complete",
        query      = query_text[:60],
        top_k      = top_k,
        hits       = len(hits),
        elapsed_ms = round((time.time() - t0) * 1000),
    )
    return hits


@mcp.tool()
def get_few_shot_examples(
    query_text: str,
    top_k:      int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve FEW_SHOT NL→SQL example pairs by semantic similarity.

    This is a separate tool from search_chunks because few-shot retrieval is
    a dedicated call with a fixed chunk_type filter. It does NOT compete with
    schema chunk retrieval for the same result slots — the orchestrator calls
    both independently and assembles the prompt from both result sets.

    WHY DENSE-ONLY (no BM25 for few-shot):
      Example matching is a semantic similarity problem, not a keyword problem.
      A user typing "how many scripts haven't been marked" should match the
      example "count of unevaluated answer scripts per board" — zero keyword
      overlap but high semantic similarity. BM25 would score this near zero.
      Dense cosine similarity captures the intent match correctly.
      This is why FEW_SHOT chunks are never indexed in OpenSearch.

    Args:
        query_text: Natural language query to match examples against.
        top_k:      Number of examples to return (default 3 — enough context,
                    not so many that they crowd out schema chunks in the prompt).

    Returns:
        List of FEW_SHOT chunk payloads with similarity scores.
        Each payload contains: nl_question, expected_sql, intent, tables.
    """
    return search_chunks(
        query_text  = query_text,
        top_k       = top_k,
        chunk_types = ["FEW_SHOT"],
    )


@mcp.tool()
def upsert_chunks(chunks: list[dict[str, Any]]) -> dict[str, int]:
    """
    Embed and upsert semantic chunks into Qdrant (ingest-time operation).

    Called by ingest.py after DDL parsing and chunk generation. Handles both
    first-time indexing and incremental updates — the upsert operation is
    idempotent because chunk_id is content-addressed (SHA-256 of the chunk
    text). Upserting a chunk with the same ID and same text is a no-op in Qdrant.

    Batching: chunks are processed in groups of 100. This is below Qdrant's
    default 100 MB gRPC request limit. With 384-dim float32 vectors, 100 chunks
    ≈ 150 KB payload — well within limits. Larger batches need server-side
    max_grpc_message_size tuning.

    Args:
        chunks: List of SemanticChunk.to_payload() dicts. Each must contain:
                  chunk_id   — UUID string (content-addressed SHA-256 of text)
                  text       — the prose text to be embedded into a vector
                  chunk_type — e.g. "TABLE", "FK_MAP", "WORKFLOW", "FEW_SHOT"
                  table_name — owning table name (empty for GLOSSARY/FEW_SHOT)
                  + all other SemanticChunk fields

    Returns:
        {"upserted": N} where N = total chunks written to Qdrant.
    """
    if not chunks:
        return {"upserted": 0}

    t0       = time.time()
    col_name = settings.qdrant.collection_name

    # Embed all chunk texts — the text field is what gets vectorised
    texts   = [c["text"] for c in chunks]
    vectors = _embed(texts)

    # Build Qdrant PointStruct list: id + vector + full payload
    points = [
        qmodels.PointStruct(
            id      = chunk["chunk_id"],
            vector  = vectors[i],
            payload = chunk,           # full payload stored for retrieval
        )
        for i, chunk in enumerate(chunks)
    ]

    # Upsert in batches of 100 to stay within gRPC size limits
    batch_size = 100
    for i in range(0, len(points), batch_size):
        _get_client().upsert(
            collection_name = col_name,
            points          = points[i : i + batch_size],
            wait            = True,    # synchronous: wait for indexing confirmation
        )

    logger.info(
        component  = "qdrant_mcp",
        event      = "upsert_complete",
        count      = len(chunks),
        elapsed_ms = round((time.time() - t0) * 1000),
    )
    return {"upserted": len(chunks)}


@mcp.tool()
def delete_chunks(changed_tables: list[str]) -> dict[str, int]:
    """
    Delete all chunks whose referenced_tables field contains any changed table.

    Called by ingest.py during INCREMENTAL DDL updates, BEFORE upsert_chunks().
    Without this step, stale chunks become orphans:
      - DDL change → text changes → new SHA-256 → new chunk_id
      - Old chunk_id stays in Qdrant with the stale schema text
      - Both old and new chunks are retrievable — stale schema in prompts

    This call removes all chunks that reference any of the changed tables,
    so only fresh chunks remain after upsert_chunks() runs.

    Filter uses OR logic: a chunk is deleted if ANY of its referenced_tables
    matches ANY entry in changed_tables.

    Args:
        changed_tables: List of table names whose chunks should be removed.
                        e.g. ["answer_script", "evaluation_attempt"]

    Returns:
        {"deleted": N} where N = number of points removed from Qdrant.
    """
    if not changed_tables:
        return {"deleted": 0}

    col_name = settings.qdrant.collection_name

    # OR filter: delete any chunk that references ANY of the changed tables
    filter_cond = qmodels.Filter(
        should=[
            qmodels.FieldCondition(
                key   = "referenced_tables",
                match = qmodels.MatchValue(value=table),
            )
            for table in changed_tables
        ]
    )

    result  = _get_client().delete(
        collection_name  = col_name,
        points_selector  = qmodels.FilterSelector(filter=filter_cond),
        wait             = True,
    )
    deleted = getattr(result, "deleted_count", 0)

    logger.info(
        component = "qdrant_mcp",
        event     = "chunks_deleted",
        tables    = sorted(changed_tables),
        deleted   = deleted,
    )
    return {"deleted": deleted}


@mcp.tool()
def ensure_collection() -> dict[str, str]:
    """
    Create the Qdrant collection if it does not already exist.

    Called by ingest.py on first run and after drop_collection() during
    --full re-ingestion.

    Collection settings:
      - Vector size: settings.qdrant.vector_size (default 384 for BGE-small)
      - Distance:    COSINE — required because BGE-small vectors are L2-normalised
      - Name:        settings.qdrant.collection_name (QDRANT_COLLECTION_NAME in .env)

    Safe to call repeatedly — returns "exists" without modifying anything if
    the collection is already there. No risk of accidental data loss.

    Returns:
        {"status": "created"} or {"status": "exists"}
    """
    col_name = settings.qdrant.collection_name
    existing = {c.name for c in _get_client().get_collections().collections}

    if col_name not in existing:
        _get_client().create_collection(
            collection_name = col_name,
            vectors_config  = qmodels.VectorParams(
                size     = settings.qdrant.vector_size,   # 384 for BGE-small-en-v1.5
                distance = qmodels.Distance.COSINE,
            ),
        )
        logger.info(component="qdrant_mcp", event="collection_created", name=col_name)
        return {"status": "created"}

    logger.info(component="qdrant_mcp", event="collection_exists", name=col_name)
    return {"status": "exists"}


@mcp.tool()
def drop_collection() -> dict[str, str]:
    """
    Drop the Qdrant collection entirely — USE WITH CAUTION.

    All vectors and payloads are permanently deleted.

    Called by ingest.py --full to perform a clean-slate rebuild. This is
    needed because content-addressed chunk IDs mean that any text change
    produces a new ID — old IDs become orphans that remain retrievable.
    drop_collection() removes all orphans in one operation.

    After calling this you MUST call ensure_collection() then upsert_chunks()
    to rebuild. The ingest.py --full flag does all three steps in order
    automatically — you should not need to call this tool directly.

    Returns:
        {"status": "dropped"} or {"status": "not_found"}
    """
    col_name = settings.qdrant.collection_name
    existing = {c.name for c in _get_client().get_collections().collections}

    if col_name in existing:
        _get_client().delete_collection(col_name)
        logger.info(component="qdrant_mcp", event="collection_dropped", name=col_name)
        return {"status": "dropped"}

    return {"status": "not_found"}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = settings.mcp.qdrant_host
    port = settings.mcp.qdrant_port

    logger.info(
        component = "qdrant_mcp",
        event     = "server_starting",
        host      = host,
        port      = port,
    )

    # FastMCP 3.x calls uvicorn internally via run_http_async().
    # You do not call uvicorn.run() directly — FastMCP manages it.
    #
    # transport="http"  = Streamable HTTP (MCP 2025-03-26 spec, recommended)
    # transport="sse"   = legacy SSE transport for older MCP clients
    #
    # Host and port come from settings.mcp which reads MCP_QDRANT_HOST /
    # MCP_QDRANT_PORT from .env — defaults are 127.0.0.1 and 5010.
    mcp.run(
        transport      = "http",
        host           = host,
        port           = port,
        json_response  = True,   # return plain JSON instead of SSE stream
        stateless_http = True,   # no session handshake required per call
    )

    # If you ever need to tune uvicorn (e.g. log level, workers):
    # mcp.run(
    #     transport      = "http",
    #     host           = host,
    #     port           = port,
    #     uvicorn_config = {"log_level": "warning", "workers": 1},
    # )