"""
mcp_servers/opensearch_server.py
──────────────────────────────────
MCP server wrapping the OpenSearch BM25 keyword store for the NL→SQL pipeline.

FASTMCP VERSION: 3.x (tested on 3.4.2)
  Import : from fastmcp import FastMCP
  Startup: mcp.run(transport="http", ...)   (FastMCP calls uvicorn internally)

  FastMCP 3.x manages uvicorn internally via run_http_async().
  You do not call uvicorn.run() directly — FastMCP does it for you.
  To pass uvicorn tuning options use the uvicorn_config parameter (see entry point).

Exposes five tools:
  search_chunks   — BM25 keyword search (called on every user query)
  index_chunks    — bulk index chunks, skips FEW_SHOT (ingest-time)
  delete_chunks   — delete stale chunks on DDL change (ingest-time)
  ensure_index    — create index with dual-field mapping if absent
  drop_index      — drop index entirely (--full re-ingestion)

WHY A SEPARATE PROCESS
  OpenSearch is the BM25 (keyword) half of the hybrid retrieval system.
  Qdrant handles dense vectors; OpenSearch handles keyword matching.
  Results from both are combined by RRF fusion in retrieval/orchestrator.py.

  Benefits of isolating it here:
    1. When opensearch-py releases a breaking API change, only this file
       needs updating — the main application is unaffected.
    2. The dual-field index mapping (H5 fix) is defined and enforced here
       at create time. No risk of a stale single-field mapping surviving.
    3. SSL certificate warnings are suppressed here once, not scattered
       across multiple callers.

DUAL-FIELD MAPPING — WHY IT EXISTS (H5 fix)
  OpenSearch tokenises text at index time. The standard analyser lowercases
  everything — good for prose ("frozen scripts" matches "Frozen evaluation")
  but breaks domain codes ("DEK", "KEK", "W", "P" must stay uppercase).

  Solution: every text field gets TWO sub-fields indexed separately:
    field "text"     — standard analyser: lowercased tokens → prose recall
    field "text.raw" — domain_code_analyser: case-preserved tokens → code recall

  search_chunks queries BOTH via multi_match best_fields so:
    user types "frozen scripts"  → "frozen" matches text (lowercased)
    user types "FROZEN"          → "FROZEN" matches text.raw (preserved)
    user types "DEK"             → "DEK" matches text.raw (would vanish in lowercase)

  IMPORTANT: this mapping cannot be changed on an existing index.
  After this fix was first applied, the index must be dropped and recreated.
  Run: python ingest.py --full   (drop_index + ensure_index + index_chunks)

FEW_SHOT EXCLUSION
  FEW_SHOT chunks are Qdrant-only — never indexed in OpenSearch.
  Example matching is a semantic similarity problem, not a keyword problem.
  BM25 misses paraphrase matches ("scripts not marked" vs "unevaluated scripts").

TRANSPORT
  FastMCP 3.x Streamable HTTP (MCP 2025-03-26 spec).
  Tools served at: POST http://<host>:<port>/mcp

STARTUP
  python mcp_servers/opensearch_server.py

CONFIG (.env — all optional, defaults shown)
  MCP_OPENSEARCH_HOST=127.0.0.1
  MCP_OPENSEARCH_PORT=5011
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# ── Add project root to sys.path so config/settings.py resolves correctly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import urllib3
from fastmcp import FastMCP
from opensearchpy import OpenSearch, helpers

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── MCP server instance ────────────────────────────────────────────────────────
mcp = FastMCP(
    name        = "opensearch-bm25",
    instructions = "BM25 keyword search and indexing for NL→SQL schema chunks",
)

# ── Index mapping definition ───────────────────────────────────────────────────
#
# This dict is passed to OpenSearch at index creation time.
# It defines two custom components:
#
# domain_pattern_tokenizer:
#   Splits on any non-word character (\W+). No lowercasing.
#   "FROZEN_STATUS" → ["FROZEN", "STATUS"]   (case preserved)
#   "board_id"      → ["board", "id"]
#
# domain_code_analyzer:
#   Uses domain_pattern_tokenizer with no additional filters.
#   Tokens stay exactly as written — used for the .raw sub-fields.
#
# The built-in "standard" analyser lowercases and removes stop words.
# Used for the main text fields so prose queries work regardless of case.
#
_INDEX_SETTINGS: dict[str, Any] = {
    "settings": {
        "analysis": {
            "analyzer": {
                "domain_code_analyzer": {
                    "type":      "custom",
                    "tokenizer": "domain_pattern_tokenizer",
                    "filter":    []           # no filters — preserve case exactly
                }
            },
            "tokenizer": {
                "domain_pattern_tokenizer": {
                    "type":    "pattern",
                    "pattern": "\\W+",        # split on non-word characters
                    "flags":   "CASE_INSENSITIVE",
                }
            },
        },
        "index": {
            "number_of_shards":   1,   # single shard — corpus is small (~200 chunks)
            "number_of_replicas": 0,   # no replicas — local development
        },
    },
    "mappings": {
        "properties": {
            # ── Exact-match classification fields ──────────────────────────
            # keyword type = not analysed, exact match only, used for filtering
            "chunk_id":          {"type": "keyword"},
            "chunk_type":        {"type": "keyword"},
            "table_name":        {"type": "keyword"},
            "referenced_tables": {"type": "keyword"},
            "domain_tags":       {"type": "keyword"},
            "fk_neighbors":      {"type": "keyword"},
            "schema_version":    {"type": "keyword"},
            "intent":            {"type": "keyword"},

            # ── H5 dual-field text mapping ─────────────────────────────────
            #
            # "text":     standard analyser → lowercased → prose recall
            # "text.raw": domain_code_analyser → case-preserved → code recall
            #
            # search_chunks queries both via multi_match best_fields.
            #
            "text": {
                "type":     "text",
                "analyzer": "standard",
                "fields": {
                    "raw": {
                        "type":     "text",
                        "analyzer": "domain_code_analyzer",
                    }
                },
            },

            # Same dual-field pattern for the NL question field
            # (present on FEW_SHOT chunks — not indexed here but mapping must cover it)
            "nl_question": {
                "type":     "text",
                "analyzer": "standard",
                "fields": {
                    "raw": {
                        "type":     "text",
                        "analyzer": "domain_code_analyzer",
                    }
                },
            },
        }
    },
}

# FEW_SHOT chunks are never indexed in OpenSearch — Qdrant only
_EXCLUDED_TYPES = {"FEW_SHOT"}

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_os_client: OpenSearch | None = None


def _get_client() -> OpenSearch:
    """
    Lazy OpenSearch client — connects on first call, reused for all subsequent calls.

    SSL handling: this deployment uses HTTPS with a self-signed certificate
    (OPENSEARCH_USE_SSL=true, OPENSEARCH_VERIFY_CERTS=false in .env).
    urllib3 InsecureRequestWarning is suppressed — the self-signed cert is
    intentional for local/dev deployment, not a security gap.

    To disable SSL entirely: set OPENSEARCH_USE_SSL=false in .env.
    For production with a real certificate: OPENSEARCH_VERIFY_CERTS=true.
    """
    global _os_client
    if _os_client is None:
        cfg = settings.opensearch

        ssl_kwargs: dict[str, Any] = {}
        if cfg.use_ssl:
            # Suppress "InsecureRequestWarning" for self-signed cert — expected here
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            ssl_kwargs = {
                "ssl_assert_hostname": False,
                "ssl_show_warn":       False,
            }

        _os_client = OpenSearch(
            hosts        = [{"host": cfg.host, "port": cfg.port}],
            http_auth    = (cfg.username, cfg.password),
            use_ssl      = cfg.use_ssl,
            verify_certs = cfg.verify_certs,
            timeout      = 30,
            **ssl_kwargs,
        )
        logger.info(
            component = "opensearch_mcp",
            event     = "client_connected",
            host      = cfg.host,
            port      = cfg.port,
        )
    return _os_client


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_chunks(
    query_text:  str,
    top_k:       int              = 20,
    chunk_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    BM25 keyword search over schema chunks — the keyword half of hybrid retrieval.

    Queries BOTH the standard (lowercase prose) field and the domain_code
    (case-preserving) field via multi_match best_fields. This means:
      - "frozen scripts"  → matches "Frozen evaluation" via text (lowercased)
      - "FROZEN"          → matches "FROZEN" status code via text.raw (preserved)
      - "DEK"             → matches DEK encryption key via text.raw
      - "board evaluation" → matches prose via text field

    FEW_SHOT chunks are always excluded regardless of chunk_types — they are
    Qdrant-only and must not appear in BM25 results.

    Called by retrieval/orchestrator.py in parallel with Qdrant dense search.
    Results are merged with dense results via RRF fusion in the orchestrator.

    Args:
        query_text:  Natural language query string (user's question as typed).
        top_k:       Maximum results to return (default 20).
        chunk_types: Optional ChunkType filter. FEW_SHOT always excluded.
                     e.g. ["TABLE", "WORKFLOW"] to narrow to those types.
                     None = all types except FEW_SHOT.

    Returns:
        List of dicts, each containing chunk payload fields plus:
          "chunk_id": document _id string
          "score":    BM25 relevance score (higher = more relevant)
        Sorted by score descending.
    """
    t0    = time.time()
    index = settings.opensearch.index_name

    # must: the query MUST match — scores are computed against these clauses
    must_clauses = [
        {
            "multi_match": {
                "query":  query_text,
                # H5: query all four fields — prose + domain code for both text fields
                "fields": ["text", "text.raw", "nl_question", "nl_question.raw"],
                "type":   "best_fields",   # score = best single-field match score
            }
        }
    ]

    # filter: post-score narrowing — does not affect relevance scores
    filter_clauses = [
        # Always exclude FEW_SHOT from BM25 — they are Qdrant-only
        {"bool": {"must_not": [{"term": {"chunk_type": "FEW_SHOT"}}]}}
    ]

    if chunk_types:
        # Include only the requested types (after removing FEW_SHOT if present)
        effective = [ct for ct in chunk_types if ct not in _EXCLUDED_TYPES]
        if effective:
            filter_clauses.append({"terms": {"chunk_type": effective}})

    response = _get_client().search(
        index = index,
        body  = {
            "query": {
                "bool": {
                    "must":   must_clauses,
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
        },
    )

    hits = [
        {"chunk_id": h["_id"], "score": h["_score"], **h["_source"]}
        for h in response.get("hits", {}).get("hits", [])
    ]

    logger.info(
        component  = "opensearch_mcp",
        event      = "search_complete",
        query      = query_text[:60],
        top_k      = top_k,
        hits       = len(hits),
        elapsed_ms = round((time.time() - t0) * 1000),
    )
    return hits


@mcp.tool()
def index_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Bulk index semantic chunks into OpenSearch (ingest-time operation).

    FEW_SHOT chunks are silently skipped — they belong in Qdrant only.
    All other chunk types (TABLE, FK_MAP, WORKFLOW, STATUS, GLOSSARY,
    AUDIT, INDEX, PARTITION) are indexed.

    Uses opensearch-py helpers.bulk() which batches requests automatically
    and reports per-item failures without stopping the entire ingest
    (raise_on_error=False). A single malformed document does not abort
    the whole operation.

    Each document uses chunk_id as its _id — upserting the same chunk_id
    twice simply overwrites the document (idempotent).

    Called by ingest.py after DDL parsing and chunk generation, always
    after delete_chunks() has removed any stale versions.

    Args:
        chunks: List of SemanticChunk.to_payload() dicts.
                Must include "chunk_id" (used as _id) and "chunk_type".

    Returns:
        {"indexed": N, "skipped_few_shot": M, "failed": K}
        where N = successfully indexed, M = FEW_SHOT skipped, K = errors.
    """
    if not chunks:
        return {"indexed": 0, "skipped_few_shot": 0, "failed": 0}

    t0 = time.time()

    # Split: OpenSearch gets all types except FEW_SHOT
    eligible         = [c for c in chunks if c.get("chunk_type") not in _EXCLUDED_TYPES]
    skipped_few_shot = len(chunks) - len(eligible)

    if not eligible:
        return {"indexed": 0, "skipped_few_shot": skipped_few_shot, "failed": 0}

    # Build bulk action list — one action per chunk
    actions = [
        {
            "_index":  settings.opensearch.index_name,
            "_id":     chunk["chunk_id"],   # content-addressed — safe to re-index
            "_source": chunk,
        }
        for chunk in eligible
    ]

    # helpers.bulk batches internally and returns (success_count, failed_list)
    success, failed_items = helpers.bulk(
        _get_client(),
        actions,
        raise_on_error = False,   # partial failure is OK — log and continue
        stats_only     = False,   # return failed item details for logging
    )
    failed = len(failed_items) if isinstance(failed_items, list) else 0

    if failed:
        logger.warning(
            component = "opensearch_mcp",
            event     = "bulk_index_partial_failure",
            success   = success,
            failed    = failed,
        )

    logger.info(
        component        = "opensearch_mcp",
        event            = "index_complete",
        indexed          = success,
        skipped_few_shot = skipped_few_shot,
        failed           = failed,
        elapsed_ms       = round((time.time() - t0) * 1000),
    )
    return {"indexed": success, "skipped_few_shot": skipped_few_shot, "failed": failed}


@mcp.tool()
def delete_chunks(changed_tables: list[str]) -> dict[str, int]:
    """
    Delete all documents whose referenced_tables contains any changed table.

    Called by ingest.py during INCREMENTAL DDL updates, always BEFORE
    index_chunks(). Without this, stale chunks become permanent orphans:
      - DDL change → chunk text changes → new SHA-256 → new chunk_id
      - Old chunk_id stays in the index with stale schema still searchable

    Uses delete_by_query with OR logic (terms filter): a document is deleted
    if referenced_tables contains ANY of the changed_tables entries.

    Args:
        changed_tables: Table names whose chunks should be removed.
                        e.g. ["answer_script", "evaluation_attempt"]

    Returns:
        {"deleted": N} where N = number of documents removed.
    """
    if not changed_tables:
        return {"deleted": 0}

    result = _get_client().delete_by_query(
        index = settings.opensearch.index_name,
        body  = {
            "query": {
                "terms": {
                    # terms = OR: delete if referenced_tables contains ANY of these
                    "referenced_tables": changed_tables
                }
            }
        },
    )
    deleted = result.get("deleted", 0)

    logger.info(
        component = "opensearch_mcp",
        event     = "chunks_deleted",
        tables    = sorted(changed_tables),
        deleted   = deleted,
    )
    return {"deleted": deleted}


@mcp.tool()
def ensure_index() -> dict[str, str]:
    """
    Create the OpenSearch index with dual-field mapping if it does not exist.

    Called by ingest.py on first run and after drop_index() during --full
    re-ingestion. Uses _INDEX_SETTINGS defined at module level which includes:
      - The domain_code_analyzer for case-preserving tokenisation
      - Dual-field mapping for text and nl_question (H5 fix)
      - 1 shard / 0 replicas for local development

    IMPORTANT — mapping changes require recreation:
      OpenSearch does not allow changing a field's analyser on an existing index.
      If the index already exists with a single-field mapping (before H5 fix),
      you must drop it first and let this recreate it.
      Run: python ingest.py --full   (handles drop + recreate + reindex)

    Safe to call multiple times — returns "exists" without touching anything
    if the index is already present.

    Returns:
        {"status": "created"} or {"status": "exists"}
    """
    index = settings.opensearch.index_name

    if _get_client().indices.exists(index=index):
        logger.info(component="opensearch_mcp", event="index_exists", name=index)
        return {"status": "exists"}

    _get_client().indices.create(index=index, body=_INDEX_SETTINGS)
    logger.info(component="opensearch_mcp", event="index_created", name=index)
    return {"status": "created"}


@mcp.tool()
def drop_index() -> dict[str, str]:
    """
    Drop the OpenSearch index entirely — USE WITH CAUTION.

    All documents are permanently deleted.

    Called by ingest.py --full before recreating the index with the current
    mapping. Two reasons this is needed:
      1. Mapping change (e.g. H5 dual-field fix): OpenSearch cannot change
         a field's analyser in-place — must drop and recreate.
      2. Clean rebuild after major DDL restructuring: removes all orphaned
         chunks in one operation.

    After calling this you MUST call ensure_index() then index_chunks()
    to rebuild. The ingest.py --full flag does all three steps automatically.

    Returns:
        {"status": "dropped"} or {"status": "not_found"}
    """
    index = settings.opensearch.index_name

    if _get_client().indices.exists(index=index):
        _get_client().indices.delete(index=index)
        logger.info(component="opensearch_mcp", event="index_dropped", name=index)
        return {"status": "dropped"}

    return {"status": "not_found"}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = settings.mcp.opensearch_host
    port = settings.mcp.opensearch_port

    logger.info(
        component = "opensearch_mcp",
        event     = "server_starting",
        host      = host,
        port      = port,
    )

    # FastMCP 3.x calls uvicorn internally via run_http_async().
    # You do not call uvicorn.run() directly — FastMCP manages it.
    #
    # transport="http" = Streamable HTTP (MCP 2025-03-26 spec, recommended)
    # transport="sse"  = legacy SSE for older MCP clients
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