"""
mcp_tools/client.py
────────────────────
Thin HTTP client for calling the four local MCP servers.

Provides two drop-in replacement classes:
  QdrantMCPClient      — same interface as QdrantIndexer
  OpenSearchMCPClient  — same interface as OpenSearchIndexer

And standalone helpers:
  call_postgres_execute()    — execute_query via postgres_server
  call_postgres_explain()    — explain_query via postgres_server
  call_corpus_log_failure()  — log_failure via corpus_server
  call_corpus_save_correction() — save_correction via corpus_server

HOW TO SWITCH
  Set USE_MCP_SERVERS=true in .env.
  PipelineRunner, SQLValidator, and ingest.py check settings.use_mcp_servers
  and instantiate either the direct client or the MCP client.

WIRE PROTOCOL (FastMCP 3.x, stateless_http=True, json_response=True)
  Request:  POST <base_url>/mcp
  Headers:  Content-Type: application/json
            Accept: application/json, text/event-stream  (required by FastMCP 3.x)
  Body:     {"jsonrpc":"2.0","id":1,"method":"tools/call",
             "params":{"name":"<tool>","arguments":{...}}}

  Success response:
    {"jsonrpc":"2.0","id":1,"result":{
        "content":[{"type":"text","text":"<json_string>"}],
        "structuredContent": <parsed_object>,
        "isError": false
    }}
    We use content[0].text when present, structuredContent as fallback.

  Error response (tool raised an exception):
    {"jsonrpc":"2.0","id":1,"result":{
        "content":[{"type":"text","text":"Error calling tool '...': <message>"}],
        "isError": true
    }}
    We raise MCPCallError with the text message.

ERROR HANDLING
  Network errors  → MCPCallError raised
  Tool isError    → MCPCallError raised with tool error message
  HTTP 4xx/5xx    → MCPCallError raised

FIXES IN THIS VERSION
─────────────────────
FIX-T1 — _call() timeout raised from 60s to 180s.
          LLM correction prompts (retry path in RetryValidator) can take
          90–120s on complex queries (observed: elapsed_ms=93085).  The
          original 60s timeout caused the MCP call to abort mid-inference,
          masking the real validation error with a spurious MCPCallError.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from config.settings import settings
from models.schema import ChunkType, SemanticChunk
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MCPCallError(Exception):
    """Raised when an MCP server call fails at the network or protocol level."""
    pass


# Headers required by FastMCP 3.x Streamable HTTP.
# Accept must include both types or the server returns 406 Not Acceptable.
_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept":       "application/json, text/event-stream",
}

# FIX-T1: raised from 60s to 180s — LLM correction prompts can take 90–120s.
# upsert_chunks on first call (model load) can also be slow.
_MCP_TIMEOUT = 180.0


def _call(base_url: str, tool: str, arguments: dict) -> Any:
    """
    Make one MCP tool call and return the parsed result.

    Uses FastMCP 3.x stateless Streamable HTTP with json_response=True.

    Response parsing:
      - Success with content: returns parsed content[0].text (JSON string).
      - Success with empty content[]: falls back to structuredContent.
        FastMCP returns empty content[] for empty result sets (e.g. Qdrant
        search with 0 hits: {"result": []}). structuredContent is still
        populated in this case.
      - Tool error (isError=true): raises MCPCallError with the error text.
      - HTTP error: raises MCPCallError with status code and body.
      - Network error: raises MCPCallError with connection details.

    Args:
        base_url:  Server base URL, e.g. "http://127.0.0.1:5010"
        tool:      Tool name matching the @mcp.tool() function name
        arguments: Dict of tool arguments

    Returns:
        Parsed tool return value (dict, list, or scalar).

    Raises:
        MCPCallError on any failure.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params":  {"name": tool, "arguments": arguments},
    }

    try:
        response = httpx.post(
            f"{base_url}/mcp",
            json    = payload,
            headers = _MCP_HEADERS,
            timeout = _MCP_TIMEOUT,   # FIX-T1: 180s — covers LLM correction prompt latency
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise MCPCallError(
            f"MCP server {base_url} returned HTTP {exc.response.status_code} "
            f"for tool '{tool}': {exc.response.text[:300]}"
        ) from exc
    except httpx.RequestError as exc:
        raise MCPCallError(
            f"Cannot reach MCP server at {base_url} for tool '{tool}': {exc}"
        ) from exc

    # Parse FastMCP 3.x JSON response envelope
    try:
        body   = response.json()
        result = body.get("result", {})
    except Exception as exc:
        raise MCPCallError(
            f"Non-JSON response from {base_url} for tool '{tool}': "
            f"{response.text[:300]}"
        ) from exc

    # Tool raised an exception server-side — isError=true, content[0].text is the message
    if result.get("isError"):
        content = result.get("content", [])
        msg     = content[0].get("text", "unknown tool error") if content else "unknown tool error"
        raise MCPCallError(f"Tool '{tool}' on {base_url} raised an error: {msg}")

    # Prefer content[0].text — plain JSON string, reliable across all FastMCP versions.
    # Fall back to structuredContent when content[] is empty (FastMCP returns empty
    # content[] for empty result sets, e.g. Qdrant 0-hit search returns {"result": []}).
    content = result.get("content", [])
    if content:
        try:
            content_text = content[0]["text"]
            try:
                return json.loads(content_text)
            except json.JSONDecodeError:
                # Plain string return value — return as-is
                return content_text
        except (KeyError, IndexError) as exc:
            raise MCPCallError(
                f"Unexpected MCP response from {base_url} for tool '{tool}': "
                f"{response.text[:300]}"
            ) from exc

    # content[] empty — use structuredContent directly.
    # FastMCP wraps list returns as {"result": [...]} and dicts directly.
    structured = result.get("structuredContent")
    if structured is not None:
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured

    raise MCPCallError(
        f"Unexpected MCP response from {base_url} for tool '{tool}': "
        f"{response.text[:300]}"
    )


# ── Qdrant MCP client — same interface as QdrantIndexer ───────────────────────

class QdrantMCPClient:
    """
    Drop-in replacement for QdrantIndexer that routes all calls through
    the qdrant_server.py MCP server (default port 5010).

    Method signatures are identical to QdrantIndexer so the orchestrator,
    ingest.py, and any other caller can swap instances without other changes.

    The embedding model lives in qdrant_server.py — this client does NOT
    load BGE-small-en-v1.5. Embedding happens server-side, once per process.
    """

    def __init__(self) -> None:
        self._base = (
            f"http://{settings.mcp.qdrant_host}:{settings.mcp.qdrant_port}"
        )
        logger.info(component="qdrant_mcp_client", event="init", base_url=self._base)

    def _call(self, tool: str, **kwargs) -> Any:
        return _call(self._base, tool, kwargs)

    # ── Collection management (ingest-time) ───────────────────────────────

    def ensure_collection(self) -> None:
        """Create collection if absent. Mirrors QdrantIndexer.ensure_collection()."""
        result = self._call("ensure_collection")
        logger.info(component="qdrant_mcp_client", event="ensure_collection",
                    status=result.get("status"))

    def drop_collection(self) -> None:
        """Drop collection entirely. Used by ingest.py --full."""
        result = self._call("drop_collection")
        logger.info(component="qdrant_mcp_client", event="drop_collection",
                    status=result.get("status"))

    # ── Chunk indexing (ingest-time) ──────────────────────────────────────

    def upsert_chunks(self, chunks: list[SemanticChunk]) -> int:
        """
        Embed and upsert chunks via the MCP server.
        Mirrors QdrantIndexer.upsert_chunks() — accepts SemanticChunk objects.
        Returns number of chunks upserted.
        """
        if not chunks:
            return 0
        # Convert SemanticChunk objects to payload dicts for JSON transport
        payloads = [c.to_payload() for c in chunks]
        result   = self._call("upsert_chunks", chunks=payloads)
        return result.get("upserted", 0)

    def delete_chunks_for_tables(self, changed_tables: set[str]) -> int:
        """
        Delete stale chunks for changed tables.
        Mirrors QdrantIndexer.delete_chunks_for_tables().
        Returns number of chunks deleted.
        """
        result = self._call("delete_chunks", changed_tables=list(changed_tables))
        return result.get("deleted", 0)

    # ── Search (query-time) ───────────────────────────────────────────────

    def search(
        self,
        query_text:     str,
        top_k:          int                      = 20,
        filter_payload: dict | None              = None,
        chunk_types:    list[ChunkType] | None   = None,
    ) -> list[dict[str, Any]]:
        """
        Dense vector search via MCP server.
        Mirrors QdrantIndexer.search().

        Note: chunk_types accepts ChunkType enum objects (same as QdrantIndexer)
        — they are converted to string values for JSON transport.
        """
        result = self._call(
            "search_chunks",
            query_text     = query_text,
            top_k          = top_k,
            chunk_types    = [ct.value for ct in chunk_types] if chunk_types else None,
            filter_payload = filter_payload,
        )
        return result if isinstance(result, list) else []

    def get_few_shot_examples(
        self,
        query_text: str,
        top_k:      int = 3,
    ) -> list[dict[str, Any]]:
        """
        FEW_SHOT example retrieval via MCP server.
        Mirrors QdrantIndexer.get_few_shot_examples().
        """
        result = self._call(
            "get_few_shot_examples",
            query_text = query_text,
            top_k      = top_k,
        )
        return result if isinstance(result, list) else []


# ── OpenSearch MCP client — same interface as OpenSearchIndexer ───────────────

class OpenSearchMCPClient:
    """
    Drop-in replacement for OpenSearchIndexer that routes all calls through
    the opensearch_server.py MCP server (default port 5011).

    Method signatures are identical to OpenSearchIndexer.
    """

    def __init__(self) -> None:
        self._base = (
            f"http://{settings.mcp.opensearch_host}:{settings.mcp.opensearch_port}"
        )
        logger.info(component="opensearch_mcp_client", event="init",
                    base_url=self._base)

    def _call(self, tool: str, **kwargs) -> Any:
        return _call(self._base, tool, kwargs)

    # ── Index management (ingest-time) ────────────────────────────────────

    def ensure_index(self) -> None:
        """Create index with dual-field mapping if absent."""
        result = self._call("ensure_index")
        logger.info(component="opensearch_mcp_client", event="ensure_index",
                    status=result.get("status"))

    def drop_index(self) -> None:
        """Drop index entirely. Used by ingest.py --full."""
        result = self._call("drop_index")
        logger.info(component="opensearch_mcp_client", event="drop_index",
                    status=result.get("status"))

    def index_chunks(self, chunks: list[SemanticChunk]) -> int:
        """
        Bulk index chunks via MCP server (FEW_SHOT skipped server-side).
        Mirrors OpenSearchIndexer.index_chunks().
        Returns number of chunks indexed.
        """
        if not chunks:
            return 0
        payloads = [c.to_payload() for c in chunks]
        result   = self._call("index_chunks", chunks=payloads)
        if result.get("failed", 0):
            logger.warning(component="opensearch_mcp_client",
                           event="partial_failure", failed=result["failed"])
        return result.get("indexed", 0)

    def delete_chunks_for_tables(self, changed_tables: set[str]) -> int:
        """
        Delete stale chunks for changed tables.
        Mirrors OpenSearchIndexer.delete_chunks_for_tables().
        Returns number of documents deleted.
        """
        result = self._call("delete_chunks", changed_tables=list(changed_tables))
        return result.get("deleted", 0)

    # ── Search (query-time) ───────────────────────────────────────────────

    def search(
        self,
        query_text:  str,
        top_k:       int                      = 20,
        chunk_types: list[ChunkType] | None   = None,
    ) -> list[dict[str, Any]]:
        """
        BM25 keyword search via MCP server.
        Mirrors OpenSearchIndexer.search().
        """
        result = self._call(
            "search_chunks",
            query_text  = query_text,
            top_k       = top_k,
            chunk_types = [ct.value for ct in chunk_types] if chunk_types else None,
        )
        return result if isinstance(result, list) else []


# ── PostgreSQL MCP helpers ─────────────────────────────────────────────────────

def call_postgres_execute(
    sql:      str,
    user_id:  str | None = None,
    max_rows: int        = 1000,
) -> dict[str, Any]:
    """
    Execute a validated SELECT via postgres_server.py MCP (port 5012).

    Drop-in for the psycopg2 pool calls in pipeline/runner.py._execute().
    Returns the same dict shape: {"rows": [...], "row_count": N, "elapsed_ms": M}
    or {"error": "...", "elapsed_ms": M} on failure.

    Raises MCPCallError on network/protocol failure (caller should treat as
    infrastructure failure and route to _failure_result()).
    """
    base = f"http://{settings.mcp.postgres_host}:{settings.mcp.postgres_port}"
    return _call(base, "execute_query", {
        "sql":      sql,
        "user_id":  user_id,
        "max_rows": max_rows,
    })


def call_postgres_explain(sql: str) -> dict[str, Any]:
    """
    Run EXPLAIN (FORMAT JSON) via postgres_server.py MCP (port 5012).

    Drop-in for the psycopg2 EXPLAIN call in sql_validator.py._step_cost().
    Returns {"total_cost": float, "elapsed_ms": int}
    or {"error": "...", "pgcode": "...", "elapsed_ms": int} on failure.

    Raises MCPCallError on network/protocol failure.
    """
    base = f"http://{settings.mcp.postgres_host}:{settings.mcp.postgres_port}"
    return _call(base, "explain_query", {"sql": sql})


# ── Corpus MCP helpers ─────────────────────────────────────────────────────────

def call_corpus_log_failure(
    nl_query:   str,
    failed_sql: str,
    error:      str,
    retries:    int = 0,
) -> dict[str, str]:
    """
    Log a failed query to the corpus via corpus_server.py MCP (port 5013).

    Drop-in for pipeline/runner.py._log_failure() local file write.
    Returns {"id": "<entry_id>", "path": "<filename>"}
    or raises MCPCallError on network failure.
    """
    base = f"http://{settings.mcp.corpus_host}:{settings.mcp.corpus_port}"
    return _call(base, "log_failure", {
        "nl_query":   nl_query,
        "failed_sql": failed_sql,
        "error":      error,
        "retries":    retries,
    })


def call_corpus_save_correction(
    entry_id:      str,
    corrected_sql: str,
    source:        str = "user_correction",
) -> dict[str, Any]:
    """
    Save a correction via corpus_server.py MCP (port 5013).

    Drop-in for cli/interface.py._handle_correction() local file write.
    Returns {"status": "ok"|"not_found", "id": entry_id}
    or raises MCPCallError on network failure.
    """
    base = f"http://{settings.mcp.corpus_host}:{settings.mcp.corpus_port}"
    return _call(base, "save_correction", {
        "entry_id":      entry_id,
        "corrected_sql": corrected_sql,
        "source":        source,
    })