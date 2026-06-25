"""
ingest.py
──────────
One-time (and on-DDL-change) ingestion entry point.

Run this before using the NL→SQL system for the first time,
and again whenever the DDL changes.

What it does:
    1. Parse the DDL into TableInventory objects
    2. Build the NetworkX FK graph
    3. Detect which tables changed (DDL hash comparison)
    4. Generate semantic chunks for changed tables only (or all on first run)
    5. Delete stale chunks from Qdrant + OpenSearch
    6. Embed + index new chunks into Qdrant + OpenSearch
    7. Update the stored DDL hash state

Usage:
    python ingest.py                  # incremental (changed tables only)
    python ingest.py --full           # force full re-ingestion
    python ingest.py --dry-run        # show what would be done, don't write
    python ingest.py --warm-reranker  # pre-download the cross-encoder
                                       # reranker model only — does not run
                                       # ingestion. Use this once during
                                       # deployment if RERANKER_ENABLED=true
                                       # is planned, to avoid a first-query
                                       # latency spike from a cold HF Hub
                                       # download in production.

FIXES IN THIS VERSION
─────────────────────
H4  — --full re-ingestion previously skipped chunk deletion (the deletion
    block was guarded by `if not is_first_run and not full`).  With content-
    addressed chunk IDs (SHA-256 of text), any text change produces a new ID
    and the old chunk becomes an orphan — still retrievable, carrying stale
    schema.  Fix: on --full, drop and recreate both Qdrant collection and
    OpenSearch index before re-indexing so no orphans survive.

LOW — removed dead `from tarfile import CompressionError` import.
LOW — traceback.print_exc() replaced with logger.exception().
"""

from __future__ import annotations

import argparse
import json
import re as _re
import sys
from pathlib import Path

from config.settings import settings
from indexing.opensearch_indexer import OpenSearchIndexer
from indexing.qdrant_indexer import QdrantIndexer
from ingestion.chunk_generator import ChunkGenerator
from ingestion.ddl_parser import DDLParser
from ingestion.graph_builder import GraphBuilder
from utils.logging_config import configure_logging, get_logger
from utils.schema_versioning import detect_changed_tables, update_stored_state

configure_logging(settings.log_dir)
logger = get_logger(__name__)


def generate_query_understanding_data(
    tables: dict,
    output_path: str = "data/query_understanding.json",
) -> None:
    """
    FIX H2+H3 — auto-generate table keywords and status codes from the
    parsed DDL and write them to data/query_understanding.json.

    Table keywords: derived from table names (split on underscore + common
    aliases), table comments, and any glossary aliases.

    Status codes: extracted from CHECK constraints in column definitions
    by scanning for IN (...) patterns with uppercase single-quoted values.
    """
    table_keywords: dict[str, str] = {}
    status_codes:   set[str]       = set()

    for table_name, inv in tables.items():
        parts = table_name.replace("_cache", "").replace("_summary", "").split("_")
        for part in parts:
            if len(part) > 2:
                table_keywords[part]       = table_name
                table_keywords[part + "s"] = table_name

        clean = (
            table_name
            .replace("_cache", "")
            .replace("_summary", "")
            .replace("_request", "")
            .replace("_mapping", "")
            .replace("_history", "")
            .replace("_log", "")
        )
        if clean != table_name:
            table_keywords[clean.replace("_", " ")] = table_name
        table_keywords[table_name.replace("_", " ")] = table_name

        for col_name, col in inv.columns.items():
            comment = inv.column_comments.get(col_name, "")
            for m in _re.finditer(r"'([A-Z][A-Z0-9_]+)'", comment):
                status_codes.add(m.group(1))

    status_codes.update([
        "W", "P", "N", "THIRD", "PRIMARY", "REVIEW", "REVAL",
        "ASSIGNED", "IN_PROGRESS", "FROZEN", "SUBMITTED", "CLOSED",
        "NOT_ASSIGNED", "SCANNED", "NOT_SCANNED", "RESCAN_NEEDED",
        "NONE", "ABSENT", "BLOCKED", "MALPRACTICE", "NOT_ELIGIBLE",
        "ACTIVE", "EXPIRED", "REVOKED", "RETIRED", "CLOUD", "WORKSTATION",
        "PENDING", "APPROVED", "REJECTED", "EXPORTED",
        "ATTEMPTED", "ELIGIBLE", "ADMITTED", "BARRED",
    ])

    output = {
        "table_keywords": table_keywords,
        "status_codes":   sorted(status_codes),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Query understanding data: {len(table_keywords)} keywords, "
          f"{len(status_codes)} status codes → {output_path}")


def warm_reranker_cache() -> None:
    """
    REVIEW SUGGESTION: pre-download the cross-encoder reranker model
    (cross-encoder/ms-marco-MiniLM-L-6-v2) so it's cached locally before
    first use, rather than downloading on the first query that triggers
    reranking.

    Why this is a separate flag, not part of the default ingestion flow:
    the reranker is an optional Phase 1+ component (RERANKER_ENABLED
    defaults to false — see retrieval/reranker.py and .env). Most
    deployments never enable it. Folding a HuggingFace Hub download into
    every `python ingest.py` run would add network dependency and latency
    to a step that has nothing to do with reranking, for the common case
    where the model is never used. --warm-reranker is opt-in for
    deployments that have already decided to enable reranking and want to
    avoid the first-query latency spike (download + load) in production.

    Safe to run regardless of RERANKER_ENABLED — it only downloads and
    caches the model; it does not change any setting or touch Qdrant/
    OpenSearch. The HuggingFace Hub cache (~/.cache/huggingface by default,
    or HF_HOME if set) is shared with any other code in this project that
    loads the same model name, so this download is reused by reranker.py at
    runtime without re-fetching.
    """
    from config.settings import settings as _settings

    model_name = _settings.reranker.model_name
    print(f"Pre-downloading reranker model: {model_name}")
    print("(22 MB, CPU-only — this only needs to run once per environment)\n")

    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder(model_name)
    except Exception as exc:
        logger.error(component="ingest", event="reranker_warmup_failed", error=str(exc))
        print(f"\n✗  Reranker pre-download failed: {exc}")
        print("   Check network access to huggingface.co, or pre-populate")
        print("   HF_HOME from an environment that does have access.")
        raise SystemExit(1)

    print(f"[OK] Reranker model cached. Future loads (including the first")
    print(f"   query with RERANKER_ENABLED=true) will read from local cache.")
    logger.info(component="ingest", event="reranker_warmup_complete", model=model_name)


def run_ingestion(full: bool = False, dry_run: bool = False) -> None:
    """
    Execute the full ingestion pipeline.

    full=True  — re-ingest all chunks regardless of DDL changes
    dry_run    — parse and show stats without writing to any index
    """
    ddl_path = Path(settings.ddl_path)
    if not ddl_path.exists():
        logger.error(component="ingest", event="ddl_not_found", path=str(ddl_path))
        print(f"ERROR: DDL file not found at {ddl_path}")
        sys.exit(1)

    ddl_text = ddl_path.read_text(encoding="utf-8")
    logger.info(component="ingest", event="start", path=str(ddl_path), bytes=len(ddl_text))

    # ── Step 1: Parse DDL ─────────────────────────────────────────────────
    print("Parsing DDL…")
    parser = DDLParser()
    tables = parser.parse(ddl_text)
    print(f"  {len(tables)} tables parsed.")

    # ── Step 2: Detect changes ────────────────────────────────────────────
    changed_tables, ddl_hash, is_first_run = detect_changed_tables(
        ddl_text  = ddl_text,
        hash_path = settings.schema_hash_path,
        tables    = tables,
    )

    if not changed_tables and not full:
        print("[OK] No DDL changes detected. Ingestion not required.")
        logger.info(component="ingest", event="no_changes")
        return

    if full:
        print("-> Full re-ingestion requested.")
    elif is_first_run:
        print(f"-> First run — ingesting all {len(changed_tables)} tables.")
    else:
        print(f"-> {len(changed_tables)} table(s) changed: {sorted(changed_tables)}")

    # ── Step 3: Build FK graph ────────────────────────────────────────────
    print("Building FK graph…")
    gb    = GraphBuilder()
    graph = gb.build(tables)
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    if not dry_run:
        import networkx as nx
        graph_path = Path("data/fk_graph.json")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(
            json.dumps(nx.node_link_data(graph), ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  FK graph saved to {graph_path}")

    # ── Step 4: Generate chunks ───────────────────────────────────────────
    print("Generating semantic chunks…")
    target_tables = tables if full else {
        k: v for k, v in tables.items() if k in changed_tables
    }

    gen    = ChunkGenerator(schema_version=ddl_hash[:12])
    chunks = gen.generate(
        tables        = target_tables,
        full_tables   = tables,
        glossary_path = settings.glossary_path,
        examples_path = "data/few_shot_examples.json",
        rules_path    = "config/heuristics.yaml",
    )

    from collections import Counter
    from models.schema import ChunkType
    type_counts = Counter(c.chunk_type.value for c in chunks)
    print(f"  {len(chunks)} chunks generated:")
    for ct in ChunkType:
        if type_counts.get(ct.value):
            print(f"    {ct.value:12s} {type_counts[ct.value]}")

    if dry_run:
        print("\nDry run — no changes written to Qdrant or OpenSearch.")
        return

    # ── Step 5: Delete stale chunks ───────────────────────────────────────
    # Instantiate indexers — MCP clients or direct clients depending on settings
    if settings.use_mcp_servers:
        from mcp_tools.client import QdrantMCPClient, OpenSearchMCPClient
        qdrant_indexer     = QdrantMCPClient()
        opensearch_indexer = OpenSearchMCPClient()
        print("  Using MCP servers for indexing (USE_MCP_SERVERS=true).")
    else:
        qdrant_indexer     = QdrantIndexer()
        opensearch_indexer = OpenSearchIndexer()

    if full:
        # FIX-H4: on --full, drop and recreate both stores so no orphaned chunks
        # survive.  Content-addressed chunk IDs mean any text change produces a
        # new ID; the old chunk stays in the index indefinitely under the old ID
        # and is still retrievable — stale schema in prompts.
        print("Full re-ingestion: dropping existing index/collection…")
        try:
            if settings.use_mcp_servers:
                qdrant_indexer.drop_collection()
            else:
                col_name = settings.qdrant.collection_name
                qdrant_indexer.client.delete_collection(col_name)
            print(f"  Qdrant collection dropped.")
        except Exception as exc:
            logger.warning(component="ingest", event="qdrant_drop_failed", error=str(exc))

        try:
            if settings.use_mcp_servers:
                opensearch_indexer.drop_index()
            else:
                idx_name = settings.opensearch.index_name
                if opensearch_indexer.client.indices.exists(index=idx_name):
                    opensearch_indexer.client.indices.delete(index=idx_name)
            print(f"  OpenSearch index dropped.")
        except Exception as exc:
            logger.warning(component="ingest", event="opensearch_drop_failed", error=str(exc))

    elif not is_first_run:
        print("Deleting stale chunks…")
        deleted_q  = qdrant_indexer.delete_chunks_for_tables(changed_tables)
        deleted_os = opensearch_indexer.delete_chunks_for_tables(changed_tables)
        print(f"  Deleted {deleted_q} from Qdrant, {deleted_os} from OpenSearch.")

    # ── Step 6: Index new chunks ──────────────────────────────────────────
    print("Indexing into Qdrant (dense vectors)…")
    qdrant_indexer.ensure_collection()
    qdrant_count = qdrant_indexer.upsert_chunks(chunks)
    print(f"  {qdrant_count} chunks upserted into Qdrant.")

    print("Indexing into OpenSearch (BM25)…")
    opensearch_indexer.ensure_index()
    opensearch_chunks = [c for c in chunks if c.chunk_type != ChunkType.FEW_SHOT]
    few_shot_excluded = len(chunks) - len(opensearch_chunks)
    os_count = opensearch_indexer.index_chunks(opensearch_chunks)
    print(f"  {os_count} chunks indexed into OpenSearch.  "
          f"({few_shot_excluded} FEW_SHOT chunks excluded — Qdrant only)")

    # ── Step 7: Update hash state ─────────────────────────────────────────
    update_stored_state(ddl_text, settings.schema_hash_path, tables)

    print("Generating query understanding data…")
    generate_query_understanding_data(tables)

    print(f"\n[OK] Ingestion complete. Schema hash: {ddl_hash[:12]}")
    logger.info(component="ingest", event="complete", chunks=len(chunks), hash=ddl_hash[:12])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL→SQL Schema Ingestion")
    parser.add_argument("--full",    action="store_true", help="Force full re-ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not write")
    parser.add_argument("--warm-reranker", action="store_true",
                         help="Pre-download the cross-encoder reranker model and exit "
                              "(does not run ingestion). Use once during deployment if "
                              "RERANKER_ENABLED=true is planned.")
    args = parser.parse_args()

    if args.warm_reranker:
        warm_reranker_cache()
        sys.exit(0)

    run_ingestion(full=args.full, dry_run=args.dry_run)