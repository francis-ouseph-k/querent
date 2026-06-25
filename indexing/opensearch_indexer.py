"""
indexing/opensearch_indexer.py
───────────────────────────────
Indexes semantic chunks into OpenSearch for BM25 keyword retrieval.

Key design: dual-field text indexing preserves both case-sensitive domain
codes and lowercase prose recall.

FEW_SHOT chunks are NOT indexed into OpenSearch — they live in Qdrant
only, because example retrieval is a semantic similarity problem.

Targeted invalidation: when a DDL change is detected, all documents
whose referenced_tables field contains any changed table are deleted
and replaced with fresh documents.

FIXES IN THIS VERSION
─────────────────────
H5  — BM25 recall was broken for prose queries.  The domain_code_analyzer
      indexes tokens with no lowercase filter (intentional for DEK/KEK/W/P
      status codes).  This means user NL queries like "frozen scripts" or
      "show pending boards" did not match chunk text that starts with a
      capital letter ("Frozen evaluation") or is entirely uppercase ("FROZEN").

      Fix: dual-field text mapping.
        "text"     — standard lowercase analyzer for prose recall
        "text_raw" — domain_code_analyzer for exact case-sensitive code matching
        "nl_question" likewise gets "nl_question_raw"

      The search() method queries both fields via multi_match so a user typing
      "frozen" matches both "frozen" (lowercase tokens) and the status code
      "FROZEN" (domain tokens) — best-of-both without losing exact-code fidelity.

      NOTE: existing index must be dropped and recreated for the mapping change
      to take effect.  Run:  python ingest.py --full
"""

from __future__ import annotations

import time
from typing import Any

from opensearchpy import OpenSearch, helpers

from config.settings import settings
from models.schema import ChunkType, SemanticChunk
from utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Index mapping and analyser configuration ──────────────────────────────────
#
# Two analyzers:
#   domain_code_analyzer  — pattern tokenizer on \W+, no lowercase filter.
#                           Preserves: DEK, KEK, W, P, N, block_status.
#                           Used for text_raw and nl_question_raw fields.
#   standard (built-in)   — standard tokenizer + lowercase filter.
#                           Used for text and nl_question fields.
#                           Matches prose queries regardless of capitalisation.
#
# Search hits both field pairs via multi_match so:
#   query "frozen scripts" → matches "Frozen" (standard) + "FROZEN" (raw)
#   query "DEK"            → matches "DEK" (raw, case-preserved)
_INDEX_SETTINGS: dict[str, Any] = {
    "settings": {
        "analysis": {
            "analyzer": {
                "domain_code_analyzer": {
                    "type":      "custom",
                    "tokenizer": "domain_pattern_tokenizer",
                    "filter":    []
                }
            },
            "tokenizer": {
                "domain_pattern_tokenizer": {
                    "type":    "pattern",
                    "pattern": "\\W+",
                    "flags":   "CASE_INSENSITIVE"
                }
            }
        },
        "index": {
            "number_of_shards":   1,
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            "chunk_id":          {"type": "keyword"},
            "chunk_type":        {"type": "keyword"},
            "table_name":        {"type": "keyword"},
            "referenced_tables": {"type": "keyword"},
            "domain_tags":       {"type": "keyword"},
            "fk_neighbors":      {"type": "keyword"},
            "schema_version":    {"type": "keyword"},
            "intent":            {"type": "keyword"},
            # H5: prose field — standard lowercase analyzer for NL query recall
            "text": {
                "type":     "text",
                "analyzer": "standard",
                # raw sub-field — domain_code_analyzer for exact code matching
                "fields": {
                    "raw": {
                        "type":     "text",
                        "analyzer": "domain_code_analyzer",
                    }
                }
            },
            # H5: same dual-field pattern for nl_question
            "nl_question": {
                "type":     "text",
                "analyzer": "standard",
                "fields": {
                    "raw": {
                        "type":     "text",
                        "analyzer": "domain_code_analyzer",
                    }
                }
            },
        }
    }
}


class OpenSearchIndexer:
    """
    Manages the OpenSearch index for BM25 keyword retrieval.

    Usage:
        indexer = OpenSearchIndexer()
        indexer.ensure_index()
        indexer.index_chunks(chunks)
    """

    def __init__(self) -> None:
        self._client: OpenSearch | None = None

    @property
    def client(self) -> OpenSearch:
        if self._client is None:
            cfg = settings.opensearch

            ssl_kwargs: dict = {}
            if cfg.use_ssl:
                ssl_kwargs = {
                    "ssl_assert_hostname": False,
                    "ssl_show_warn":       False,
                }
                import urllib3 as _urllib3
                _urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)

            logger.warning(
                component    = "opensearch_ingestor",
                event        = "client_creation",
                host         = cfg.host,
                port         = cfg.port,
                use_ssl      = cfg.use_ssl,
                verify_certs = cfg.verify_certs,
            )

            self._client = OpenSearch(
                hosts        = [{"host": cfg.host, "port": cfg.port}],
                http_auth    = (cfg.username, cfg.password),
                use_ssl      = cfg.use_ssl,
                verify_certs = cfg.verify_certs,
                timeout      = 30,
                **ssl_kwargs,
            )
        return self._client

    def ensure_index(self) -> None:
        """Create the index with dual-field mapping if it does not exist."""
        index_name = settings.opensearch.index_name
        if self.client.indices.exists(index=index_name):
            logger.info(component="opensearch_indexer", event="index_exists", name=index_name)
            return

        self.client.indices.create(index=index_name, body=_INDEX_SETTINGS)
        logger.info(component="opensearch_indexer", event="index_created", name=index_name)

    def index_chunks(self, chunks: list[SemanticChunk]) -> int:
        """
        Bulk index chunks into OpenSearch.
        FEW_SHOT chunks are skipped — they live in Qdrant only.
        Returns the number of documents indexed.
        """
        eligible = [c for c in chunks if c.chunk_type != ChunkType.FEW_SHOT]
        if not eligible:
            return 0

        t0         = time.time()
        index_name = settings.opensearch.index_name

        actions = [
            {
                "_index":  index_name,
                "_id":     c.chunk_id,
                "_source": c.to_payload(),
            }
            for c in eligible
        ]

        success, failed = helpers.bulk(
            self.client,
            actions,
            raise_on_error=False,
            stats_only=False,
        )

        if failed:
            logger.warning(
                component="opensearch_indexer",
                event="bulk_index_partial_failure",
                success=success,
                failed=len(failed),
            )

        elapsed = (time.time() - t0) * 1000
        logger.info(
            component="opensearch_indexer",
            event="index_complete",
            count=success,
            elapsed_ms=round(elapsed),
        )
        return success

    def delete_chunks_for_tables(self, changed_tables: set[str]) -> int:
        """
        Delete all documents whose referenced_tables contains any changed table.
        """
        if not changed_tables:
            return 0

        index_name = settings.opensearch.index_name
        query = {
            "query": {
                "terms": {
                    "referenced_tables": list(changed_tables)
                }
            }
        }

        result  = self.client.delete_by_query(index=index_name, body=query)
        deleted = result.get("deleted", 0)

        logger.info(
            component="opensearch_indexer",
            event="chunks_deleted",
            tables=sorted(changed_tables),
            deleted=deleted,
        )
        return deleted

    def search(
        self,
        query_text:  str,
        top_k:       int              = 20,
        chunk_types: list[ChunkType]  | None = None,
    ) -> list[dict[str, Any]]:
        """
        BM25 keyword search using both prose and domain-code fields.

        H5: queries text (standard/lowercase) and text.raw (domain_code_analyzer)
        so NL prose queries and exact domain code queries both recall correctly.
        """
        index_name = settings.opensearch.index_name

        must_clauses: list[dict] = [
            {
                "multi_match": {
                    "query":  query_text,
                    # H5: include both lowercase prose fields and raw code fields
                    "fields": ["text", "text.raw", "nl_question", "nl_question.raw"],
                    "type":   "best_fields",
                }
            }
        ]

        filter_clauses: list[dict] = []

        if chunk_types:
            filter_clauses.append({
                "terms": {
                    "chunk_type": [ct.value for ct in chunk_types]
                }
            })

        # Exclude FEW_SHOT from BM25 search
        filter_clauses.append({
            "bool": {
                "must_not": [
                    {"term": {"chunk_type": ChunkType.FEW_SHOT.value}}
                ]
            }
        })

        body = {
            "query": {
                "bool": {
                    "must":   must_clauses,
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
        }

        response = self.client.search(index=index_name, body=body)
        hits     = response.get("hits", {}).get("hits", [])

        return [
            {
                "chunk_id": hit["_id"],
                "score":    hit["_score"],
                **hit["_source"],
            }
            for hit in hits
        ]