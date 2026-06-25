"""
indexing/qdrant_indexer.py
───────────────────────────
Embeds semantic chunks using BGE-small-en-v1.5 and upserts them into Qdrant.
Also handles targeted invalidation — deleting only the chunks whose
referenced_tables overlap with the set of changed tables.

FEW_SHOT chunks are indexed into Qdrant only (not OpenSearch).
All other chunk types go to both Qdrant and OpenSearch.
"""

from __future__ import annotations

import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from config.settings import settings
from models.schema import ChunkType, SemanticChunk
from utils.logging_config import get_logger

logger = get_logger(__name__)


class QdrantIndexer:
    """
    Manages Qdrant collection and chunk upserts.

    Usage:
        indexer = QdrantIndexer()
        indexer.ensure_collection()
        indexer.upsert_chunks(chunks)
    """

    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._embedder: SentenceTransformer | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                host=settings.qdrant.host,
                port=settings.qdrant.port,
                timeout=30,
            )
        return self._client

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info(component="qdrant_indexer", event="loading_embedding_model",
                        model=settings.embedding.model_name)
            self._embedder = SentenceTransformer(
                settings.embedding.model_name,
                device=settings.embedding.device,
            )
        return self._embedder

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        col_name = settings.qdrant.collection_name
        existing = {c.name for c in self.client.get_collections().collections}

        if col_name not in existing:
            self.client.create_collection(
                collection_name=col_name,
                vectors_config=qmodels.VectorParams(
                    size=settings.qdrant.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(component="qdrant_indexer", event="collection_created", name=col_name)
        else:
            logger.info(component="qdrant_indexer", event="collection_exists", name=col_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches. Returns list of float vectors."""
        batch_size = settings.embedding.batch_size
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch   = texts[i : i + batch_size]
            vectors = self.embedder.encode(
                batch,
                normalize_embeddings=True,   # cosine similarity requires normalised vectors
                show_progress_bar=False,
            ).tolist()
            all_vectors.extend(vectors)

        return all_vectors

    def upsert_chunks(self, chunks: list[SemanticChunk]) -> int:
        """
        Embed and upsert chunks into Qdrant.
        Returns the number of chunks upserted.
        """
        if not chunks:
            return 0

        t0       = time.time()
        col_name = settings.qdrant.collection_name
        texts    = [c.text for c in chunks]
        vectors  = self.embed_texts(texts)

        points = [
            qmodels.PointStruct(
                # FIX #7 (Low) — original code converted UUID to int via
                # int % 2**63 which risks birthday-paradox collisions at scale
                # and adds unnecessary complexity. Qdrant natively supports
                # UUID strings as point IDs — use them directly.
                id      = c.chunk_id,
                vector  = vectors[i],
                payload = c.to_payload(),
            )
            for i, c in enumerate(chunks)
        ]

        # Upsert in batches of 100 to avoid request size limits
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=col_name,
                points=points[i : i + batch_size],
                wait=True,
            )

        elapsed = (time.time() - t0) * 1000
        logger.info(
            component="qdrant_indexer",
            event="upsert_complete",
            count=len(chunks),
            elapsed_ms=round(elapsed),
        )
        return len(chunks)

    def delete_chunks_for_tables(self, changed_tables: set[str]) -> int:
        """
        Delete all chunks whose referenced_tables overlap with changed_tables.
        Called on DDL change for targeted re-indexing.
        Returns the number of points deleted.
        """
        if not changed_tables:
            return 0

        col_name = settings.qdrant.collection_name

        # Qdrant filter: payload.referenced_tables contains any changed table
        filter_condition = qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="referenced_tables",
                    match=qmodels.MatchValue(value=table),
                )
                for table in changed_tables
            ]
        )

        result = self.client.delete(
            collection_name=col_name,
            points_selector=qmodels.FilterSelector(filter=filter_condition),
            wait=True,
        )

        count = getattr(result, "deleted_count", 0)
        logger.info(
            component="qdrant_indexer",
            event="chunks_deleted",
            tables=sorted(changed_tables),
            deleted=count,
        )
        return count

    def search(
        self,
        query_text: str,
        top_k:      int              = 20,
        filter_payload: dict | None  = None,
        chunk_types: list[ChunkType] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dense vector search. Returns a list of payload dicts with scores.
        chunk_types: if provided, restricts results to those chunk types.
        """
        col_name = settings.qdrant.collection_name
        vector   = self.embed_texts([query_text])[0]

        qdrant_filter = None
        conditions    = []

        if chunk_types:
            conditions.append(
                qmodels.Filter(
                    should=[
                        qmodels.FieldCondition(
                            key="chunk_type",
                            match=qmodels.MatchValue(value=ct.value),
                        )
                        for ct in chunk_types
                    ]
                )
            )

        if filter_payload:
            for k, v in filter_payload.items():
                conditions.append(
                    qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                )

        if conditions:
            qdrant_filter = qmodels.Filter(must=conditions) if len(conditions) > 1 else conditions[0]

        # qdrant-client >= 1.7.0 removed client.search() in favour of
        # client.query_points().  Two API differences from the old call:
        #   - parameter renamed: query_vector= → query=
        #   - return type: QueryResponse object; use .points for the hit list
        results = self.client.query_points(
            collection_name=col_name,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        hits = results.points

        return [
            {"chunk_id": hit.id, "score": hit.score, **hit.payload}
            for hit in hits
        ]

    def get_few_shot_examples(self, query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve FEW_SHOT examples by semantic similarity to the query."""
        return self.search(
            query_text  = query_text,
            top_k       = top_k,
            chunk_types = [ChunkType.FEW_SHOT],
        )


# _chunk_id_to_int removed — FIX #7: Qdrant supports UUID strings as point
# IDs natively. Using strings directly eliminates the int % 2**63 modulo
# which risks birthday-paradox collisions and is an unnecessary conversion.