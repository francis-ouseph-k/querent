"""
retrieval/reranker.py
──────────────────────
Cross-encoder reranker — Phase 1+ optional component.

RRF merges rank positions but doesn't ask "does this chunk actually
answer this query?". The cross-encoder reads each (query, chunk) pair
together and produces a true relevance score.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22 MB — CPU-only, no GPU needed
  - ~80ms for 20 pairs on a modern CPU

When to enable:
  Set RERANKER_ENABLED=true in .env ONLY after measuring retrieval
  recall@K from the observability logs. If correct chunks are retrieved
  but SQL is still wrong, the bottleneck is generation — this won't help.
  Enable only when retrieval quality is the confirmed gap.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from config.settings import settings
from models.schema import SemanticChunk
from utils.logging_config import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Wraps the cross-encoder model for relevance reranking.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, chunks, top_k=10)
    """

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            model_name = settings.reranker.model_name
            logger.info(component="reranker", event="loading_model", model=model_name)
            self._model = CrossEncoder(model_name)
        return self._model

    def rerank(
        self,
        query:  str,
        chunks: list[SemanticChunk],
        top_k:  int | None = None,
    ) -> list[SemanticChunk]:
        """
        Rerank chunks by cross-encoder relevance score.
        Returns top_k chunks sorted by score descending.
        """
        top_k = top_k or settings.reranker.top_k_output

        if not chunks:
            return []

        # Build (query, chunk_text) pairs
        pairs  = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        # Sort by score descending
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        result = [chunk for chunk, _ in ranked[:top_k]]

        logger.debug(
            component="reranker",
            event="reranked",
            input_count=len(chunks),
            output_count=len(result),
            top_score=round(float(ranked[0][1]), 3) if ranked else 0,
        )

        return result
