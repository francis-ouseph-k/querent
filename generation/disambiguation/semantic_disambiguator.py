"""
generation/disambiguation/semantic_disambiguator.py
───────────────────────────────────────────────────

Utility for semantic disambiguation using a lightweight sentence‑transformer.

Provides :class:`SemanticDisambiguator` that loads ``all‑MiniLM‑L6‑v2`` (via
``sentence‑transformers``) and scores candidate disambiguation options against the
full user query. The highest‑scoring option above a configurable similarity
threshold and with a sufficient confidence margin is returned.

The implementation purposefully avoids heavy dependencies – the model is only
loaded once at import time. Option embeddings are cached to reduce latency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ``sentence_transformers`` is optional; import lazily.
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
except Exception:  # pragma: no cover – defensive fallback
    SentenceTransformer = None  # type: ignore[assignment]
    util = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "sentence_transformers not installed – semantic disambiguation disabled"
    )

# H-4 fix: module-level embedding cache replaces @lru_cache on a bound method.
# Keyed by text string only (not by self), preventing GC leaks.
_EMBEDDING_CACHE: Dict[str, object] = {}
_EMBEDDING_CACHE_MAXSIZE = 1024

class SemanticDisambiguator:
    """Semantic resolver based on MiniLM‑L6‑v2 embeddings.

    Parameters
    ----------
    model_name: str, optional
        Model name on the HuggingFace Hub. Defaults to
        ``"sentence-transformers/all-MiniLM-L6-v2"`` which balances performance
        and size (~30 MB).
    threshold: float, optional
        Cosine similarity threshold required to accept a candidate. 0.75 is recommended
        for schema disambiguation to avoid false positives.
    margin: float, optional
        Minimum confidence gap between the best and second-best option scores.
        Ensures highly ambiguous cases fall back to rule-based logic. Default 0.10.
    device: str, optional
        Device to load the model on (e.g. "cpu", "cuda"). Default "cpu".
    cache_dir: str | Path, optional
        Directory to cache the model. By default ``~/.cache/torch/sentence_transformers``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.75,
        margin: float = 0.10,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.threshold = threshold
        self.margin = margin
        self.device = device
        
        if SentenceTransformer is None:
            # Model cannot be loaded – disable functionality.
            self.model = None
            return

        self.logger.debug("Initializing SemanticDisambiguator with model %s on %s", model_name, device)
        try:
            self.model = SentenceTransformer(
                model_name, 
                device=device,
                cache_folder=str(cache_dir) if cache_dir else None
            )
            self.logger.info("Successfully loaded MiniLM model %s for semantic disambiguation", model_name)
        except Exception as exc:  # pragma: no cover – model download failures
            self.logger.error("Failed to load MiniLM model: %s", exc)
            self.model = None

    def _encode_text(self, text: str) -> object:
        """Encode a single string, with module-level caching.

        H-4 fix: moved from @functools.lru_cache (which pinned `self` in
        the cache key, preventing GC and leaking tensors) to a module-level
        dict cache keyed only by text. Cache is bounded to 1024 entries
        with FIFO eviction.
        """
        if text in _EMBEDDING_CACHE:
            return _EMBEDDING_CACHE[text]
        embedding = self.model.encode(text, convert_to_tensor=True)
        # FIFO eviction when cache is full
        if len(_EMBEDDING_CACHE) >= _EMBEDDING_CACHE_MAXSIZE:
            oldest_key = next(iter(_EMBEDDING_CACHE))
            del _EMBEDDING_CACHE[oldest_key]
        _EMBEDDING_CACHE[text] = embedding
        return embedding

    def _get_option_embeddings(self, option_texts: List[str]) -> object:
        """Fetch option embeddings from cache, or compute and cache them."""
        embs = [self._encode_text(text) for text in option_texts]
        return torch.stack(embs)  # type: ignore

    def resolve(
        self, 
        query: str, 
        options: List[str], 
        enrich_option: Optional[Callable[[str], str]] = None
    ) -> Optional[str]:
        """Return the most similar option to *query* or ``None``.

        Parameters
        ----------
        query : str
            The user's original query.
        options : List[str]
            The raw string options to choose from (e.g., ["course", "subject"]).
        enrich_option : Callable[[str], str], optional
            A function that takes a raw option and returns an enriched string
            (e.g., adding DDL comments like "course: academic course offered...").
            Richer text greatly improves MiniLM's disambiguation performance.

        Returns
        -------
        Optional[str]
            The best matching original option, if it meets the threshold and margin.
            Otherwise None.
        """
        if not self.model or not options:
            return None

        # Determine the text to embed for each option
        # Apply lower().strip() normalisation for better semantic matching
        option_texts = [
            (enrich_option(opt) if enrich_option else opt).lower().strip() 
            for opt in options
        ]

        # Embed the query
        query_emb = self._encode_text(query.lower().strip())
        
        # Get option embeddings (utilizes caching)
        option_embs = self._get_option_embeddings(option_texts)

        # util.cos_sim returns a matrix; we take the first row.
        sims = util.cos_sim(query_emb, option_embs)[0]
        
        # Convert to float list for easier sorting/logging
        scores = [float(s) for s in sims]
        
        best_idx = int(sims.argmax())
        best_score = scores[best_idx]
        
        # Calculate margin against the second best option, if there is one
        sorted_scores = sorted(scores, reverse=True)
        second_best_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = best_score - second_best_score

        self.logger.debug(
            "Semantic similarity scores for %r: %s",
            query,
            {opt: float(s) for opt, s in zip(options, scores)},
        )

        if best_score >= self.threshold and margin >= self.margin:
            self.logger.info(
                "Semantic disambiguation selected %r (score %.3f, margin %.3f)",
                options[best_idx],
                best_score,
                margin
            )
            return options[best_idx]
            
        self.logger.info(
            "Semantic disambiguation rejected best option %r. "
            "(score %.3f vs threshold %.2f, margin %.3f vs required %.2f)",
            options[best_idx], best_score, self.threshold, margin, self.margin
        )
        return None

# Module‑level singleton for reuse in the pipeline.
_semantic_disambiguator: Optional[SemanticDisambiguator] = None

def get_semantic_disambiguator() -> SemanticDisambiguator:
    """Retrieve (or create) a shared :class:`SemanticDisambiguator` instance.
    The function ensures that the heavy model is only loaded once per process.
    """
    global _semantic_disambiguator
    if _semantic_disambiguator is None:
        _semantic_disambiguator = SemanticDisambiguator()
    return _semantic_disambiguator

"""End of semantic_disambiguator.py"""
