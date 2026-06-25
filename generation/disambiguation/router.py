"""
generation/disambiguation/router.py
───────────────────────────────────

Intent router that evaluates a query against registered DisambiguationSpecs.
"""
import re
from typing import Callable, List, Optional, Tuple

from .spec import DisambiguationSpec
from .semantic_disambiguator import get_semantic_disambiguator
from utils.logging_config import get_logger

# L-4 fix: use project-standard structlog logger instead of stdlib logging
logger = get_logger(__name__)

class DisambiguationRouter:
    """Evaluates user queries against known ambiguity rules and semantic models."""

    def __init__(self, specs: List[DisambiguationSpec]):
        self.specs = specs
        self.semantic_disambiguator = get_semantic_disambiguator()

    def detect_ambiguity(
        self, 
        query: str, 
        entities: List[str],
        enrich_option: Optional[Callable[[str], str]] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Evaluate if the query requires user clarification.
        
        Returns
        -------
        Tuple[bool, List[str], List[str]]
            (is_ambiguous, manual_options, auto_resolved_choices)
            If ambiguous, returns (True, options, []).
            If unambiguous, returns (False, [], auto_resolved_choices).
        """
        lower = query.lower()
        auto_resolved_choices = []

        for spec in self.specs:
            # 1. Does the term appear in the query?
            if not spec.is_triggered(lower):
                continue

            # 2. Is there an explicit resolver word? (e.g. "failed percent")
            if spec.is_resolved(lower):
                continue

            # 3. Entity-based belt-and-braces filtering
            # If extracted entities already narrow to exactly one option, it's resolved.
            if entities:
                entity_set = set(entities)
                matching_options = [
                    opt for opt in spec.options
                    if any(re.search(rf"\b{re.escape(tbl)}\b", opt) for tbl in entity_set)
                ]
                if len(matching_options) == 1:
                    continue  # unambiguously resolved by entity presence

            # 4. Semantic Disambiguation via SLM (MiniLM)
            semantic_choice = self.semantic_disambiguator.resolve(
                lower, 
                spec.options,
                enrich_option=enrich_option
            )
            if semantic_choice:
                # L-4 fix: use structlog keyword args instead of extra={}
                # 2026-06-25 fix: structlog BoundLogger treats the first positional
                # arg as 'event'; passing both a positional string and event= kwarg
                # crashes with TypeError. Use event= only.
                logger.info(
                    component="disambiguation_router",
                    event="semantic_disambiguation_resolved",
                    term=spec.term,
                    chosen=semantic_choice,
                )
                auto_resolved_choices.append(semantic_choice)
                continue # Model successfully disambiguated it!

            # Unresolved ambiguity confirmed — return first match only to not overwhelm user
            logger.info(
                component="disambiguation_router",
                event="ambiguity_detected",
                term=spec.term,
                option_count=len(spec.options),
            )
            return True, spec.options, []

        return False, [], auto_resolved_choices