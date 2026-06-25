"""spec.py
Dataclass defining a single Disambiguation Specification.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import re

@dataclass
class DisambiguationSpec:
    """A specification for a single ambiguous term in the NL-to-SQL pipeline.
    
    Attributes
    ----------
    term : str
        The primary ambiguous term to match (e.g., 'failed', 'pending').
    options : List[str]
        A list of descriptions representing the valid schema mappings/interpretations.
        These are displayed to the user if the query cannot be disambiguated, or
        used for semantic SLM matching.
    resolvers : List[str], optional
        A list of explicit keywords that, if present alongside `term`, resolve
        the ambiguity automatically (e.g. ['percent', 'cutoff'] for 'failed').
    custom_matcher : Callable[[str], bool], optional
        An optional hook for custom regex/logic matching. If provided, it overrides
        the standard `\\bterm\\b` whole-word check.
    """
    term: str
    options: List[str]
    resolvers: List[str] = field(default_factory=list)
    custom_matcher: Optional[Callable[[str], bool]] = None

    def __post_init__(self):
        """Pre-compile regex patterns for performance."""
        self._term_pattern = re.compile(rf"\b{re.escape(self.term)}\b", re.IGNORECASE)
        self._resolver_patterns = [
            re.compile(rf"\b{re.escape(r)}\b", re.IGNORECASE) 
            for r in self.resolvers
        ]

    def is_triggered(self, query_lower: str) -> bool:
        """Check if the ambiguous term is present in the query."""
        if self.custom_matcher:
            return self.custom_matcher(query_lower)
        # Default: whole word match
        return bool(self._term_pattern.search(query_lower))

    def is_resolved(self, query_lower: str) -> bool:
        """Check if the query already contains a known resolver word."""
        if not self._resolver_patterns:
            return False
        return any(pattern.search(query_lower) for pattern in self._resolver_patterns)
