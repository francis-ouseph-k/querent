"""
generation.llm
──────────────
Provider abstraction for the NL→SQL querying layer.

    from generation.llm import get_provider, describe_active_llm

`get_provider()` returns the process-wide LLMProvider selected by LLM_PROVIDER.
`describe_active_llm()` returns secret-free provenance for the startup banner
without constructing a client.
"""

from generation.llm.base import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    ProviderInfo,
)
from generation.llm.factory import (
    build_provider,
    describe_active_llm,
    get_provider,
)

__all__ = [
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "ProviderInfo",
    "build_provider",
    "describe_active_llm",
    "get_provider",
]
