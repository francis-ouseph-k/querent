"""
config/llm_providers.py
────────────────────────
Provider names, aliases, and per-provider defaults — the single source of truth
for "which LLMs can LLM_PROVIDER select".

This module deliberately has NO imports. config/settings.py needs these
constants to validate LLM_PROVIDER at startup, and generation/llm/factory.py
needs them to build the provider; putting them in a leaf module keeps that
dependency acyclic (settings must never import from generation/).

CONFIGURATION SHAPE
───────────────────
Each hosted provider is configured by the same triple, following the .env
convention used across the project:

    LLM_<PROVIDER>_BASE_URL     OpenAI-compatible /v1 endpoint
    LLM_<PROVIDER>_API_KEY      secret
    LLM_<PROVIDER>_MODEL        model identifier sent in the request body

The local provider is the exception, and only for backward compatibility: it
keeps its established LLM_BASE_URL / LLM_MODEL_PATH names so no existing .env
breaks, and adds LLM_PRIMARY_MODEL for the model id llama-server is serving.

ADDING A PROVIDER
─────────────────
  1. Add the canonical name to SUPPORTED_PROVIDERS and a row to SPEC.
  2. Add the three fields in config/settings.py::LLMSettings and a branch in
     its active_* accessors.
Nothing in generation/llm/ changes — factory.py builds every hosted provider
from the active_* accessors alone.
"""

# Canonical provider names accepted by LLM_PROVIDER.
LOCAL       = "local"              # llama.cpp direct (HTTP server or in-process) — DEFAULT
LOCAL_LC    = "local_langchain"    # the same llama-server, reached through LangChain
MISTRAL     = "mistral"            # Mistral La Plateforme (OpenAI-compatible /v1)
GEMINI      = "gemini"             # Google Gemini (OpenAI-compatibility endpoint)

SUPPORTED_PROVIDERS: tuple[str, ...] = (LOCAL, LOCAL_LC, MISTRAL, GEMINI)

# Providers that serve the locally hosted GGUF. These are the only ones the
# fine-tuned-model ↔ prompt-profile contract (config/model_profile.py) applies
# to: a hosted frontier model is never a fine-tune of our corpus, so the "ft"
# training-parity prompt is meaningless for it.
LOCAL_PROVIDERS: tuple[str, ...] = (LOCAL, LOCAL_LC)

# Providers reached over LangChain rather than the native llama.cpp path.
LANGCHAIN_PROVIDERS: tuple[str, ...] = (LOCAL_LC, MISTRAL, GEMINI)

# Tolerated spellings → canonical name. Keeps a typo from silently selecting a
# different model than intended (the settings validator raises on anything
# that does not resolve to a supported name).
ALIASES: dict[str, str] = {
    "qwen":            LOCAL,
    "qwen-coder":      LOCAL,
    "qwen_coder":      LOCAL,
    "llama":           LOCAL,
    "llama_cpp":       LOCAL,
    "llamacpp":        LOCAL,
    "primary":         LOCAL,
    "local-langchain": LOCAL_LC,
    "mistralai":       MISTRAL,
    "google":          GEMINI,
    "gemini-flash":    GEMINI,
    "google_genai":    GEMINI,
}


class ProviderSpec:
    """Static, secret-free description of one provider."""

    __slots__ = ("display_name", "label", "default_model", "default_base_url")

    def __init__(self, display_name: str, label: str,
                 default_model: str, default_base_url: str) -> None:
        self.display_name     = display_name
        self.label            = label
        self.default_model    = default_model
        self.default_base_url = default_base_url


# Model ids and endpoints below are the identifiers each vendor publishes for
# its OpenAI-compatible surface. Do not invent variants.
SPEC: dict[str, ProviderSpec] = {
    LOCAL: ProviderSpec(
        display_name     = "Qwen Coder 3B",
        label            = "Local",
        default_model    = "qwen2.5-coder-3b-instruct",
        default_base_url = "",            # LLM_BASE_URL; blank = in-process GGUF
    ),
    LOCAL_LC: ProviderSpec(
        display_name     = "Qwen Coder 3B",
        label            = "Local (LangChain)",
        default_model    = "qwen2.5-coder-3b-instruct",
        default_base_url = "",            # reuses LLM_BASE_URL
    ),
    MISTRAL: ProviderSpec(
        display_name     = "Mistral",
        label            = "Mistral (LangChain)",
        default_model    = "mistral-small-latest",
        default_base_url = "https://api.mistral.ai/v1",
    ),
    GEMINI: ProviderSpec(
        display_name     = "Gemini Flash 2",
        label            = "Google Gemini (LangChain)",
        default_model    = "gemini-2.0-flash",
        default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai",
    ),
}


def normalise(provider: str) -> str:
    """Lower-case, strip, and resolve aliases. Does NOT validate membership."""
    key = (provider or "").strip().lower()
    return ALIASES.get(key, key)
