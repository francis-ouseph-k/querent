"""
generation/llm/factory.py
──────────────────────────
Composition root for the LLM querying layer: reads config.settings once and
returns a ready LLMProvider. This is the ONLY module that knows how a value of
LLM_PROVIDER becomes a client object.

TRANSPORT DECISION
──────────────────
All three LangChain-served providers go through ONE client type — LangChain's
ChatOpenAI — pointed at a different base_url:

    local_langchain  →  llama-server's own /v1        (LLM_BASE_URL)
    mistral          →  https://api.mistral.ai/v1     (LLM_MISTRAL_BASE_URL)
    gemini           →  Google's OpenAI-compatibility endpoint
                                                      (LLM_GEMINI_BASE_URL)

Every one of those endpoints is an OpenAI-compatible /v1 surface published by
the vendor, which is what makes the uniform treatment correct rather than a
shortcut. The payoff is concrete: one dependency instead of three, one code
path to reason about, endpoints that stay configurable for proxies or regional
hosts, and a per-vendor difference reduced to three .env values. Swapping any
single provider onto its native integration later (ChatMistralAI,
ChatGoogleGenerativeAI) is a change to _build_langchain's client construction
and nothing else — the LLMProvider port above it does not move.

The vendor import is deferred into the builder. A machine that runs only the
local Qwen model never needs langchain-openai installed, and a machine running
Gemini never needs llama-cpp-python. Selecting a provider whose package is
missing raises LLMConfigurationError naming the exact pip command, instead of
an ImportError traceback from deep inside the pipeline.

SECRETS
───────
API keys are read from settings (which reads .env) and handed straight to the
client. They are never logged, never placed in ProviderInfo, and never
returned to a caller. The only startup output is ProviderInfo.banner(), built
from the provider label and model id alone.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config import llm_providers as lp
from config.settings import settings
from generation.llm.base import (
    LLMConfigurationError,
    LLMProvider,
    ProviderInfo,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

# llama-server ignores the bearer token but the OpenAI client requires a
# non-empty one. This is a placeholder, not a credential.
_LOCAL_PLACEHOLDER_KEY = "not-needed-for-local"


def _info_for(provider: str, model_id: str) -> ProviderInfo:
    spec = lp.SPEC[provider]
    return ProviderInfo(
        display_name   = spec.display_name,
        provider_label = spec.label,
        model_id       = model_id,
        is_local       = provider in lp.LOCAL_PROVIDERS,
    )


def _model_id_for_banner(provider: str) -> str:
    """
    Human-readable model identity — no secrets, no client construction.

    For the native local path the GGUF filename is the honest answer (that file
    is what actually produces tokens). For every LangChain path the model id
    plus endpoint is what identifies the run.
    """
    if provider == lp.LOCAL:
        name = Path(settings.llm.model_path).name
        return f"{name} @ {settings.llm.base_url}" if settings.llm.base_url else name
    base = settings.llm.active_base_url or "base_url unset"
    return f"{settings.llm.active_model} @ {base}"


# ── builders ──────────────────────────────────────────────────────────────
def _build_local() -> LLMProvider:
    """Native llama.cpp path — unchanged behaviour, the default."""
    from generation.llm.local_llama import LocalLlamaProvider
    return LocalLlamaProvider(display_name=lp.SPEC[lp.LOCAL].display_name)


def _build_langchain(provider: str) -> LLMProvider:
    """One builder for every OpenAI-compatible endpoint."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMConfigurationError(
            f"LLM_PROVIDER={provider} is served through LangChain and requires "
            f"langchain-openai. Install it with:  pip install langchain-openai"
        ) from exc

    base_url = settings.llm.active_base_url
    if not base_url:
        if provider == lp.LOCAL_LC:
            raise LLMConfigurationError(
                "LLM_PROVIDER=local_langchain requires LLM_BASE_URL to point at "
                "a running llama-server (e.g. http://localhost:8080/v1). Use "
                "LLM_PROVIDER=local for the in-process fallback path."
            )
        raise LLMConfigurationError(
            f"No base URL configured for LLM_PROVIDER={provider}. "
            f"Set LLM_{provider.upper()}_BASE_URL in .env (see .env.example)."
        )

    api_key = settings.llm.active_api_key
    if provider in lp.LOCAL_PROVIDERS:
        api_key = api_key or _LOCAL_PLACEHOLDER_KEY
    elif not api_key:
        raise LLMConfigurationError(
            f"LLM_{provider.upper()}_API_KEY is not set. Add it to .env "
            f"(see .env.example) before selecting LLM_PROVIDER={provider}."
        )

    model = settings.llm.active_model or lp.SPEC[provider].default_model

    kwargs = dict(
        base_url    = base_url,
        api_key     = api_key,
        model       = model,
        temperature = settings.llm.temperature,
        max_tokens  = settings.llm.max_tokens,
        timeout     = settings.llm.active_timeout,
    )
    # Repetition penalties are a llama.cpp tuning knob (see LLM_FREQUENCY_PENALTY
    # in .env). llama-server honours them; the hosted OpenAI-compatible surfaces
    # do not uniformly accept them, and Gemini's rejects unknown sampling fields
    # outright — so they are sent on the local path only.
    if provider in lp.LOCAL_PROVIDERS:
        kwargs["frequency_penalty"] = settings.llm.frequency_penalty
        kwargs["presence_penalty"]  = settings.llm.presence_penalty

    chat = ChatOpenAI(**kwargs)

    from generation.llm.langchain_provider import LangChainChatProvider
    return LangChainChatProvider(chat, _info_for(provider, f"{model} @ {base_url}"))


_BUILDERS = {
    lp.LOCAL:    _build_local,
    lp.LOCAL_LC: lambda: _build_langchain(lp.LOCAL_LC),
    lp.MISTRAL:  lambda: _build_langchain(lp.MISTRAL),
    lp.GEMINI:   lambda: _build_langchain(lp.GEMINI),
}


def build_provider(provider: str | None = None) -> LLMProvider:
    """Construct the provider named by LLM_PROVIDER (already normalised by settings)."""
    name = lp.normalise(provider or settings.llm.provider)
    builder = _BUILDERS.get(name)
    if builder is None:
        raise LLMConfigurationError(
            f"Unknown LLM_PROVIDER={name!r}. "
            f"Supported: {', '.join(lp.SUPPORTED_PROVIDERS)}."
        )
    logger.info(
        component="llm_factory",
        event="provider_selected",
        provider=name,
        model=_model_id_for_banner(name),
    )
    return builder()


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    """
    Process-wide singleton. The local provider holds a multi-GB GGUF handle on
    the in-process path, so it must not be rebuilt per query; the hosted clients
    are cheap but benefit from connection reuse.
    """
    return build_provider()


def describe_active_llm() -> ProviderInfo:
    """
    Secret-free provenance for the active LLM, WITHOUT constructing a client.

    Called at startup (main.py / batch_run.py) before the pipeline loads, so it
    must not require an API key or a loaded model — a banner should never be the
    thing that fails a run.
    """
    name = lp.normalise(settings.llm.provider)
    return _info_for(name, _model_id_for_banner(name))
