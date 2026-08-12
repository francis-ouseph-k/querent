"""
generation/llm/langchain_provider.py
────────────────────────────────────
LangChain-backed provider. One class serves EVERY hosted model.

DESIGN NOTE — why one wrapper instead of one class per vendor
─────────────────────────────────────────────────────────────
LangChain's BaseChatModel already normalises the vendor differences that would
otherwise become conditionals in this codebase: message roles, the request
shape, retry/timeout handling, and — via `usage_metadata` — the token accounting
that Mistral and Gemini report under different keys. So the per-vendor code is
reduced to "which BaseChatModel do I construct, with which kwargs", and that
lives in factory.py as a small builder function. This class holds the *shared*
adaptation: role-dict → LangChain messages, AIMessage → LLMResponse.

Adding a fourth hosted provider therefore means adding one builder function in
factory.py and one registry entry. No change here, and no change in
SQLGenerator or the retry loop.

CONTENT NORMALISATION
─────────────────────
AIMessage.content is `str` for most providers, but the multi-modal content-block
list (`[{"type": "text", "text": ...}, ...]`) is also valid and Gemini can emit
it. `_content_to_text` collapses both to a plain string so the downstream
JSON-contract parser in SQLGenerator never sees a list.
"""

from __future__ import annotations

from typing import Any

from generation.llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    ProviderInfo,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _content_to_text(content: Any) -> str:
    """Collapse str | list[content-block] into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # LangChain text blocks use {"type": "text", "text": "..."}
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "".join(parts).strip()
    return str(content).strip()


class LangChainChatProvider(LLMProvider):
    """Wraps any LangChain BaseChatModel behind the LLMProvider port."""

    def __init__(self, chat_model: Any, info: ProviderInfo) -> None:
        self._model = chat_model
        self._info = info

    def info(self) -> ProviderInfo:
        return self._info

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        # max_tokens / temperature are bound onto the chat model at construction
        # time (factory.py) because that is where LangChain expects them; they
        # are accepted here only to satisfy the shared LLMProvider signature.
        from langchain_core.messages import HumanMessage, SystemMessage

        lc_messages: list[Any] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        try:
            ai = self._model.invoke(lc_messages)
        except Exception as exc:
            logger.exception("external_inference_error")
            logger.error(
                component="sql_generator",
                event="external_inference_error",
                provider=self._info.provider_label,
                model=self._info.model_id,
                error=str(exc),
            )
            raise LLMProviderError(str(exc)) from exc

        usage = getattr(ai, "usage_metadata", None) or {}
        return LLMResponse(
            text              = _content_to_text(getattr(ai, "content", "")),
            prompt_tokens     = usage.get("input_tokens"),
            completion_tokens = usage.get("output_tokens"),
        )
