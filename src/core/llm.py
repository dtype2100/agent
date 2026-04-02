"""
src/core/llm.py
───────────────
LLM 팩토리 및 Mock 구현.

build_llm(config) → BaseChatModel
  - mock_mode=True  : _MockChatModel (API 키 불필요)
  - openai          : ChatOpenAI
  - anthropic       : ChatAnthropic (lazy import)
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.core.config import Settings


def build_llm(config: Settings) -> BaseChatModel:
    """설정을 기반으로 LLM 인스턴스를 생성한다."""
    if config.mock_mode:
        return _MockChatModel()

    if config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.llm_model, api_key=config.openai_api_key)

    if config.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # pip install langchain-anthropic
        return ChatAnthropic(model=config.llm_model, api_key=config.anthropic_api_key)

    raise ValueError(f"Unsupported llm_provider: {config.llm_provider!r}")


class _MockChatModel(BaseChatModel):
    """API 호출 없이 고정 응답을 반환하는 더미 모델 (테스트/개발용)."""

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1].content if messages else ""
        reply = f"[Mock] Echo: {str(last)[:80]}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def bind_tools(self, tools: list, **kwargs: Any) -> "_MockChatModel":  # type: ignore[override]
        return self
