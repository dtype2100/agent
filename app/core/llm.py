from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from app.core.config import Settings


def build_llm(config: Settings) -> BaseChatModel:
    if config.mock_mode:
        return _MockChatModel()
    if config.llm_provider == "openai":
        return ChatOpenAI(model=config.llm_model, api_key=config.openai_api_key)
    if config.llm_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic이 설치되지 않았습니다.\n"
                "pip install langchain-anthropic 을 실행한 후 다시 시도하세요."
            )
        return ChatAnthropic(model=config.llm_model, api_key=config.anthropic_api_key)
    raise ValueError(f"알 수 없는 LLM_PROVIDER: {config.llm_provider!r}")


class _MockChatModel(BaseChatModel):
    """API 호출 없이 고정 응답을 반환하는 Mock LLM."""

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_human = next(
            (m.content for m in reversed(messages) if m.type == "human"), ""
        )
        reply = f"[mock] 질문을 받았습니다: {last_human[:80]}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "BaseChatModel":
        # Mock 모드에서는 도구를 무시하고 self를 반환 (tool_calls 없는 응답을 반환함)
        return self
