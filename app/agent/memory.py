from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class ConversationMemory:
    """멀티턴 대화 이력을 관리한다. SystemMessage는 항상 index 0에 고정된다."""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self._messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

    def add_user(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))

    def add_ai(self, message: AIMessage) -> None:
        self._messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

    @property
    def messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def clear(self, keep_system: bool = True) -> None:
        self._messages = self._messages[:1] if keep_system else []

    def __len__(self) -> int:
        return len(self._messages) - 1  # 시스템 메시지 제외
