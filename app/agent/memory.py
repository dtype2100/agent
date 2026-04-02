from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class ConversationMemory:
    """멀티턴 대화 이력을 관리한다. SystemMessage는 항상 index 0에 고정된다."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_turns: int | None = None,
    ):
        self._messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        self._max_turns = max_turns

    def _trim(self) -> None:
        """max_turns 초과 시 가장 오래된 턴(HumanMessage 기준)을 제거한다."""
        if not self._max_turns:
            return
        human_indices = [
            i for i, m in enumerate(self._messages) if isinstance(m, HumanMessage)
        ]
        if len(human_indices) > self._max_turns:
            cutoff = human_indices[-self._max_turns]
            self._messages = self._messages[:1] + self._messages[cutoff:]

    def add_user(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))
        self._trim()

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
