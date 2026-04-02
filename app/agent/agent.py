from typing import Generator

from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from app.agent.memory import ConversationMemory
from app.agent.tools import ToolRegistry
from app.core.config import Settings
from app.core.llm import build_llm


class Agent:
    """도구 호출(tool calling)을 지원하는 범용 대화형 에이전트.

    사용 예시:
        agent = Agent(config)
        print(agent.chat("15의 제곱근은?"))
        print(agent.chat("방금 결과에 2를 더하면?"))  # 메모리 유지
        agent.reset()
    """

    def __init__(self, config: Settings, registry: ToolRegistry | None = None):
        self.config = config
        self._base_llm = build_llm(config)
        self.registry = registry or ToolRegistry()
        self._llm_with_tools = self.registry.bind_to_llm(self._base_llm)
        self.memory = ConversationMemory(system_prompt=config.system_prompt)

    def chat(self, user_message: str) -> str:
        """사용자 메시지를 받아 에이전트 응답을 반환한다.

        도구 호출이 있으면 내부적으로 처리하고 최종 텍스트 응답을 반환한다.
        대화 이력(memory)은 자동으로 누적된다.
        """
        self.memory.add_user(user_message)
        steps = 0

        while steps < self.config.max_agent_steps:
            response: AIMessage = self._llm_with_tools.invoke(self.memory.messages)
            self.memory.add_ai(response)

            if not response.tool_calls:
                return response.content

            # 모든 도구 호출 실행
            for tc in response.tool_calls:
                result = self.registry.dispatch(tc["name"], tc["args"])
                self.memory.add_tool_result(tc["id"], result)

            steps += 1

        return "최대 스텝 수에 도달했습니다. 질문을 다시 표현해 주세요."

    def set_retriever(self, retriever: BaseRetriever) -> None:
        """document_retrieval 도구에 실제 retriever를 주입한다.

        set_retriever 호출 후에는 LLM 바인딩도 자동으로 갱신된다.
        """
        self.registry.set_retriever(retriever)
        self._llm_with_tools = self.registry.bind_to_llm(self._base_llm)

    def chat_stream(self, user_message: str) -> Generator[dict, None, None]:
        """Tool 실행 이벤트와 최종 답변을 이벤트 딕셔너리로 yield한다.

        Event types:
            {"type": "tool_call",   "name": str, "args": dict}
            {"type": "tool_result", "name": str, "result": str}
            {"type": "token",       "content": str}
            {"type": "done",        "answer": str}
            {"type": "error",       "message": str}
        """
        self.memory.add_user(user_message)
        steps = 0
        try:
            while steps < self.config.max_agent_steps:
                response: AIMessage = self._llm_with_tools.invoke(self.memory.messages)
                self.memory.add_ai(response)

                if not response.tool_calls:
                    content = response.content
                    # 단어 단위로 토큰 스트리밍 시뮬레이션
                    words = content.split(" ")
                    for i, word in enumerate(words):
                        yield {"type": "token", "content": word if i == 0 else " " + word}
                    yield {"type": "done", "answer": content}
                    return

                for tc in response.tool_calls:
                    yield {"type": "tool_call", "name": tc["name"], "args": tc["args"]}
                    result = self.registry.dispatch(tc["name"], tc["args"])
                    self.memory.add_tool_result(tc["id"], result)
                    yield {"type": "tool_result", "name": tc["name"], "result": result}

                steps += 1

            answer = "최대 스텝 수에 도달했습니다. 질문을 다시 표현해 주세요."
            yield {"type": "done", "answer": answer}
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    def reset(self) -> None:
        """대화 이력을 초기화한다. 시스템 프롬프트는 유지된다."""
        self.memory.clear(keep_system=True)
