"""
src/rag/agent/nodes.py
──────────────────────
LangGraph 노드 함수 정의.

각 함수는 AgentState를 받아 상태 업데이트 딕셔너리를 반환한다.
함수 자체는 순수 함수에 가깝게 유지하고, LLM/도구 등의 의존성은
workflow.py에서 partial로 주입한다.

노드 목록:
- call_model(state, llm)         : LLM 호출 노드
- should_continue(state)         : 조건부 엣지 라우터 (도구 호출 여부 판단)
- trim_messages(state, max_turns): 메모리 관리 노드 (선택적)
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.rag.agent.state import AgentState


def call_model(state: AgentState, llm: BaseChatModel) -> dict:
    """
    LLM을 호출하여 응답 메시지를 생성하는 노드.

    Parameters
    ----------
    state : AgentState
        현재 그래프 상태. state["messages"]를 LLM에 전달한다.
    llm   : BaseChatModel
        도구가 바인딩된 LLM 인스턴스 (workflow.py에서 partial로 주입).

    Returns
    -------
    dict
        {"messages": [AIMessage]} — reducer에 의해 state["messages"]에 누적.
    """
    response: AIMessage = llm.invoke(list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    마지막 메시지에 tool_calls가 있는지 확인하여 다음 노드를 결정한다.

    Parameters
    ----------
    state : AgentState
        현재 그래프 상태.

    Returns
    -------
    str
        "tools"  : tool_calls가 있으면 도구 실행 노드로 라우팅.
        "end"    : tool_calls가 없으면 그래프 종료.
    """
    last: BaseMessage = list(state["messages"])[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def trim_messages_node(state: AgentState, max_turns: int = 20) -> dict:
    """
    대화 히스토리가 max_turns를 초과하면 오래된 메시지를 제거하는 노드.

    SystemMessage는 항상 보존한다.

    Parameters
    ----------
    state     : AgentState
        현재 그래프 상태.
    max_turns : int
        보존할 최대 대화 턴 수 (HumanMessage 기준). workflow.py에서 partial 주입.
    """
    messages = list(state["messages"])
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    human_indices = [i for i, m in enumerate(non_system) if isinstance(m, HumanMessage)]
    if len(human_indices) > max_turns:
        cutoff = human_indices[-max_turns]
        non_system = non_system[cutoff:]

    # 상태를 교체하기 위해 기존 메시지를 제거한 뒤 새 목록을 반환
    # 주의: messages reducer는 add이므로 여기서는 외부에서 checkpoint reset을 사용하거나
    # Custom reducer를 정의해야 완전한 교체가 가능하다.
    # 현재 구현은 WorkflowBuilder에서 별도 처리 필요.
    return {"messages": system_msgs + non_system}
