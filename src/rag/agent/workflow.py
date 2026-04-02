"""
src/rag/agent/workflow.py
──────────────────────────
LangGraph StateGraph 조립 및 컴파일.

nodes.py의 노드 함수와 tools.py의 도구를 연결하여
완전한 ReAct 에이전트 그래프를 조립한다.

그래프 구조:
    START → agent → (should_continue) → tools → agent → ...
                                      ↘ END

주요 함수:
- build_graph(llm, tools, settings)  : 컴파일된 그래프 반환
- build_rag_graph(llm, retriever, settings) : RAG 특화 그래프 반환
"""
from __future__ import annotations

from functools import partial

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.core.config import Settings
from src.rag.agent.nodes import call_model, should_continue
from src.rag.agent.state import AgentState
from src.rag.agent.tools import ToolRegistry


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    settings: Settings | None = None,
) -> "CompiledGraph":  # type: ignore[name-defined]  # noqa: F821
    """
    ReAct 패턴의 LangGraph 에이전트 그래프를 조립하고 컴파일한다.

    Parameters
    ----------
    llm      : BaseChatModel
        도구가 바인딩되지 않은 베이스 LLM.
    tools    : list[BaseTool]
        에이전트가 사용할 도구 목록.
    settings : Settings | None
        애플리케이션 설정 (현재 미사용, 확장용).

    Returns
    -------
    CompiledStateGraph
        MemorySaver 체크포인터가 적용된 컴파일된 그래프.
        thread_id 별로 대화 상태가 자동 유지된다.
    """
    # 도구를 LLM에 바인딩
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    tool_node = ToolNode(tools) if tools else None

    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("agent", partial(call_model, llm=llm_with_tools))
    if tool_node:
        graph.add_node("tools", tool_node)

    # 엣지 연결
    graph.add_edge(START, "agent")
    if tool_node:
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)

    # MemorySaver: thread_id 기반으로 대화 상태를 메모리에 보존
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def build_rag_graph(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    settings: Settings | None = None,
) -> "CompiledGraph":  # type: ignore[name-defined]  # noqa: F821
    """
    문서 검색 도구가 포함된 RAG 특화 에이전트 그래프를 생성한다.

    Parameters
    ----------
    llm       : BaseChatModel
        베이스 LLM.
    retriever : VectorStoreRetriever
        문서 검색에 사용할 retriever.
    settings  : Settings | None
        애플리케이션 설정.
    """
    registry = ToolRegistry()
    registry.set_retriever(retriever)
    return build_graph(llm=llm, tools=registry.get_tools(), settings=settings)
