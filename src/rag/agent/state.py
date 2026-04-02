"""
src/rag/agent/state.py
──────────────────────
LangGraph 에이전트 상태 스키마.

AgentState는 그래프의 모든 노드가 공유하는 단일 상태 객체다.
messages 필드에 operator.add reducer를 적용하여
각 노드의 반환값이 자동으로 메시지 리스트에 추가된다.

이 파일은 순수 데이터 스키마만 정의한다.
비즈니스 로직은 nodes.py, 그래프 조립은 workflow.py를 참조.
"""
from __future__ import annotations

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    LangGraph StateGraph의 상태 스키마.

    Fields
    ------
    messages : Annotated[Sequence[BaseMessage], operator.add]
        대화 메시지 히스토리.
        operator.add reducer를 통해 각 노드가 반환하는
        {"messages": [...]} 의 메시지가 기존 리스트에 누적된다.

    Notes
    -----
    추가 필드가 필요한 경우 (예: 검색된 문서, 사용자 ID) 이 TypedDict를 확장한다.
    확장 예:
        class ExtendedState(AgentState):
            retrieved_docs: list[str]
            user_id: str
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
