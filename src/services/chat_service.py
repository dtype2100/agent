"""
src/services/chat_service.py
─────────────────────────────
채팅 비즈니스 로직 서비스.

LangGraph 그래프를 통해 에이전트를 실행하고,
세션 레지스트리(SessionStore)와 연동하여 대화 상태를 관리한다.

LangGraph MemorySaver가 thread_id(= session_id) 기준으로
대화 메시지를 자동 보존하므로, 별도 메시지 저장 로직은 불필요하다.

주요 메서드:
- chat(request)        → ChatResponse (단일 응답)
- chat_stream(request) → AsyncGenerator[str, None] (SSE 스트리밍)
- get_session_info(session_id) → SessionInfo
- reset_session(session_id)    → None
- rag_query(request)   → QueryResponse (RAG 파이프라인 단일 응답)
- rag_query_stream(request) → AsyncGenerator[str, None]
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import Settings
from src.core.exceptions import SessionNotFoundError
from src.db.session import SessionStore
from src.rag.agent.prompts import DEFAULT_SYSTEM_PROMPT
from src.schemas.chat import ChatRequest, ChatResponse, SessionInfo
from src.schemas.rag import QueryRequest, QueryResponse


class ChatService:
    """
    채팅 및 RAG 파이프라인 오케스트레이터.

    Parameters
    ----------
    graph        : CompiledStateGraph
        workflow.build_graph() 또는 build_rag_graph()로 생성한 그래프.
    session_store: SessionStore
        세션 메타데이터 레지스트리.
    vector_store : VectorStore
        RAG 쿼리용 벡터 스토어.
    settings     : Settings
        애플리케이션 설정.
    """

    def __init__(self, graph, session_store: SessionStore, vector_store, settings: Settings) -> None:
        self._graph = graph
        self._sessions = session_store
        self._vector_store = vector_store
        self._settings = settings

    # ── Chat (Agent) ──────────────────────────────────────────────────────────

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        에이전트에 메시지를 전송하고 응답을 반환한다.

        Parameters
        ----------
        request : ChatRequest
            사용자 메시지와 선택적 session_id.
        """
        session_id = request.session_id or str(uuid.uuid4())
        self._sessions.touch(session_id)

        config = {"configurable": {"thread_id": session_id}}
        input_messages = self._build_input(request.message, session_id)

        result = await asyncio.to_thread(
            self._graph.invoke,
            {"messages": input_messages},
            config,
        )

        answer = self._extract_answer(result)
        state = self._graph.get_state(config)
        msg_count = len(list(state.values.get("messages", [])))

        return ChatResponse(session_id=session_id, answer=answer, message_count=msg_count)

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        에이전트 응답을 SSE 형식으로 스트리밍한다.

        Yields
        ------
        str
            "data: <json>\\n\\n" 형식의 SSE 이벤트 문자열.
        """
        session_id = request.session_id or str(uuid.uuid4())
        self._sessions.touch(session_id)

        config = {"configurable": {"thread_id": session_id}}
        input_messages = self._build_input(request.message, session_id)

        yield self._sse("session", {"session_id": session_id})

        try:
            async for event in self._graph.astream_events(
                {"messages": input_messages},
                config,
                version="v2",
            ):
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = getattr(chunk, "content", "")
                    if token:
                        yield self._sse("token", {"token": token})
                elif kind == "on_tool_start":
                    yield self._sse("tool_call", {"name": event.get("name", "")})
                elif kind == "on_tool_end":
                    yield self._sse("tool_result", {"output": str(event["data"].get("output", ""))})
        except Exception as exc:
            yield self._sse("error", {"message": str(exc)})
        finally:
            yield self._sse("done", {})

    # ── Session ───────────────────────────────────────────────────────────────

    def get_session_info(self, session_id: str) -> SessionInfo:
        """세션 정보를 반환한다. 세션이 없으면 SessionNotFoundError."""
        if not self._sessions.exists(session_id):
            raise SessionNotFoundError(f"Session {session_id!r} not found")
        config = {"configurable": {"thread_id": session_id}}
        state = self._graph.get_state(config)
        msg_count = len(list(state.values.get("messages", [])))
        return SessionInfo(session_id=session_id, message_count=msg_count)

    def reset_session(self, session_id: str) -> None:
        """세션 메타데이터를 제거한다. MemorySaver 상태는 thread_id로 분리되므로 별도 처리 불필요."""
        self._sessions.remove(session_id)

    # ── RAG Query ─────────────────────────────────────────────────────────────

    async def rag_query(self, request: QueryRequest) -> QueryResponse:
        """
        RAG 파이프라인을 통해 질의에 답한다.

        1. 벡터 스토어에서 관련 문서를 검색한다.
        2. 문서를 컨텍스트로 LLM에 전달하여 답변을 생성한다.
        """
        from src.rag.retrieval.contextual import retrieve_with_scores  # noqa: PLC0415
        from src.rag.agent.prompts import build_rag_prompt  # noqa: PLC0415
        from langchain_core.messages import HumanMessage  # noqa: PLC0415
        from src.core.llm import build_llm  # noqa: PLC0415

        docs_with_scores = await asyncio.to_thread(
            retrieve_with_scores, self._vector_store, request.query, self._settings.retrieval_k
        )
        contexts = [doc.page_content for doc, _ in docs_with_scores]

        llm = build_llm(self._settings)
        system_msg = build_rag_prompt(contexts)
        human_msg = HumanMessage(content=request.query)

        response = await asyncio.to_thread(llm.invoke, [system_msg, human_msg])
        answer = getattr(response, "content", str(response))

        return QueryResponse(query=request.query, answer=answer, contexts=contexts)

    async def rag_query_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """RAG 쿼리 결과를 SSE 스트리밍으로 반환한다."""
        try:
            response = await self.rag_query(request)
            for i, ctx in enumerate(response.contexts):
                yield self._sse("context", {"index": i, "content": ctx})
            for token in response.answer.split(" "):
                yield self._sse("token", {"token": token + " "})
        except Exception as exc:
            yield self._sse("error", {"message": str(exc)})
        finally:
            yield self._sse("done", {})

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_input(self, message: str, session_id: str) -> list:
        """
        첫 번째 메시지인 경우 시스템 메시지를 앞에 추가한다.

        LangGraph MemorySaver가 이전 메시지를 보존하므로,
        기존 세션에는 시스템 메시지를 중복 추가하지 않는다.
        """
        config = {"configurable": {"thread_id": session_id}}
        state = self._graph.get_state(config)
        existing = list(state.values.get("messages", []))

        messages = []
        if not existing:
            messages.append(SystemMessage(content=self._settings.system_prompt or DEFAULT_SYSTEM_PROMPT))
        messages.append(HumanMessage(content=message))
        return messages

    @staticmethod
    def _extract_answer(result: dict) -> str:
        messages = result.get("messages", [])
        if not messages:
            return ""
        last = messages[-1]
        return getattr(last, "content", str(last))

    @staticmethod
    def _sse(event: str, data: dict) -> str:
        return f"data: {json.dumps({'event': event, **data}, ensure_ascii=False)}\n\n"
