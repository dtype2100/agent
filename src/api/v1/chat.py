"""
src/api/v1/chat.py
───────────────────
채팅 및 RAG 쿼리 API 엔드포인트 (v1).

이 파일은 순수 라우팅 레이어다.
모든 비즈니스 로직은 ChatService에 위임한다.

엔드포인트 목록:
- POST   /v1/chat               : 에이전트 채팅 (단일 응답)
- POST   /v1/chat/stream        : 에이전트 채팅 (SSE 스트리밍)
- GET    /v1/sessions/{id}      : 세션 정보 조회
- DELETE /v1/sessions/{id}      : 세션 초기화
- POST   /v1/rag/query          : RAG 파이프라인 쿼리
- POST   /v1/rag/query/stream   : RAG 쿼리 (SSE 스트리밍)
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.core.dependencies import ChatServiceDep
from src.core.exceptions import SessionNotFoundError
from src.schemas.chat import ChatRequest, ChatResponse, SessionInfo
from src.schemas.rag import QueryRequest, QueryResponse

router = APIRouter(tags=["chat"])


# ── Agent Chat ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    """에이전트에 메시지를 전송하고 단일 응답을 반환한다."""
    return await service.chat(request)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, service: ChatServiceDep) -> StreamingResponse:
    """에이전트 응답을 Server-Sent Events(SSE)로 스트리밍한다."""
    return StreamingResponse(
        service.chat_stream(request),
        media_type="text/event-stream",
    )


# ── Session Management ─────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, service: ChatServiceDep) -> SessionInfo:
    """세션 ID에 해당하는 세션 정보를 반환한다."""
    try:
        return service.get_session_info(session_id)
    except SessionNotFoundError as exc:
        raise exc.to_http() from exc


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, service: ChatServiceDep) -> None:
    """세션을 초기화(삭제)한다."""
    service.reset_session(session_id)


# ── RAG Query ──────────────────────────────────────────────────────────────────

@router.post("/rag/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest, service: ChatServiceDep) -> QueryResponse:
    """RAG 파이프라인을 통해 문서 기반 질의응답을 수행한다."""
    return await service.rag_query(request)


@router.post("/rag/query/stream")
async def rag_query_stream(request: QueryRequest, service: ChatServiceDep) -> StreamingResponse:
    """RAG 쿼리 결과를 SSE 스트리밍으로 반환한다."""
    return StreamingResponse(
        service.rag_query_stream(request),
        media_type="text/event-stream",
    )
