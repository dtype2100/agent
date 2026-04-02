import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.dependencies import (
    PipelineDep,
    SessionStoreDep,
    SettingsDep,
    get_or_create_agent,
)
from app.schemas.agent import ChatRequest, ChatResponse, SessionInfo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse, summary="에이전트 채팅")
async def chat(
    body: ChatRequest,
    settings: SettingsDep,
    pipeline: PipelineDep,
    session_store: SessionStoreDep,
):
    """에이전트와 대화한다. session_id를 전달하면 이전 대화 이력을 유지한다."""
    sid, agent = get_or_create_agent(body.session_id, session_store, settings, pipeline)
    lock = session_store.get_lock(sid)
    try:
        def _run():
            with lock:
                ans = agent.chat(body.message)
                return ans, len(agent.memory)

        answer, message_count = await asyncio.to_thread(_run)
    except Exception:
        logger.exception("Agent chat failed")
        raise HTTPException(
            status_code=502,
            detail="에이전트(LLM) 호출에 실패했습니다.",
        ) from None

    return ChatResponse(session_id=sid, answer=answer, message_count=message_count)


@router.post("/chat/stream", summary="에이전트 채팅 (SSE)")
async def chat_stream(
    body: ChatRequest,
    settings: SettingsDep,
    pipeline: PipelineDep,
    session_store: SessionStoreDep,
):
    """SSE 형식으로 에이전트 응답을 전송한다 (토큰 스트리밍이 아닌 단계별 이벤트).

    Events:
        {"type": "session",     "session_id": str}
        {"type": "tool_call",   "name": str, "args": dict}
        {"type": "tool_result", "name": str, "result": str}
        {"type": "token",       "content": str}
        {"type": "done",        "answer": str}
        {"type": "error",       "message": str}
    """
    sid, agent = get_or_create_agent(body.session_id, session_store, settings, pipeline)
    lock = session_store.get_lock(sid)

    def generate():
        yield f"data: {json.dumps({'type': 'session', 'session_id': sid}, ensure_ascii=False)}\n\n"
        try:
            with lock:
                for event in agent.chat_stream(body.message):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/sessions/{session_id}", response_model=SessionInfo, summary="세션 정보 조회")
def get_session(session_id: str, session_store: SessionStoreDep):
    agent = session_store.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"세션을 찾을 수 없습니다: {session_id}")
    lock = session_store.get_lock(session_id)
    with lock:
        message_count = len(agent.memory)
    return SessionInfo(session_id=session_id, message_count=message_count)


@router.delete("/sessions/{session_id}", summary="세션 초기화")
def reset_session(session_id: str, session_store: SessionStoreDep):
    """대화 이력을 초기화한다 (세션 객체는 유지, 시스템 프롬프트만 남음)."""
    if not session_store.reset_agent(session_id):
        raise HTTPException(status_code=404, detail=f"세션을 찾을 수 없습니다: {session_id}")
    return {"session_id": session_id, "reset": True}
