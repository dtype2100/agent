"""
src/schemas/chat.py
────────────────────
채팅/에이전트 관련 Pydantic 요청·응답 스키마.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="기존 세션 ID (없으면 신규 생성)")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="사용 또는 생성된 세션 ID")
    answer: str = Field(..., description="에이전트 응답 텍스트")
    message_count: int = Field(..., description="현재 세션의 총 메시지 수")


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
