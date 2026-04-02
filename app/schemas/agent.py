from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID (없으면 신규 생성)")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    message_count: int = Field(..., description="현재 세션의 누적 메시지 수")


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
