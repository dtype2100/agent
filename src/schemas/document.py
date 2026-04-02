"""
src/schemas/document.py
────────────────────────
문서 인덱싱 관련 Pydantic 요청·응답 스키마.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IndexTextsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="인덱싱할 텍스트 목록")


class IndexResponse(BaseModel):
    indexed: int = Field(..., description="인덱싱된 청크 수")
    filename: Optional[str] = Field(None, description="업로드 파일명 (파일 업로드 시)")
