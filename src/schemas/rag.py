"""
src/schemas/rag.py
───────────────────
RAG 쿼리 관련 Pydantic 요청·응답 스키마.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="검색 및 질의 텍스트")


class QueryResponse(BaseModel):
    query: str
    answer: str
    contexts: list[str] = Field(default_factory=list, description="검색된 컨텍스트 문서 목록")
