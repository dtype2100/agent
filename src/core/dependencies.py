"""
src/core/dependencies.py
────────────────────────
FastAPI 의존성 주입 바인딩.

app.state에 저장된 서비스 인스턴스를 엔드포인트에 주입한다.
서비스 초기화는 main.py의 lifespan에서 수행된다.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.core.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_chat_service(request: Request):
    """app.state.chat_service → ChatService 인스턴스."""
    return request.app.state.chat_service


def get_doc_service(request: Request):
    """app.state.doc_service → DocService 인스턴스."""
    return request.app.state.doc_service


# Annotated 타입 별칭 — 엔드포인트에서 타입 힌트로 사용
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.services.chat_service import ChatService
    from src.services.doc_service import DocService

ChatServiceDep = Annotated["ChatService", Depends(get_chat_service)]
DocServiceDep = Annotated["DocService", Depends(get_doc_service)]
