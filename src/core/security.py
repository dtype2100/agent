"""
src/core/security.py
────────────────────
API 키 인증 의존성.

설정에 API_KEY가 지정된 경우 모든 보호된 엔드포인트는
X-API-Key 헤더를 검사한다. 타이밍 공격 방지를 위해
hmac.compare_digest 를 사용한다.
"""
from __future__ import annotations

import hmac

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from src.core.config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key_if_configured(
    request: Request,
    settings: Settings = Depends(get_settings),
    api_key_header: str | None = Depends(_api_key_header),
) -> None:
    """
    API_KEY 환경변수가 설정된 경우에만 헤더를 검증한다.
    미설정 시 모든 요청을 허용한다.
    """
    if not settings.api_key:
        return
    if not api_key_header or not hmac.compare_digest(
        api_key_header.encode(), settings.api_key.encode()
    ):
        from src.core.exceptions import AuthenticationError
        raise AuthenticationError("Invalid or missing X-API-Key header")


def verify_api_key_dependency(
    request: Request,
    settings: Settings = Depends(get_settings),
    api_key_header: str | None = Depends(_api_key_header),
) -> None:
    """FastAPI 라우터에 직접 주입 가능한 래퍼."""
    require_api_key_if_configured(request, settings, api_key_header)
