"""
src/core/exceptions.py
──────────────────────
애플리케이션 전역 커스텀 예외 계층.

- AppError: 모든 도메인 예외의 베이스 클래스
- NotFoundError: 리소스를 찾을 수 없을 때 (HTTP 404)
- ValidationError: 입력 검증 실패 (HTTP 422)
- AuthenticationError: 인증 실패 (HTTP 401)
- UpstreamError: 외부 서비스(LLM, VectorDB 등) 오류 (HTTP 502)
"""
from __future__ import annotations

from fastapi import HTTPException, status


class AppError(Exception):
    """도메인 예외 베이스. HTTP 예외로 변환될 수 있다."""

    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail: str = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.default_detail
        super().__init__(self.detail)

    def to_http(self) -> HTTPException:
        return HTTPException(status_code=self.http_status, detail=self.detail)


class NotFoundError(AppError):
    """요청한 리소스가 존재하지 않음."""

    http_status = status.HTTP_404_NOT_FOUND
    default_detail = "Resource not found"


class ValidationError(AppError):
    """입력 데이터 검증 실패."""

    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_detail = "Validation error"


class AuthenticationError(AppError):
    """API 키 또는 인증 토큰 불일치."""

    http_status = status.HTTP_401_UNAUTHORIZED
    default_detail = "Authentication failed"


class UpstreamError(AppError):
    """외부 서비스 호출 실패 (LLM API, Vector DB 등)."""

    http_status = status.HTTP_502_BAD_GATEWAY
    default_detail = "Upstream service error"


class SessionNotFoundError(NotFoundError):
    """요청한 세션 ID가 존재하지 않거나 만료됨."""

    default_detail = "Session not found or expired"


class DocumentIndexError(UpstreamError):
    """문서 인덱싱 중 오류."""

    default_detail = "Failed to index document"
