"""
src/core/interceptors.py
────────────────────────
전역 요청/응답 인터셉터 (Starlette Middleware).

포함된 미들웨어:
- RequestLoggingMiddleware : 요청 메서드·경로·처리 시간·상태 코드 로깅
- CorrelationIdMiddleware  : X-Correlation-ID 헤더 주입 (분산 추적용)
- AppErrorMiddleware       : AppError → JSON HTTP 응답 변환

사용 예:
    from src.core.interceptors import attach_middleware
    attach_middleware(app)
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from src.core.exceptions import AppError

logger = logging.getLogger(__name__)


# ── Request Logging ────────────────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """모든 인바운드 요청의 메서드, 경로, 처리 시간, 상태 코드를 INFO 레벨로 기록."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d  (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


# ── Correlation ID ─────────────────────────────────────────────────────────────

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    요청마다 고유한 X-Correlation-ID를 생성하거나 클라이언트가 보낸 값을 사용.
    응답 헤더에도 동일한 ID를 포함시켜 분산 추적을 돕는다.
    """

    HEADER = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = request.headers.get(self.HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers[self.HEADER] = correlation_id
        return response


# ── AppError → HTTP ────────────────────────────────────────────────────────────

class AppErrorMiddleware(BaseHTTPMiddleware):
    """
    도메인 예외(AppError 서브클래스)를 JSON HTTP 응답으로 변환.
    FastAPI의 기본 예외 핸들러와 충돌하지 않도록 AppError만 처리한다.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except AppError as exc:
            http_exc = exc.to_http()
            return JSONResponse(
                status_code=http_exc.status_code,
                content={"detail": http_exc.detail},
            )


# ── 편의 함수 ──────────────────────────────────────────────────────────────────

def attach_middleware(app: FastAPI) -> None:
    """
    표준 미들웨어 스택을 FastAPI 앱에 등록한다.
    main.py 의 lifespan 밖에서 호출해야 한다 (앱 생성 직후).

    등록 순서 = LIFO (마지막에 추가된 것이 가장 먼저 실행).
    """
    app.add_middleware(AppErrorMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
