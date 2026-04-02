"""선택적 API 키 검증."""

from fastapi import HTTPException, Request, status

from app.core.config import Settings, get_settings


def require_api_key_if_configured(request: Request, settings: Settings) -> None:
    """`API_KEY`가 설정된 경우에만 `X-API-Key` 헤더를 검증한다."""
    expected = (settings.api_key or "").strip()
    if not expected:
        return
    got = request.headers.get("X-API-Key", "")
    if got != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 누락된 API 키입니다.",
        )


def verify_api_key_dependency(request: Request) -> None:
    """라우터 공통 의존성: `API_KEY` 설정 시에만 헤더를 검사한다."""
    require_api_key_if_configured(request, get_settings())
