from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.session_store import SessionStore
from app.rag.pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.rag_pipeline = RAGPipeline.from_config(settings)
    app.state.session_store = SessionStore(
        max_sessions=settings.max_sessions,
        ttl_seconds=settings.session_ttl_seconds,
    )
    yield
    app.state.session_store.clear()


def _parse_cors_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="RAG / Agent API",
    description=(
        "범용 RAG 파이프라인과 Tool-calling Agent를 REST API로 제공합니다.\n\n"
        "- **Mock 모드**: `MOCK_MODE=true` 설정 시 API 키 없이 실행 가능\n"
        "- **SSE**: `/rag/query/stream`, `/agent/chat/stream` — 완전 응답 후 "
        "문단/단어 단위로 나누어 전송합니다 (LLM 토큰 스트리밍과 다름)\n"
        "- **세션**: `session_id` 멀티턴, `MAX_SESSIONS` / `SESSION_TTL_SECONDS` 로 정리\n"
        "- **선택 보안**: `API_KEY` 설정 시 모든 라우트에 `X-API-Key` 필요"
    ),
    version="1.1.0",
    lifespan=lifespan,
)

origins = _parse_cors_origins(get_settings().cors_origins)
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router)


@app.get("/health", tags=["health"])
def health(request: Request):
    settings = get_settings()
    store: SessionStore = request.app.state.session_store
    return {
        "status": "ok",
        "mock_mode": settings.mock_mode,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "vector_store": settings.vector_store,
        "active_sessions": len(store),
    }
