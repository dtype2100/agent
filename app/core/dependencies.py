from typing import Annotated, cast

from fastapi import Depends, Request

from app.agent.agent import Agent
from app.core.config import Settings, get_settings
from app.core.session_store import SessionStore
from app.rag.pipeline import RAGPipeline


# ── Settings ─────────────────────────────────────────────────
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ── RAGPipeline ──────────────────────────────────────────────
def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.rag_pipeline


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


# ── Session store ────────────────────────────────────────────
def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


SessionStoreDep = Annotated[SessionStore, Depends(get_session_store)]


# ── Agent per session ────────────────────────────────────────
def get_or_create_agent(
    session_id: str | None,
    store: SessionStore,
    settings: Settings,
    pipeline: RAGPipeline,
) -> tuple[str, Agent]:
    """세션 ID로 Agent를 조회하거나 신규 생성한다."""

    def factory() -> Agent:
        agent = Agent(settings)
        agent.set_retriever(pipeline.retriever)
        return agent

    sid, agent = store.get_or_create(session_id, factory)
    return sid, cast(Agent, agent)
