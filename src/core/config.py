"""
src/core/config.py
──────────────────
애플리케이션 전역 설정.
Pydantic BaseSettings를 통해 환경변수 자동 바인딩.
get_settings()는 lru_cache로 싱글턴 반환.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = "text-embedding-3-small"

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = ""
    qdrant_collection: str = "app_docs"
    qdrant_vector_size: int = 1536

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_k: int = 4

    # ── Agent ─────────────────────────────────────────────────────────────────
    max_agent_steps: int = 10
    system_prompt: str = "You are a helpful assistant."
    max_memory_turns: int = 20

    # ── Session ───────────────────────────────────────────────────────────────
    max_sessions: int = 1000
    session_ttl_seconds: int = 3600

    # ── HTTP / Security ───────────────────────────────────────────────────────
    cors_origins: str = ""
    api_key: str = ""
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    # ── Dev ───────────────────────────────────────────────────────────────────
    mock_mode: bool = False

    @field_validator("qdrant_vector_size")
    @classmethod
    def _positive_vector_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("qdrant_vector_size must be >= 1")
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
