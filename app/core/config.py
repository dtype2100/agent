from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Embeddings ───────────────────────────────────────────
    embed_model: str = "text-embedding-3-small"

    # ── Vector store ─────────────────────────────────────────
    vector_store: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = ""
    qdrant_collection: str = "app_docs"
    # OpenAI text-embedding-3-small 기본 1536; 모델 변경 시 임베딩 차원과 일치시킬 것
    qdrant_vector_size: int = 1536

    # ── RAG ──────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_k: int = 4

    # ── Agent / 세션 ─────────────────────────────────────────
    max_agent_steps: int = 10
    system_prompt: str = "You are a helpful assistant."
    max_sessions: int = 1000
    session_ttl_seconds: int = 3600

    # ── HTTP / 보안 ─────────────────────────────────────────
    # 쉼표로 구분. 비어 있으면 CORS 미들웨어 비활성
    cors_origins: str = ""
    # 비어 있으면 검증 안 함. 설정 시 모든 API에 X-API-Key 필요
    api_key: str = ""
    max_upload_bytes: int = 10 * 1024 * 1024

    # ── Dev ──────────────────────────────────────────────────
    mock_mode: bool = False

    @field_validator("qdrant_vector_size")
    @classmethod
    def _positive_vector_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("qdrant_vector_size는 1 이상이어야 합니다.")
        return v


@lru_cache
def get_settings() -> Settings:
    """앱 전체에서 공유하는 Settings 싱글톤을 반환한다."""
    return Settings()
