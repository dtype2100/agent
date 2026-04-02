"""
src/db/vector_store.py
──────────────────────
Dense 벡터 스토어 팩토리.

지원 백엔드:
- "memory" : LangChain InMemoryVectorStore (프로세스 재시작 시 휘발)
- "qdrant"  : QdrantVectorStore (외부 Qdrant 서버 또는 로컬 인메모리)

get_vector_store(config, embeddings) → VectorStore
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.core.config import Settings


def get_vector_store(config: Settings, embeddings: Embeddings) -> VectorStore:
    """설정과 임베딩 모델을 받아 벡터 스토어 인스턴스를 반환한다."""
    if config.vector_store == "memory":
        from langchain_core.vectorstores import InMemoryVectorStore
        return InMemoryVectorStore(embedding=embeddings)

    if config.vector_store == "qdrant":
        return _build_qdrant(config, embeddings)

    raise ValueError(f"Unsupported vector_store: {config.vector_store!r}")


def _build_qdrant(config: Settings, embeddings: Embeddings) -> VectorStore:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    from langchain_qdrant import QdrantVectorStore

    if config.qdrant_url:
        client = QdrantClient(url=config.qdrant_url)
    else:
        client = QdrantClient(":memory:")

    existing = {c.name for c in client.get_collections().collections}
    if config.qdrant_collection not in existing:
        client.create_collection(
            collection_name=config.qdrant_collection,
            vectors_config=VectorParams(
                size=config.qdrant_vector_size,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=config.qdrant_collection,
        embedding=embeddings,
    )
