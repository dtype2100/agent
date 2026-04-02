from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from app.core.config import Settings


def get_vectorstore(config: Settings, embeddings: Embeddings) -> VectorStore:
    """설정에 따라 벡터스토어 인스턴스를 생성하여 반환한다."""
    if config.vector_store == "memory":
        return InMemoryVectorStore(embedding=embeddings)

    if config.vector_store == "qdrant":
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from langchain_qdrant import QdrantVectorStore
        except ImportError:
            raise ImportError(
                "Qdrant 백엔드를 사용하려면 추가 패키지가 필요합니다.\n"
                "pip install qdrant-client langchain-qdrant 를 실행한 후 다시 시도하세요."
            )

        if config.qdrant_url:
            client = QdrantClient(url=config.qdrant_url)
        else:
            # 로컬 인메모리 Qdrant (테스트용)
            client = QdrantClient(":memory:")

        # 컬렉션이 없으면 자동 생성 (1536차원 기본값)
        existing = [c.name for c in client.get_collections().collections]
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

    raise ValueError(f"알 수 없는 VECTOR_STORE: {config.vector_store!r}")
