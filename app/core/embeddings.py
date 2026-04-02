import hashlib
import math
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings


def build_embeddings(config: Settings) -> Embeddings:
    if config.mock_mode:
        return _MockEmbeddings()
    return OpenAIEmbeddings(model=config.embed_model, api_key=config.openai_api_key)


class _MockEmbeddings(Embeddings):
    """API 호출 없이 결정적 hash 기반 벡터를 반환하는 Mock Embeddings."""

    DIMS = 1536

    def _to_vector(self, text: str) -> List[float]:
        digest = hashlib.md5(text.encode()).digest()
        # 16바이트 digest를 반복해서 1536차원 생성
        raw = []
        seed = digest
        while len(raw) < self.DIMS:
            seed = hashlib.md5(seed).digest()
            raw.extend(b / 255.0 for b in seed)
        raw = raw[: self.DIMS]
        # L2 정규화
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._to_vector(text)
