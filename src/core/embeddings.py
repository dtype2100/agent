"""
src/core/embeddings.py
──────────────────────
Embeddings 팩토리 및 Mock 구현.

build_embeddings(config) → Embeddings
  - mock_mode=True : _MockEmbeddings (결정론적 해시 기반 벡터)
  - default        : OpenAIEmbeddings
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from langchain_core.embeddings import Embeddings

from src.core.config import Settings


def build_embeddings(config: Settings) -> Embeddings:
    """설정을 기반으로 Embeddings 인스턴스를 생성한다."""
    if config.mock_mode:
        return _MockEmbeddings()

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.embed_model, api_key=config.openai_api_key)


class _MockEmbeddings(Embeddings):
    """
    MD5 해시로 결정론적 1536차원 단위벡터를 생성하는 더미 임베딩.
    API 키 없이 로컬 테스트에 사용한다.
    """

    _DIM = 1536

    def _to_vector(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).digest()
        # 16 바이트 → 반복 패딩 → 1536 요소
        raw = [b / 255.0 - 0.5 for b in (digest * (self._DIM // 16 + 1))[: self._DIM]]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._to_vector(text)
