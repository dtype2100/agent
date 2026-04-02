"""
src/rag/chunking/semantic.py
─────────────────────────────
의미 기반 청킹 (Semantic Chunking) 스캐폴딩.

고정 크기 분할 대신, 임베딩 유사도를 이용하여 의미적으로
자연스러운 경계에서 텍스트를 분할한다.

알고리즘:
1. 문장 단위로 분리
2. 연속 문장 쌍의 임베딩 코사인 유사도 계산
3. 유사도가 임계값 이하인 지점을 청크 경계로 결정
4. 최소/최대 크기 제약 적용

주요 클래스:
- SemanticChunker : 의미 기반 청커
"""
from __future__ import annotations

import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class SemanticChunker:
    """
    임베딩 유사도 기반 의미 청커.

    Parameters
    ----------
    embeddings        : Embeddings
        문장 임베딩 생성에 사용할 모델.
    similarity_threshold : float
        청크 분리 임계값. 연속 문장 쌍의 유사도가 이 값 이하이면 경계로 간주.
        낮을수록 더 세밀하게 분리. (기본값: 0.8)
    min_chunk_size    : int
        최소 청크 크기 (문자 수). 이보다 작은 청크는 다음 청크에 병합. (기본값: 100)
    max_chunk_size    : int
        최대 청크 크기 (문자 수). 이를 초과하면 강제 분리. (기본값: 1500)

    Notes
    -----
    - 임베딩 API 비용이 발생하므로 대량 처리 시 배치 크기를 고려할 것.
    - LangChain의 SemanticChunker (langchain_experimental)로 대체 가능.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        similarity_threshold: float = 0.8,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
    ) -> None:
        self._embeddings = embeddings
        self._threshold = similarity_threshold
        self._min_size = min_chunk_size
        self._max_size = max_chunk_size

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        문서 목록을 의미 경계 기준으로 청킹한다.

        Parameters
        ----------
        documents : list of Document
            청킹할 원본 문서 리스트.
        """
        result: list[Document] = []
        for doc in documents:
            chunks = self._split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                result.append(
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )
        return result

    def _split_text(self, text: str) -> list[str]:
        """텍스트를 의미 경계에서 분할한다."""
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return [text] if text.strip() else []

        embeddings = self._embeddings.embed_documents(sentences)
        boundaries = {0}
        for i in range(len(sentences) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self._threshold:
                boundaries.add(i + 1)

        chunks: list[str] = []
        sorted_boundaries = sorted(boundaries) + [len(sentences)]
        for start, end in zip(sorted_boundaries, sorted_boundaries[1:]):
            chunk = " ".join(sentences[start:end])
            if len(chunk) >= self._min_size:
                chunks.append(chunk)
            elif chunks:
                chunks[-1] += " " + chunk

        return self._enforce_max_size(chunks)

    def _enforce_max_size(self, chunks: list[str]) -> list[str]:
        """최대 크기를 초과하는 청크를 단순 분할한다."""
        result: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self._max_size:
                result.append(chunk)
            else:
                for i in range(0, len(chunk), self._max_size):
                    result.append(chunk[i : i + self._max_size])
        return result
