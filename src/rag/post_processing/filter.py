"""
src/rag/post_processing/filter.py
───────────────────────────────────
검색 결과 필터링 및 중복 제거 스캐폴딩.

Reranker 이후 또는 검색 직후에 적용하여 품질이 낮거나
중복된 문서를 제거한다.

주요 클래스:
- DocumentFilter        : 필터 추상 인터페이스
- ScoreThresholdFilter  : 유사도 점수 임계값 필터
- MMRFilter             : Maximal Marginal Relevance 다양성 필터
- DuplicateFilter       : 텍스트 유사도 기반 중복 제거
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from langchain_core.documents import Document


class DocumentFilter(ABC):
    """문서 필터 추상 베이스 클래스."""

    @abstractmethod
    def filter(self, documents: list[Document]) -> list[Document]:
        """
        문서 목록을 필터링하여 정제된 목록을 반환한다.

        Parameters
        ----------
        documents : list[Document]
            필터링할 문서 목록.
        """
        ...


class ScoreThresholdFilter(DocumentFilter):
    """
    메타데이터의 score 값이 임계값 미만인 문서를 제거한다.

    Parameters
    ----------
    min_score : float
        최소 허용 유사도 점수 (기본값: 0.5).
    score_key : str
        Document.metadata에서 점수를 읽을 키 (기본값: "score").
    """

    def __init__(self, min_score: float = 0.5, score_key: str = "score") -> None:
        self._min_score = min_score
        self._score_key = score_key

    def filter(self, documents: list[Document]) -> list[Document]:
        return [
            doc for doc in documents
            if doc.metadata.get(self._score_key, 1.0) >= self._min_score
        ]


class DuplicateFilter(DocumentFilter):
    """
    Jaccard 유사도 기반 중복 문서 제거 필터.

    Parameters
    ----------
    similarity_threshold : float
        이 값 이상으로 유사한 문서를 중복으로 간주 (기본값: 0.85).
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold

    def filter(self, documents: list[Document]) -> list[Document]:
        unique: list[Document] = []
        for doc in documents:
            if not any(
                self._jaccard(doc.page_content, existing.page_content) >= self._threshold
                for existing in unique
            ):
                unique.append(doc)
        return unique

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)


class MMRFilter(DocumentFilter):
    """
    Maximal Marginal Relevance (MMR) 기반 다양성 필터.

    높은 관련성을 유지하면서 문서 간 다양성을 극대화한다.
    임베딩 벡터가 Document.metadata["embedding"]에 있어야 한다.

    Parameters
    ----------
    k             : int
        최종 반환 문서 수 (기본값: 4).
    lambda_mult   : float
        관련성(1.0)과 다양성(0.0) 간 가중치 (기본값: 0.5).
    """

    def __init__(self, k: int = 4, lambda_mult: float = 0.5) -> None:
        self._k = k
        self._lambda = lambda_mult

    def filter(self, documents: list[Document]) -> list[Document]:
        embeddings = [doc.metadata.get("embedding") for doc in documents]
        if not all(embeddings):
            return documents[: self._k]

        selected: list[int] = [0]
        remaining = list(range(1, len(documents)))

        while len(selected) < self._k and remaining:
            mmr_scores = {
                i: self._lambda * self._sim(embeddings[i], embeddings[selected[0]])
                   - (1 - self._lambda) * max(
                       self._sim(embeddings[i], embeddings[s]) for s in selected
                   )
                for i in remaining
            }
            best = max(mmr_scores, key=lambda x: mmr_scores[x])
            selected.append(best)
            remaining.remove(best)

        return [documents[i] for i in selected]

    @staticmethod
    def _sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0
