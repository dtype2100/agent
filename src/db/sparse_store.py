"""
src/db/sparse_store.py
──────────────────────
Sparse 벡터 스토어 (BM25 기반 키워드 검색).

Hybrid Retrieval에서 dense 검색을 보완하기 위해 사용된다.
현재는 BM25Retriever(rank_bm25 패키지)를 래핑하는 스캐폴딩이며,
실제 구현 시 ElasticSearch, OpenSearch 등으로 교체 가능하다.

주요 클래스:
- SparseStore : 문서 인덱싱 및 키워드 검색 인터페이스
"""
from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class SparseRetrieverProtocol(Protocol):
    """Sparse 검색기가 구현해야 하는 인터페이스."""

    def add_documents(self, documents: list[Document]) -> None:
        ...

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        ...


class BM25SparseStore:
    """
    BM25 기반 인메모리 Sparse 검색 스토어.

    Parameters
    ----------
    k1 : float
        BM25 term frequency 포화 파라미터 (기본값 1.5).
    b  : float
        BM25 문서 길이 정규화 파라미터 (기본값 0.75).

    Notes
    -----
    - 의존성: ``pip install rank-bm25``
    - 프로덕션에서는 ElasticSearch 또는 OpenSearch로 교체를 권장한다.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._documents: list[Document] = []
        self._bm25 = None  # rank_bm25.BM25Okapi instance (lazy init)

    def add_documents(self, documents: list[Document]) -> None:
        """문서를 인덱스에 추가하고 BM25 인덱스를 재빌드한다."""
        self._documents.extend(documents)
        self._build_index()

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """
        BM25 점수 기준으로 상위 k 개의 문서를 반환한다.

        Parameters
        ----------
        query : str
            검색 쿼리.
        k : int
            반환할 최대 문서 수.
        """
        if self._bm25 is None or not self._documents:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._documents[i] for i in top_indices]

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi  # pip install rank-bm25
        except ImportError as exc:
            raise ImportError("rank-bm25 is required: pip install rank-bm25") from exc

        tokenized = [doc.page_content.lower().split() for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized, k1=self._k1, b=self._b)
