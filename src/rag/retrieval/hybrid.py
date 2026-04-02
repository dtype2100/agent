"""
src/rag/retrieval/hybrid.py
────────────────────────────
Hybrid Retrieval (Dense + Sparse 결합) 스캐폴딩.

Dense 검색(벡터 유사도)과 Sparse 검색(BM25 키워드)의 결과를
Reciprocal Rank Fusion(RRF) 또는 가중 합산으로 결합한다.

Dense만으로는 키워드 정밀도가 낮고, Sparse만으로는 의미 유사도가
낮아지는 문제를 상호 보완한다.

주요 클래스:
- HybridRetriever : Dense + Sparse 결합 검색기
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.db.sparse_store import BM25SparseStore


class HybridRetriever:
    """
    RRF(Reciprocal Rank Fusion) 기반 Hybrid 검색기.

    Parameters
    ----------
    vector_store  : VectorStore
        Dense 검색용 벡터 스토어.
    sparse_store  : BM25SparseStore
        Sparse 검색용 BM25 스토어.
    k             : int
        최종 반환 문서 수 (기본값: 4).
    rrf_k         : int
        RRF 상수. 클수록 하위 랭크 문서의 영향이 커진다. (기본값: 60)
    dense_weight  : float
        Dense 점수 가중치 (0~1). (기본값: 0.5)
    sparse_weight : float
        Sparse 점수 가중치 (0~1). (기본값: 0.5)

    Notes
    -----
    dense_weight + sparse_weight가 반드시 1.0일 필요는 없으나,
    합이 1.0인 경우 점수를 정규화된 비율로 해석할 수 있다.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_store: BM25SparseStore,
        k: int = 4,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> None:
        self._vector_store = vector_store
        self._sparse_store = sparse_store
        self._k = k
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def retrieve(self, query: str) -> list[Document]:
        """
        Dense + Sparse 검색 결과를 RRF로 결합하여 반환한다.

        Parameters
        ----------
        query : str
            검색 쿼리.
        """
        fetch_k = self._k * 3  # 충분한 후보를 가져온 뒤 RRF 적용

        dense_docs = self._vector_store.similarity_search(query, k=fetch_k)
        sparse_docs = self._sparse_store.retrieve(query, k=fetch_k)

        return self._rrf_merge(dense_docs, sparse_docs)

    def _rrf_merge(
        self,
        dense_results: list[Document],
        sparse_results: list[Document],
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion으로 두 결과 리스트를 결합한다.

        RRF 점수: score(d) = Σ 1 / (rrf_k + rank(d, list_i))
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        def _doc_id(doc: Document) -> str:
            return doc.metadata.get("id") or doc.page_content[:64]

        for rank, doc in enumerate(dense_docs := dense_results):
            did = _doc_id(doc)
            scores[did] = scores.get(did, 0.0) + self._dense_weight / (self._rrf_k + rank + 1)
            doc_map[did] = doc

        for rank, doc in enumerate(sparse_docs := sparse_results):
            did = _doc_id(doc)
            scores[did] = scores.get(did, 0.0) + self._sparse_weight / (self._rrf_k + rank + 1)
            doc_map[did] = doc

        sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
        return [doc_map[did] for did in sorted_ids[: self._k]]
