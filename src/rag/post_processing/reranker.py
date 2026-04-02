"""
src/rag/post_processing/reranker.py
─────────────────────────────────────
검색 결과 재순위화 (Reranking) 스캐폴딩.

1차 검색(Dense/Hybrid)으로 가져온 후보 문서를 쿼리와의
의미적 관련도 기준으로 재순위화하여 상위 k 개를 반환한다.

지원 예정 백엔드:
- CrossEncoder (sentence-transformers) : 로컬 Cross-Encoder 모델
- Cohere Rerank API                     : Cohere API 기반 재순위화
- FlashRank                             : 경량 로컬 리랭커

주요 클래스:
- Reranker             : 리랭커 추상 인터페이스
- CrossEncoderReranker : Cross-Encoder 기반 구현
- CohereReranker       : Cohere API 기반 구현 스캐폴딩
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class Reranker(ABC):
    """리랭커 추상 베이스 클래스."""

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        """
        쿼리와 문서 관련도를 재평가하여 상위 top_k 개를 반환한다.

        Parameters
        ----------
        query     : str
            원본 검색 쿼리.
        documents : list[Document]
            1차 검색 결과 후보 문서.
        top_k     : int
            최종 반환 문서 수.
        """
        ...


class CrossEncoderReranker(Reranker):
    """
    Sentence-Transformers Cross-Encoder 기반 리랭커.

    Parameters
    ----------
    model_name : str
        Cross-Encoder 모델명.
        기본값: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        한국어: "bongsoo/moco-sentenceBERT-klue-v1" 등

    Notes
    -----
    의존성: ``pip install sentence-transformers``
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self._model_name = model_name
        self._model = None  # lazy init

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required: pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]


class CohereReranker(Reranker):
    """
    Cohere Rerank API 기반 리랭커 스캐폴딩.

    Parameters
    ----------
    api_key    : str
        Cohere API 키.
    model      : str
        사용할 Cohere 리랭커 모델명. 기본값: "rerank-multilingual-v3.0"

    Notes
    -----
    의존성: ``pip install cohere``
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-multilingual-v3.0",
    ) -> None:
        self._api_key = api_key
        self._model = model

    def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        try:
            import cohere  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("cohere is required: pip install cohere") from exc

        co = cohere.Client(self._api_key)
        texts = [doc.page_content for doc in documents]
        response = co.rerank(model=self._model, query=query, documents=texts, top_n=top_k)

        return [documents[result.index] for result in response.results]
