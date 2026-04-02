"""
src/rag/retrieval/contextual.py
─────────────────────────────────
기본 Dense 벡터 검색 (Contextual Retrieval).

VectorStore의 similarity_search를 래핑하여 통일된 인터페이스를 제공한다.
Anthropic의 Contextual Retrieval 기법 적용 시 이 레이어에서
청크에 문서 요약 컨텍스트를 주입한다.

Functions
---------
build_retriever(vectorstore, config) → VectorStoreRetriever
retrieve(retriever, query)           → list[Document]
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from src.core.config import Settings


def build_retriever(vectorstore: VectorStore, config: Settings) -> VectorStoreRetriever:
    """
    벡터 스토어로부터 검색기를 생성한다.

    Parameters
    ----------
    vectorstore : VectorStore
        검색 대상 벡터 스토어.
    config      : Settings
        retrieval_k (반환 문서 수) 값을 사용한다.
    """
    return vectorstore.as_retriever(search_kwargs={"k": config.retrieval_k})


def retrieve(retriever: VectorStoreRetriever, query: str) -> list[Document]:
    """
    쿼리로 관련 문서를 검색한다.

    Parameters
    ----------
    retriever : VectorStoreRetriever
        build_retriever()로 생성한 검색기.
    query     : str
        검색 쿼리 문자열.
    """
    return retriever.invoke(query)


def retrieve_with_scores(
    vectorstore: VectorStore, query: str, k: int = 4
) -> list[tuple[Document, float]]:
    """
    유사도 점수와 함께 문서를 검색한다.

    Parameters
    ----------
    vectorstore : VectorStore
        검색 대상 벡터 스토어.
    query       : str
        검색 쿼리.
    k           : int
        반환할 최대 문서 수.

    Returns
    -------
    list of (Document, score)
        유사도 점수가 높은 순으로 정렬된 (문서, 점수) 튜플 목록.
    """
    return vectorstore.similarity_search_with_score(query, k=k)
