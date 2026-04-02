from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from app.core.config import Settings


def build_retriever(vectorstore: VectorStore, config: Settings) -> BaseRetriever:
    """VectorStore로부터 retriever를 생성한다."""
    return vectorstore.as_retriever(search_kwargs={"k": config.retrieval_k})


def retrieve(retriever: BaseRetriever, query: str) -> List[Document]:
    """쿼리에 대한 관련 문서를 검색한다."""
    return retriever.invoke(query)
