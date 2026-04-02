from dataclasses import dataclass, field
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from app.core.config import Settings
from app.core.embeddings import build_embeddings
from app.core.llm import build_llm
from app.rag.loader import split_documents
from app.rag.retriever import build_retriever, retrieve
from app.rag.vectorstore import get_vectorstore


@dataclass
class RAGPipeline:
    config: Settings
    vectorstore: VectorStore
    llm: BaseChatModel
    retriever: BaseRetriever

    @classmethod
    def from_config(cls, config: Settings) -> "RAGPipeline":
        """Config로부터 RAGPipeline 인스턴스를 생성한다."""
        embeddings = build_embeddings(config)
        vs = get_vectorstore(config, embeddings)
        retriever = build_retriever(vs, config)
        llm = build_llm(config)
        return cls(config=config, vectorstore=vs, llm=llm, retriever=retriever)

    def index(self, documents: List[Document], auto_split: bool = True) -> int:
        """문서를 벡터스토어에 색인한다. 추가된 청크 수를 반환한다."""
        if auto_split:
            documents = split_documents(documents, self.config)
        self.vectorstore.add_documents(documents)
        return len(documents)

    def ask(self, query: str) -> dict[str, Any]:
        """쿼리에 대해 관련 문서를 검색하고 LLM으로 답변을 생성한다.

        Returns:
            {
                "query": str,
                "contexts": list[str],        # 검색된 청크 텍스트
                "answer": str,                # LLM 생성 답변
                "source_documents": list[Document]
            }

        ragas 평가 파이프라인(rag_eval/)과 호환되는 형식으로 반환한다.
        """
        context_docs = retrieve(self.retriever, query)
        if not context_docs:
            return {
                "query": query,
                "contexts": [],
                "answer": "관련 문서를 찾지 못했습니다.",
                "source_documents": [],
            }

        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        messages = [
            SystemMessage(content="주어진 컨텍스트만을 사용해 질문에 답하세요. 확실하지 않으면 모른다고 말하세요."),
            HumanMessage(content=f"컨텍스트:\n{context_text}\n\n질문: {query}"),
        ]
        response = self.llm.invoke(messages)

        return {
            "query": query,
            "contexts": [doc.page_content for doc in context_docs],
            "answer": response.content,
            "source_documents": context_docs,
        }
