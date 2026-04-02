"""
src/rag/chunking/hierarchical.py
─────────────────────────────────
계층적 청킹 (Hierarchical Chunking / Parent-Child) 스캐폴딩.

큰 "부모 청크"와 작은 "자식 청크"를 함께 생성한다.
검색 시에는 자식 청크를 사용하여 정밀도를 높이고,
컨텍스트 전달 시에는 부모 청크를 사용하여 충분한 맥락을 제공한다.

주요 클래스:
- HierarchicalChunk   : 부모-자식 관계를 가진 청크 데이터 클래스
- HierarchicalChunker : 계층적 청크 생성기
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class HierarchicalChunk:
    """부모-자식 관계를 가진 청크."""
    chunk_id: str
    content: str
    parent_id: Optional[str]
    level: int  # 0 = 최상위 부모, 1 = 자식, 2 = 손자 ...
    metadata: dict = field(default_factory=dict)

    def to_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                "chunk_id": self.chunk_id,
                "parent_id": self.parent_id,
                "level": self.level,
                **self.metadata,
            },
        )


class HierarchicalChunker:
    """
    Parent-Child 계층 구조로 문서를 청킹한다.

    Parameters
    ----------
    parent_chunk_size  : int
        부모 청크 크기 (문자 수). 기본값: 1500
    child_chunk_size   : int
        자식 청크 크기 (문자 수). 기본값: 400
    chunk_overlap      : int
        청크 간 오버랩 (문자 수). 기본값: 50

    Notes
    -----
    LangChain의 ParentDocumentRetriever와 함께 사용하면 효과적이다.
    부모 청크는 InMemoryStore 또는 DB에 저장하고,
    자식 청크는 벡터 스토어에 인덱싱한다.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> None:
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(
        self, documents: list[Document]
    ) -> tuple[list[HierarchicalChunk], list[HierarchicalChunk]]:
        """
        문서를 부모/자식 청크 쌍으로 분할한다.

        Parameters
        ----------
        documents : list of Document
            분할할 원본 문서.

        Returns
        -------
        parents : list of HierarchicalChunk
            부모 청크 목록 (InMemoryStore 등에 저장).
        children : list of HierarchicalChunk
            자식 청크 목록 (벡터 스토어에 인덱싱).
        """
        parents: list[HierarchicalChunk] = []
        children: list[HierarchicalChunk] = []

        parent_docs = self._parent_splitter.split_documents(documents)
        for parent_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            parents.append(
                HierarchicalChunk(
                    chunk_id=parent_id,
                    content=parent_doc.page_content,
                    parent_id=None,
                    level=0,
                    metadata=parent_doc.metadata,
                )
            )
            child_docs = self._child_splitter.split_documents([parent_doc])
            for child_doc in child_docs:
                children.append(
                    HierarchicalChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=child_doc.page_content,
                        parent_id=parent_id,
                        level=1,
                        metadata=child_doc.metadata,
                    )
                )

        return parents, children

    def get_child_documents(self, documents: list[Document]) -> list[Document]:
        """자식 청크만 Document 형태로 반환한다 (벡터 스토어 인덱싱용)."""
        _, children = self.split_documents(documents)
        return [c.to_document() for c in children]
