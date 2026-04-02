"""
src/rag/retrieval/graph_search.py
──────────────────────────────────
그래프 기반 검색 스캐폴딩 (GraphRAG).

지식 그래프(GraphStore)를 통해 엔티티-관계 기반으로 탐색하여
벡터 검색의 의미 유사도만으로는 찾기 어려운 다중 홉 관계를
포함한 문서를 검색한다.

주요 클래스:
- GraphRetriever : 그래프 탐색 기반 검색기
"""
from __future__ import annotations

from langchain_core.documents import Document

from src.db.graph_store import GraphStore


class GraphRetriever:
    """
    지식 그래프 기반 문서 검색기.

    쿼리에서 엔티티를 추출하고, 그래프 내 이웃 노드를 탐색하여
    관련 문서를 반환한다.

    Parameters
    ----------
    graph_store    : GraphStore
        엔티티-관계 그래프 스토어.
    vector_store   : VectorStore (optional)
        엔티티 설명을 기반으로 추가 문서를 검색할 벡터 스토어.
    max_depth      : int
        그래프 탐색 최대 깊이 (기본값: 2).
    max_entities   : int
        쿼리당 추출할 최대 엔티티 수 (기본값: 5).

    Notes
    -----
    엔티티 추출(NER)은 LLM 기반 또는 SpaCy를 사용할 수 있다.
    의존성: ``pip install spacy`` + ``python -m spacy download ko_core_news_sm``
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store=None,
        max_depth: int = 2,
        max_entities: int = 5,
    ) -> None:
        self._graph = graph_store
        self._vector_store = vector_store
        self._max_depth = max_depth
        self._max_entities = max_entities

    def retrieve(self, query: str) -> list[Document]:
        """
        그래프 탐색으로 관련 문서를 검색한다.

        Parameters
        ----------
        query : str
            검색 쿼리.

        Returns
        -------
        list[Document]
            그래프 이웃 엔티티의 설명을 기반으로 구성된 문서 목록.
        """
        entities = self._extract_entities(query)
        result_docs: list[Document] = []

        for entity_name in entities[: self._max_entities]:
            neighbors = self._graph.search_neighbors(entity_name, depth=self._max_depth)
            for neighbor in neighbors:
                doc = Document(
                    page_content=neighbor.description or neighbor.name,
                    metadata={
                        "entity": neighbor.name,
                        "entity_type": neighbor.entity_type,
                        "source": "graph",
                    },
                )
                result_docs.append(doc)

        if self._vector_store and entities:
            entity_query = " ".join(entities)
            vector_docs = self._vector_store.similarity_search(entity_query, k=4)
            result_docs.extend(vector_docs)

        return self._deduplicate(result_docs)

    def _extract_entities(self, query: str) -> list[str]:
        """
        쿼리에서 엔티티를 추출한다.

        현재는 단순 명사 추출 스텁 구현.
        실제 구현 시 SpaCy NER 또는 LLM 기반 추출로 교체한다.

        Parameters
        ----------
        query : str
            엔티티를 추출할 쿼리 문자열.
        """
        # TODO: SpaCy NER 또는 LLM 기반 엔티티 추출로 교체
        tokens = query.split()
        return [t for t in tokens if t[0].isupper()][:self._max_entities] or tokens[:2]

    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in docs:
            key = doc.page_content[:64]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique
