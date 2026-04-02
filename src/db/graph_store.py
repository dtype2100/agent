"""
src/db/graph_store.py
─────────────────────
그래프 데이터베이스 연동 스캐폴딩 (GraphRAG 지원용).

GraphStore는 엔티티-관계 기반 지식 그래프를 저장하고,
graph_search.py의 그래프 탐색 검색과 연동된다.

지원 예정 백엔드:
- Neo4j   : langchain_community.graphs.Neo4jGraph
- Memgraph: 호환 드라이버 사용

주요 클래스:
- GraphStore : 그래프 스토어 추상 인터페이스
- Neo4jGraphStore : Neo4j 백엔드 구현 스캐폴딩
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Entity:
    """지식 그래프의 엔티티 노드."""
    name: str
    entity_type: str
    description: str = ""


@dataclass
class Relationship:
    """두 엔티티 간의 관계 엣지."""
    source: str
    target: str
    relation: str
    weight: float = 1.0


class GraphStore(ABC):
    """그래프 스토어 추상 베이스 클래스."""

    @abstractmethod
    def add_entities(self, entities: list[Entity]) -> None:
        """엔티티 목록을 그래프에 추가한다."""
        ...

    @abstractmethod
    def add_relationships(self, relationships: list[Relationship]) -> None:
        """관계 목록을 그래프에 추가한다."""
        ...

    @abstractmethod
    def search_neighbors(self, entity_name: str, depth: int = 2) -> list[Entity]:
        """
        주어진 엔티티의 이웃 노드를 depth 홉 내에서 탐색한다.

        Parameters
        ----------
        entity_name : str
            탐색 시작 엔티티.
        depth : int
            탐색 깊이 (기본값: 2).
        """
        ...

    @abstractmethod
    def query(self, cypher_or_gremlin: str) -> list[dict]:
        """
        네이티브 쿼리 언어(Cypher / Gremlin)로 그래프를 조회한다.

        Parameters
        ----------
        cypher_or_gremlin : str
            실행할 쿼리 문자열.
        """
        ...


class Neo4jGraphStore(GraphStore):
    """
    Neo4j 백엔드 구현 스캐폴딩.

    Parameters
    ----------
    url      : str  Neo4j Bolt URL (e.g. "bolt://localhost:7687")
    username : str  인증 사용자명
    password : str  인증 비밀번호

    Notes
    -----
    의존성: ``pip install langchain-community neo4j``
    """

    def __init__(self, url: str, username: str, password: str) -> None:
        self._url = url
        self._username = username
        self._password = password
        self._graph = None  # langchain_community.graphs.Neo4jGraph (lazy init)

    def _connect(self):
        if self._graph is None:
            from langchain_community.graphs import Neo4jGraph  # noqa: PLC0415
            self._graph = Neo4jGraph(
                url=self._url, username=self._username, password=self._password
            )
        return self._graph

    def add_entities(self, entities: list[Entity]) -> None:
        graph = self._connect()
        for e in entities:
            graph.query(
                "MERGE (n:Entity {name: $name}) SET n.type=$type, n.desc=$desc",
                {"name": e.name, "type": e.entity_type, "desc": e.description},
            )

    def add_relationships(self, relationships: list[Relationship]) -> None:
        graph = self._connect()
        for r in relationships:
            graph.query(
                """
                MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                MERGE (a)-[rel:RELATES {type: $rel}]->(b)
                SET rel.weight = $weight
                """,
                {"src": r.source, "tgt": r.target, "rel": r.relation, "weight": r.weight},
            )

    def search_neighbors(self, entity_name: str, depth: int = 2) -> list[Entity]:
        graph = self._connect()
        rows = graph.query(
            f"MATCH (n:Entity {{name: $name}})-[*1..{depth}]-(m:Entity) RETURN m",
            {"name": entity_name},
        )
        return [
            Entity(
                name=row["m"]["name"],
                entity_type=row["m"].get("type", ""),
                description=row["m"].get("desc", ""),
            )
            for row in rows
        ]

    def query(self, cypher_or_gremlin: str) -> list[dict]:
        return self._connect().query(cypher_or_gremlin)
