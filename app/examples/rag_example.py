"""
RAG 파이프라인 예제.

사용법:
    # Mock 모드 (API 키 불필요)
    MOCK_MODE=true python app/examples/rag_example.py

    # OpenAI 실제 모드
    OPENAI_API_KEY=sk-... python app/examples/rag_example.py

    # Qdrant 백엔드 (Docker 필요)
    VECTOR_STORE=qdrant QDRANT_URL=http://localhost:6333 \\
        OPENAI_API_KEY=sk-... python app/examples/rag_example.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.config import Settings
from app.rag.loader import load_texts
from app.rag.pipeline import RAGPipeline


SAMPLE_DOCS = [
    """RAG(Retrieval-Augmented Generation)는 외부 지식 베이스에서 관련 문서를 검색하여
LLM의 답변 품질을 향상시키는 기법입니다. 검색 결과를 프롬프트에 포함시켜
환각(hallucination)을 줄이고 최신 정보를 반영할 수 있습니다.""",

    """벡터 데이터베이스는 텍스트를 고차원 벡터로 변환하여 저장하고,
코사인 유사도나 L2 거리를 기반으로 의미적으로 유사한 문서를 검색합니다.
대표적인 벡터 DB로는 Qdrant, Pinecone, Weaviate, Chroma 등이 있습니다.""",

    """LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.
Document, VectorStore, Retriever, Chain 등의 추상화를 제공하여
RAG, 에이전트, 멀티모달 파이프라인을 빠르게 구축할 수 있습니다.""",

    """청킹(Chunking)은 긴 문서를 작은 조각으로 분할하는 과정입니다.
청크 크기(chunk_size)와 오버랩(chunk_overlap)을 적절히 설정해야
검색 품질과 컨텍스트 완결성 사이의 균형을 맞출 수 있습니다.""",

    """임베딩 모델은 텍스트를 숫자 벡터로 변환합니다.
OpenAI의 text-embedding-3-small은 1536차원 벡터를 생성하며,
가성비가 좋아 RAG 파이프라인에서 널리 사용됩니다.""",
]


def main():
    print("=" * 60)
    print("RAG 파이프라인 예제")
    print("=" * 60)

    # 1. Config 생성
    config = Settings()
    mode = "Mock" if config.mock_mode else f"OpenAI ({config.llm_model})"
    store = config.vector_store
    print(f"모드: {mode} | 벡터스토어: {store}\n")

    # 2. 파이프라인 초기화
    print("파이프라인 초기화 중...")
    pipeline = RAGPipeline.from_config(config)

    # 3. 문서 색인
    print("문서 색인 중...")
    docs = load_texts(SAMPLE_DOCS)
    n = pipeline.index(docs)
    print(f"  → {n}개 청크 색인 완료\n")

    # 4. 질의응답
    queries = [
        "RAG가 환각을 줄이는 방법은?",
        "청크 크기 설정 시 고려할 점은?",
        "LangChain이 제공하는 주요 추상화는?",
    ]

    for q in queries:
        print(f"Q: {q}")
        result = pipeline.ask(q)
        print(f"A: {result['answer']}")
        print(f"   (검색된 청크 수: {len(result['contexts'])}개)")
        print()


if __name__ == "__main__":
    main()
