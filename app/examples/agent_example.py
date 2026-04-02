"""
Agent 예제.

사용법:
    # Mock 모드 (API 키 불필요)
    MOCK_MODE=true python app/examples/agent_example.py

    # OpenAI 실제 모드
    OPENAI_API_KEY=sk-... python app/examples/agent_example.py

    # Anthropic 모드 (langchain-anthropic 설치 필요)
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... \\
        LLM_MODEL=claude-sonnet-4-6 python app/examples/agent_example.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.agent.agent import Agent
from app.core.config import Settings
from app.rag.loader import load_texts
from app.rag.pipeline import RAGPipeline


def demo_calculator(agent: Agent):
    print("── 1. 계산기 도구 ─────────────────────────────")
    questions = [
        "2의 10제곱을 계산해줘",
        "방금 결과를 1000으로 나누면?",
    ]
    for q in questions:
        print(f"사용자: {q}")
        answer = agent.chat(q)
        print(f"에이전트: {answer}\n")


def demo_rag_integration(agent: Agent, config: Settings):
    print("── 2. RAG + Agent 통합 ────────────────────────")

    # RAG 파이프라인 구성 및 문서 색인
    pipeline = RAGPipeline.from_config(config)
    docs = load_texts([
        "파이썬은 1991년 귀도 반 로섬이 만든 고수준 프로그래밍 언어입니다. "
        "가독성과 간결함을 중시하며, 데이터 과학, 웹 개발, 자동화 등에 널리 사용됩니다.",
        "머신러닝은 데이터로부터 패턴을 학습하는 인공지능의 한 분야입니다. "
        "지도학습, 비지도학습, 강화학습으로 분류됩니다.",
        "트랜스포머(Transformer)는 2017년 'Attention is All You Need' 논문에서 제안된 "
        "신경망 아키텍처로, 현재 대부분의 대형 언어 모델의 기반이 됩니다.",
    ])
    pipeline.index(docs)

    # retriever를 에이전트에 연결
    agent.set_retriever(pipeline.retriever)
    agent.reset()

    questions = [
        "파이썬 언어에 대해 문서에서 찾아서 알려줘",
        "트랜스포머 아키텍처가 뭔지 설명해줘",
    ]
    for q in questions:
        print(f"사용자: {q}")
        answer = agent.chat(q)
        print(f"에이전트: {answer}\n")


def main():
    print("=" * 60)
    print("Agent 예제")
    print("=" * 60)

    config = Settings()
    mode = "Mock" if config.mock_mode else f"{config.llm_provider} ({config.llm_model})"
    print(f"모드: {mode}\n")

    agent = Agent(config)

    demo_calculator(agent)
    demo_rag_integration(agent, config)


if __name__ == "__main__":
    main()
