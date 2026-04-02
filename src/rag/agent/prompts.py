"""
src/rag/agent/prompts.py
─────────────────────────
에이전트 프롬프트 템플릿 정의.

모든 프롬프트 문자열을 이 파일에 집중 관리하여
로직 코드(nodes.py)와 프롬프트 텍스트를 분리한다.

포함 항목:
- DEFAULT_SYSTEM_PROMPT : 기본 시스템 프롬프트
- RAG_SYSTEM_PROMPT     : 문서 기반 응답에 특화된 시스템 프롬프트
- build_rag_prompt()    : 검색 문서를 컨텍스트로 삽입한 프롬프트 생성
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest assistant. "
    "Answer questions clearly and concisely. "
    "If you are unsure about something, say so."
)

RAG_SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions based on provided documents.

Instructions:
- Use ONLY the information from the retrieved context documents to answer.
- If the context does not contain enough information, clearly state that you don't know.
- Always cite which part of the context supports your answer.
- Be concise and precise.

Context documents:
{context}
"""

TOOL_SYSTEM_PROMPT = """\
You are an intelligent assistant with access to tools.
Use the provided tools to find accurate information before answering.
Think step by step and use tools when needed to give precise, grounded answers.
"""

# ── 프롬프트 빌더 ──────────────────────────────────────────────────────────────

def build_chat_prompt(system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> ChatPromptTemplate:
    """
    일반 대화용 ChatPromptTemplate을 생성한다.

    Parameters
    ----------
    system_prompt : str
        시스템 프롬프트 텍스트.

    Returns
    -------
    ChatPromptTemplate
        [SystemMessage, MessagesPlaceholder("messages")] 구조의 프롬프트.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def build_rag_prompt(context_docs: list[str]) -> SystemMessage:
    """
    검색된 문서를 컨텍스트로 삽입한 RAG 시스템 메시지를 생성한다.

    Parameters
    ----------
    context_docs : list[str]
        검색된 문서 텍스트 목록.

    Returns
    -------
    SystemMessage
        컨텍스트가 포함된 시스템 메시지. AgentState.messages의 첫 번째로 삽입한다.
    """
    context = "\n\n---\n\n".join(
        f"[Document {i + 1}]\n{text}" for i, text in enumerate(context_docs)
    )
    return SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
