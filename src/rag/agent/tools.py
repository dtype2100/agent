"""
src/rag/agent/tools.py
──────────────────────
에이전트 도구 정의 및 레지스트리.

도구는 @tool 데코레이터를 사용한 일반 함수로 정의한다.
LangGraph의 ToolNode와 함께 사용하기 위해
langchain_core.tools.BaseTool 인터페이스를 준수한다.

포함 도구:
- calculator          : AST 기반 수식 계산 (eval 미사용)
- web_search          : 웹 검색 스텁
- make_retrieval_tool : 문서 검색 도구 팩토리 (retriever 주입)

레지스트리:
- ToolRegistry : 활성 도구 목록 관리 및 LLM 바인딩
"""
from __future__ import annotations

import ast
import operator as op
from typing import Any

from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStoreRetriever


# ── 도구 정의 ──────────────────────────────────────────────────────────────────

_ALLOWED_OPS: dict[type, Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """
    수학 표현식을 안전하게 계산한다.

    AST를 파싱하여 허용된 연산자(+, -, *, /, **, %, //)만 실행한다.
    eval()을 사용하지 않아 코드 인젝션으로부터 안전하다.

    Parameters
    ----------
    expression : str
        계산할 수식 문자열 (최대 256자). 예: "2 ** 10 + 42"
    """
    if len(expression) > 256:
        return "Error: expression too long (max 256 chars)"
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def web_search(query: str) -> str:
    """
    웹 검색을 수행한다 (현재 스텁 구현).

    Parameters
    ----------
    query : str
        검색 쿼리 문자열.

    Notes
    -----
    실제 구현 시 Tavily, SerpAPI, DuckDuckGo 등의 API로 교체한다.
    의존성 예: ``pip install tavily-python``
    """
    return f"[Web search stub] No results for: {query!r}. Implement with Tavily or SerpAPI."


def make_retrieval_tool(retriever: VectorStoreRetriever) -> BaseTool:
    """
    주어진 retriever를 감싸는 문서 검색 도구를 생성한다.

    Parameters
    ----------
    retriever : VectorStoreRetriever
        contextual.build_retriever()로 생성한 검색기.

    Returns
    -------
    BaseTool
        LangGraph ToolNode와 호환되는 도구 인스턴스.
    """

    @tool
    def document_retrieval(query: str) -> str:
        """
        내부 문서 저장소에서 쿼리와 관련된 문서를 검색한다.

        Parameters
        ----------
        query : str
            검색 쿼리.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(
            f"[Source {i + 1}] {doc.page_content}" for i, doc in enumerate(docs)
        )

    return document_retrieval


# ── 레지스트리 ─────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    에이전트가 사용할 도구 목록을 관리한다.

    기본 도구: calculator
    선택 도구: document_retrieval (set_retriever() 호출 시 활성화)

    Methods
    -------
    set_retriever(retriever)   : 문서 검색 도구를 활성화한다.
    get_tools()                : 현재 활성 도구 목록을 반환한다.
    bind_to_llm(llm)           : LLM에 도구를 바인딩한 모델을 반환한다.
    dispatch(name, args)       : 도구 이름과 인자로 도구를 실행한다.
    """

    def __init__(self) -> None:
        self._base_tools: list[BaseTool] = [calculator]
        self._retrieval_tool: BaseTool | None = None

    def set_retriever(self, retriever: VectorStoreRetriever) -> None:
        """문서 검색 도구를 생성하여 레지스트리에 등록한다."""
        self._retrieval_tool = make_retrieval_tool(retriever)

    def get_tools(self) -> list[BaseTool]:
        tools = list(self._base_tools)
        if self._retrieval_tool:
            tools.append(self._retrieval_tool)
        return tools

    def bind_to_llm(self, llm):
        """LLM에 현재 도구 목록을 바인딩하여 반환한다."""
        return llm.bind_tools(self.get_tools())

    def dispatch(self, name: str, args: dict) -> str:
        """도구 이름으로 도구를 찾아 실행한다."""
        tool_map = {t.name: t for t in self.get_tools()}
        if name not in tool_map:
            return f"Unknown tool: {name!r}"
        return str(tool_map[name].invoke(args))
