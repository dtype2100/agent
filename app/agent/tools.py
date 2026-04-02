import ast
import operator

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool

_MAX_EXPR_LEN = 256

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _eval_ast(node: ast.AST) -> float:
    """숫자 연산 AST만 평가한다 (`eval` 미사용)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("숫자가 아닌 상수는 사용할 수 없습니다.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_ast(node.operand)
    if isinstance(node, ast.BinOp):
        op_fn = _BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError("허용되지 않는 이항 연산입니다.")
        return op_fn(_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError("허용되지 않는 표현식입니다.")


@tool
def calculator(expression: str) -> str:
    """수학 표현식을 계산합니다 (덧셈·뺄셈·곱셈·나눗셈·거듭제곱 등, `eval` 없음).

    예시: calculator(expression='(10 + 5) * 2')
    """
    expr = (expression or "").strip()
    if not expr:
        return "오류: 빈 식입니다."
    if len(expr) > _MAX_EXPR_LEN:
        return f"오류: 식이 너무 깁니다 (최대 {_MAX_EXPR_LEN}자)."

    try:
        tree = ast.parse(expr, mode="eval")
        if not isinstance(tree, ast.Expression):
            return "오류: 잘못된 식입니다."
        value = _eval_ast(tree.body)
        if value == int(value):
            return str(int(value))
        return str(value)
    except ZeroDivisionError:
        return "오류: 0으로 나눌 수 없습니다."
    except (SyntaxError, ValueError, TypeError) as e:
        return f"오류: {e}"


@tool
def web_search(query: str) -> str:
    """웹에서 최신 정보를 검색합니다.
    현재는 stub입니다 — SerpAPI, Tavily 등의 실제 검색 API로 교체하세요.
    """
    return (
        f"[web_search stub] '{query}'에 대한 실제 결과가 없습니다. "
        "실제 검색 API(SerpAPI, Tavily 등)를 연동하세요."
    )


def _make_document_retrieval_tool(retriever: BaseRetriever):
    """실제 retriever를 클로저로 감싼 document_retrieval 도구를 생성한다."""

    @tool
    def document_retrieval(query: str) -> str:
        """지식 베이스에서 쿼리와 관련된 문서 청크를 검색합니다."""
        docs = retriever.invoke(query)
        if not docs:
            return "관련 문서를 찾지 못했습니다."
        return "\n\n".join(
            f"[출처: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    return document_retrieval


class ToolRegistry:
    """도구 목록을 관리하고 LLM에 바인딩하는 레지스트리."""

    def __init__(self):
        # web_search는 stub이므로 기본 도구에서 제외한다.
        # 실제 검색 API를 연동한 뒤 self._tools.append(web_search)로 추가할 것.
        self._tools = [calculator]
        self._retrieval_tool = None

    @property
    def tools(self) -> list:
        tools = list(self._tools)
        if self._retrieval_tool is not None:
            tools.append(self._retrieval_tool)
        return tools

    def set_retriever(self, retriever: BaseRetriever) -> None:
        """실제 retriever를 주입하여 document_retrieval 도구를 활성화한다."""
        self._retrieval_tool = _make_document_retrieval_tool(retriever)

    def bind_to_llm(self, llm: BaseChatModel) -> BaseChatModel:
        """도구 목록을 LLM에 바인딩하여 반환한다."""
        return llm.bind_tools(self.tools)

    def dispatch(self, tool_name: str, tool_args: dict) -> str:
        """도구 이름과 인자로 도구를 실행하고 결과를 반환한다."""
        for t in self.tools:
            if t.name == tool_name:
                return str(t.invoke(tool_args))
        return f"오류: 알 수 없는 도구 '{tool_name}'"
