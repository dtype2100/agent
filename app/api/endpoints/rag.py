import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.dependencies import PipelineDep
from app.schemas.rag import QueryRequest, QueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse, summary="RAG 질의응답")
async def query(body: QueryRequest, pipeline: PipelineDep):
    """관련 문서를 검색하고 LLM으로 답변을 생성한다."""

    try:
        result = await asyncio.to_thread(pipeline.ask, body.query)
    except Exception:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=502,
            detail="검색 또는 LLM 호출에 실패했습니다.",
        ) from None

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        contexts=result["contexts"],
    )


@router.post("/query/stream", summary="RAG 질의응답 (SSE)")
async def query_stream(body: QueryRequest, pipeline: PipelineDep):
    """SSE로 응답을 전송한다.

    **주의**: LLM이 토큰 단위로 스트리밍하는 것이 아니라, 전체 답변을 만든 뒤
    컨텍스트·답변을 순서대로 잘라 보냅니다.

    Events:
        {"type": "context", "index": int, "content": str}
        {"type": "token",   "content": str}
        {"type": "done",    "answer": str, "contexts": list[str]}
        {"type": "error",   "message": str}
    """

    async def generate():
        try:
            result = await asyncio.to_thread(pipeline.ask, body.query)
        except Exception:
            logger.exception("RAG stream query failed")
            yield f"data: {json.dumps({'type': 'error', 'message': '검색 또는 LLM 호출에 실패했습니다.'}, ensure_ascii=False)}\n\n"
            return

        for i, ctx in enumerate(result["contexts"]):
            yield f"data: {json.dumps({'type': 'context', 'index': i, 'content': ctx}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'token', 'content': result['answer']}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'answer': result['answer'], 'contexts': result['contexts']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
