import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.dependencies import PipelineDep, SettingsDep
from app.rag.loader import load_documents, load_texts
from app.schemas.documents import IndexResponse, IndexTextsRequest

router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md"}


@router.post("/texts", response_model=IndexResponse, summary="텍스트 색인")
async def index_texts(body: IndexTextsRequest, pipeline: PipelineDep):
    """문자열 리스트를 청킹 후 벡터스토어에 색인한다."""

    def _run():
        docs = load_texts(body.texts)
        return pipeline.index(docs)

    n = await asyncio.to_thread(_run)
    return IndexResponse(indexed=n)


@router.post("/upload", response_model=IndexResponse, summary="파일 업로드 및 색인")
async def upload_document(
    file: UploadFile,
    pipeline: PipelineDep,
    settings: SettingsDep,
):
    """텍스트(.txt/.md) 또는 PDF 파일을 업로드하여 벡터스토어에 색인한다."""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"지원하지 않는 파일 형식입니다. 허용: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    max_bytes = settings.max_upload_bytes
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"파일이 너무 큽니다. 최대 {max_bytes} bytes (MAX_UPLOAD_BYTES)",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    def _index():
        try:
            docs = load_documents([tmp_path])
            return pipeline.index(docs)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    n = await asyncio.to_thread(_index)
    return IndexResponse(indexed=n, filename=filename)
