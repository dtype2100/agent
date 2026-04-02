"""
src/api/v1/document.py
───────────────────────
문서 인덱싱 API 엔드포인트 (v1).

이 파일은 순수 라우팅 레이어다.
모든 비즈니스 로직은 DocService에 위임한다.

엔드포인트 목록:
- POST /v1/documents/texts  : 인라인 텍스트 인덱싱
- POST /v1/documents/upload : 파일 업로드 및 인덱싱
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status

from src.core.config import get_settings
from src.core.dependencies import DocServiceDep
from src.core.exceptions import DocumentIndexError
from src.schemas.document import IndexResponse, IndexTextsRequest

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_SUFFIXES = {".txt", ".pdf", ".md"}


# ── Inline Texts ───────────────────────────────────────────────────────────────

@router.post("/texts", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def index_texts(request: IndexTextsRequest, service: DocServiceDep) -> IndexResponse:
    """인라인 텍스트 목록을 청킹하여 벡터 스토어에 인덱싱한다."""
    try:
        return await service.index_texts(request.texts)
    except DocumentIndexError as exc:
        raise exc.to_http() from exc


# ── File Upload ────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile, service: DocServiceDep) -> IndexResponse:
    """
    파일을 업로드하고 청킹하여 벡터 스토어에 인덱싱한다.

    지원 형식: .txt, .pdf, .md
    최대 파일 크기: MAX_UPLOAD_BYTES (기본 10 MB)
    """
    settings = get_settings()
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix!r}. Allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}",
        )

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.max_upload_bytes // 1024 // 1024} MB",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return await service.index_file(tmp_path, filename)
    except DocumentIndexError as exc:
        raise exc.to_http() from exc
    finally:
        tmp_path.unlink(missing_ok=True)
