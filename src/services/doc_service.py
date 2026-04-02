"""
src/services/doc_service.py
────────────────────────────
문서 인덱싱 비즈니스 로직 서비스.

인라인 텍스트 및 파일 업로드 문서를 청킹하여 벡터 스토어에 인덱싱한다.

주요 메서드:
- index_texts(texts)     → IndexResponse (인라인 텍스트 인덱싱)
- index_file(path, name) → IndexResponse (업로드 파일 인덱싱)
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.vectorstores import VectorStore

from src.core.config import Settings
from src.core.exceptions import DocumentIndexError
from src.rag.ingestion.base import load_documents, load_texts, split_documents
from src.schemas.document import IndexResponse


class DocService:
    """
    문서 인덱싱 서비스.

    Parameters
    ----------
    vector_store : VectorStore
        문서 청크를 저장할 벡터 스토어.
    settings     : Settings
        청킹 파라미터(chunk_size, chunk_overlap) 등.
    """

    def __init__(self, vector_store: VectorStore, settings: Settings) -> None:
        self._store = vector_store
        self._settings = settings

    async def index_texts(self, texts: list[str]) -> IndexResponse:
        """
        인라인 텍스트 목록을 청킹하여 벡터 스토어에 인덱싱한다.

        Parameters
        ----------
        texts : list[str]
            인덱싱할 텍스트 목록.
        """
        try:
            docs = load_texts(texts)
            chunks = split_documents(docs, self._settings)
            await asyncio.to_thread(self._store.add_documents, chunks)
            return IndexResponse(indexed=len(chunks))
        except Exception as exc:
            raise DocumentIndexError(f"Failed to index texts: {exc}") from exc

    async def index_file(self, file_path: str | Path, filename: str) -> IndexResponse:
        """
        파일을 읽어 청킹하고 벡터 스토어에 인덱싱한다.

        Parameters
        ----------
        file_path : str or Path
            인덱싱할 파일의 경로.
        filename  : str
            원본 파일명 (응답 메타데이터용).

        Raises
        ------
        DocumentIndexError
            지원하지 않는 파일 형식이거나 인덱싱 중 오류 발생 시.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        allowed = {".txt", ".pdf", ".md"}

        if suffix not in allowed:
            raise DocumentIndexError(
                f"Unsupported file type: {suffix!r}. Allowed: {', '.join(sorted(allowed))}"
            )

        try:
            docs = await asyncio.to_thread(load_documents, [path])
            chunks = split_documents(docs, self._settings)
            await asyncio.to_thread(self._store.add_documents, chunks)
            return IndexResponse(indexed=len(chunks), filename=filename)
        except DocumentIndexError:
            raise
        except Exception as exc:
            raise DocumentIndexError(f"Failed to index file {filename!r}: {exc}") from exc
