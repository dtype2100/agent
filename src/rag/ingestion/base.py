"""
src/rag/ingestion/base.py
─────────────────────────
기본 문서 로딩 및 텍스트 분할 유틸리티.

Functions
---------
load_documents(paths)         : 파일 경로 목록 → Document 리스트 (.txt / .pdf / .md)
load_texts(texts)             : 문자열 리스트 → Document 리스트 (인라인 텍스트)
split_documents(docs, config) : RecursiveCharacterTextSplitter로 청킹
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Settings


def load_documents(paths: list[str | Path]) -> list[Document]:
    """
    파일 경로 목록을 읽어 Document 리스트를 반환한다.

    지원 형식:
    - .txt / .md : TextLoader (UTF-8)
    - .pdf       : PyPDFLoader (pypdf 필요)

    Parameters
    ----------
    paths : list of str or Path
        읽을 파일 경로 목록.
    """
    docs: list[Document] = []
    for p in paths:
        path = Path(p)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader  # noqa: PLC0415
            loader = PyPDFLoader(str(path))
        else:
            from langchain_community.document_loaders import TextLoader  # noqa: PLC0415
            loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def load_texts(texts: list[str]) -> list[Document]:
    """
    문자열 리스트를 Document 리스트로 변환한다.

    Parameters
    ----------
    texts : list of str
        인라인 텍스트 목록. 각 항목이 독립된 Document가 된다.
    """
    return [Document(page_content=t, metadata={"source": "inline"}) for t in texts]


def split_documents(docs: list[Document], config: Settings) -> list[Document]:
    """
    RecursiveCharacterTextSplitter로 문서를 청킹한다.

    Parameters
    ----------
    docs   : list of Document
        분할할 원본 문서 목록.
    config : Settings
        chunk_size, chunk_overlap 값을 사용한다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(docs)
