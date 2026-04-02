from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings


def load_documents(paths: List[Union[str, Path]]) -> List[Document]:
    """텍스트(.txt) 및 PDF(.pdf) 파일을 LangChain Document 리스트로 로드한다."""
    from langchain_community.document_loaders import TextLoader

    docs: List[Document] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
        if p.suffix.lower() == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
            except ImportError:
                raise ImportError(
                    "PDF 로딩에는 pypdf가 필요합니다.\n"
                    "pip install pypdf 를 실행한 후 다시 시도하세요."
                )
            loader = PyPDFLoader(str(p))
        else:
            loader = TextLoader(str(p), encoding="utf-8", autodetect_encoding=True)
        docs.extend(loader.load())
    return docs


def load_texts(texts: List[str]) -> List[Document]:
    """문자열 리스트를 Document 리스트로 변환한다."""
    return [Document(page_content=t, metadata={"source": "inline"}) for t in texts]


def split_documents(docs: List[Document], config: Settings) -> List[Document]:
    """문서를 청크로 분할한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(docs)
