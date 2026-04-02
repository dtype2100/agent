"""
src/rag/ingestion/layout_parser.py
───────────────────────────────────
레이아웃 인식 문서 파싱 스캐폴딩.

PDF, DOCX 등의 문서에서 제목, 단락, 표, 이미지 캡션 등의
레이아웃 구조를 보존하며 파싱한다.

지원 예정 백엔드:
- Unstructured (unstructured.io) : 레이아웃 감지 + 요소 분류
- Azure Document Intelligence     : 클라우드 기반 고정밀 파싱
- PDFMiner / pdfplumber           : 로컬 레이아웃 파싱

주요 클래스:
- LayoutElement  : 파싱된 레이아웃 요소 (타입 + 내용 + 위치)
- LayoutParser   : 레이아웃 파서 추상 인터페이스
- UnstructuredLayoutParser : Unstructured 백엔드 구현 스캐폴딩
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from langchain_core.documents import Document


class ElementType(str, Enum):
    TITLE = "title"
    NARRATIVE_TEXT = "narrative_text"
    TABLE = "table"
    IMAGE = "image"
    LIST_ITEM = "list_item"
    HEADER = "header"
    FOOTER = "footer"
    UNKNOWN = "unknown"


@dataclass
class LayoutElement:
    """파싱된 레이아웃 단위 요소."""
    element_type: ElementType
    text: str
    page_number: int = 0
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)
    metadata: dict = field(default_factory=dict)

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "element_type": self.element_type.value,
                "page_number": self.page_number,
                **self.metadata,
            },
        )


class LayoutParser(ABC):
    """레이아웃 파서 추상 베이스 클래스."""

    @abstractmethod
    def parse(self, file_path: str | Path) -> list[LayoutElement]:
        """
        파일을 파싱하여 레이아웃 요소 목록을 반환한다.

        Parameters
        ----------
        file_path : str or Path
            파싱할 파일 경로 (.pdf, .docx 등).
        """
        ...

    def parse_to_documents(self, file_path: str | Path) -> list[Document]:
        """parse() 결과를 LangChain Document 리스트로 변환한다."""
        return [el.to_document() for el in self.parse(file_path)]


class UnstructuredLayoutParser(LayoutParser):
    """
    Unstructured 라이브러리 기반 레이아웃 파서.

    Parameters
    ----------
    strategy : str
        파싱 전략 ("fast" | "hi_res" | "ocr_only").
        - "fast"    : 텍스트 레이어만 추출 (빠름)
        - "hi_res"  : 레이아웃 감지 모델 사용 (정확, 느림)
        - "ocr_only": OCR 전용 (스캔 문서)

    Notes
    -----
    의존성: ``pip install unstructured[pdf]``
    """

    def __init__(self, strategy: str = "fast") -> None:
        self._strategy = strategy

    def parse(self, file_path: str | Path) -> list[LayoutElement]:
        try:
            from unstructured.partition.auto import partition  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("unstructured is required: pip install 'unstructured[pdf]'") from exc

        elements = partition(filename=str(file_path), strategy=self._strategy)
        result: list[LayoutElement] = []
        for el in elements:
            el_type = ElementType(getattr(el, "category", "unknown").lower())
            result.append(
                LayoutElement(
                    element_type=el_type,
                    text=str(el),
                    page_number=getattr(el.metadata, "page_number", 0) or 0,
                )
            )
        return result
