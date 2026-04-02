"""
src/rag/ingestion/pattern_parser.py
────────────────────────────────────
패턴/정규식 기반 문서 파싱 스캐폴딩.

구조화된 텍스트 형식(Markdown, HTML, 로그, CSV)에서
정규식 또는 파싱 규칙으로 섹션을 추출한다.

주요 클래스:
- PatternParser        : 패턴 파서 추상 인터페이스
- MarkdownSectionParser: Markdown 헤더 기반 섹션 분리
- RegexPatternParser   : 사용자 정의 정규식 패턴으로 텍스트 추출
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


@dataclass
class ParsedSection:
    """파싱된 섹션 단위."""
    title: str
    content: str
    level: int = 0  # 헤더 계층 (0 = 최상위)
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={"section_title": self.title, "level": self.level, **self.metadata},
        )


class PatternParser(ABC):
    """패턴 파서 추상 베이스 클래스."""

    @abstractmethod
    def parse(self, text: str) -> list[ParsedSection]:
        """
        텍스트를 파싱하여 섹션 목록을 반환한다.

        Parameters
        ----------
        text : str
            파싱할 원본 텍스트.
        """
        ...

    def parse_file(self, file_path: str | Path, encoding: str = "utf-8") -> list[ParsedSection]:
        """파일을 읽어 parse()를 호출한다."""
        return self.parse(Path(file_path).read_text(encoding=encoding))

    def parse_to_documents(self, text: str) -> list[Document]:
        return [sec.to_document() for sec in self.parse(text)]


class MarkdownSectionParser(PatternParser):
    """
    Markdown 헤더(#, ##, ###)를 기준으로 섹션을 분리하는 파서.

    Parameters
    ----------
    max_level : int
        분리할 최대 헤더 레벨 (기본값: 3, 즉 h1~h3).
    """

    def __init__(self, max_level: int = 3) -> None:
        self._max_level = max_level
        self._header_re = re.compile(r"^(#{1,%d})\s+(.+)$" % max_level, re.MULTILINE)

    def parse(self, text: str) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        matches = list(self._header_re.finditer(text))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                sections.append(ParsedSection(title=title, content=content, level=level))

        return sections


class RegexPatternParser(PatternParser):
    """
    사용자 정의 정규식 패턴으로 텍스트에서 섹션을 추출하는 파서.

    Parameters
    ----------
    section_pattern : str
        섹션 구분자를 찾는 정규식. 그룹 1 = 제목, 그룹 2 = 내용 (선택).
    flags : int
        re 모듈 플래그 (기본값: re.MULTILINE).

    Example
    -------
    # 로그 파일에서 ERROR 블록 추출
    parser = RegexPatternParser(r"^(ERROR \\d{4}-\\d{2}-\\d{2}.+?)(?=^ERROR|\\Z)")
    """

    def __init__(self, section_pattern: str, flags: int = re.MULTILINE | re.DOTALL) -> None:
        self._pattern = re.compile(section_pattern, flags)

    def parse(self, text: str) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        for i, match in enumerate(self._pattern.finditer(text)):
            groups = match.groups()
            title = groups[0].strip() if groups else f"section_{i}"
            content = groups[1].strip() if len(groups) > 1 else match.group(0).strip()
            sections.append(ParsedSection(title=title, content=content))
        return sections
