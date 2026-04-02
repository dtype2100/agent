"""
src/db/relational_db.py
───────────────────────
관계형 DB 연동 스캐폴딩 (SQLAlchemy 기반).

문서 메타데이터, 사용자 피드백, 평가 결과 등 구조화된 데이터를
저장하기 위한 레이어.

주요 컴포넌트:
- Base          : SQLAlchemy declarative base
- DocumentRecord: 인덱싱된 문서 메타데이터 테이블
- get_engine()  : DB 엔진 팩토리
- get_session() : SQLAlchemy AsyncSession 컨텍스트 매니저
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy ORM 베이스 클래스."""
    pass


class DocumentRecord(Base):
    """
    인덱싱된 문서 청크 메타데이터 테이블.

    Columns
    -------
    id         : 내부 기본키
    source     : 원본 파일명 또는 URL
    chunk_index: 문서 내 청크 순서
    content    : 청크 텍스트 내용
    created_at : 인덱싱 시각
    """
    __tablename__ = "document_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def get_engine(database_url: str = "sqlite+aiosqlite:///./app.db"):
    """
    SQLAlchemy 비동기 엔진을 생성한다.

    Parameters
    ----------
    database_url : str
        SQLAlchemy 연결 URL.
        기본값: 로컬 SQLite (``sqlite+aiosqlite:///./app.db``)
        PostgreSQL 예: ``postgresql+asyncpg://user:pass@host/db``

    Notes
    -----
    의존성: ``pip install sqlalchemy aiosqlite``  (SQLite의 경우)
             ``pip install sqlalchemy asyncpg``    (PostgreSQL의 경우)
    """
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415
    return create_async_engine(database_url, echo=False)


@asynccontextmanager
async def get_session(engine) -> AsyncGenerator:
    """
    비동기 DB 세션 컨텍스트 매니저.

    Usage
    -----
    async with get_session(engine) as session:
        session.add(record)
        await session.commit()
    """
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    async with AsyncSession(engine, expire_on_commit=False) as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables(engine) -> None:
    """테이블이 없으면 생성한다 (앱 시작 시 호출)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
