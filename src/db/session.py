"""
src/db/session.py
─────────────────
세션 레지스트리.

LangGraph MemorySaver가 대화 상태(messages)를 관리하므로,
여기서는 session_id의 메타데이터(생성 시각, 마지막 접근)만 추적한다.

주요 기능:
- TTL 기반 만료 및 자동 정리
- 최대 세션 수 초과 시 LRU 방식 퇴출
- thread-safe (RLock + per-session Lock)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class SessionMeta:
    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_accessed_at: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_accessed_at = time.monotonic()

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.monotonic() - self.last_accessed_at) > ttl_seconds


class SessionStore:
    """
    세션 메타데이터 저장소.

    Parameters
    ----------
    max_sessions : int
        동시 보유 가능한 최대 세션 수. 초과 시 가장 오래된 세션 퇴출.
    ttl_seconds : float
        마지막 접근 이후 세션 만료까지의 시간(초).
    """

    def __init__(self, max_sessions: int = 1000, ttl_seconds: float = 3600) -> None:
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._data: dict[str, SessionMeta] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._store_lock = threading.RLock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def touch(self, session_id: str) -> SessionMeta:
        """세션을 갱신하거나 신규 등록한다."""
        with self._store_lock:
            self._prune_expired()
            if session_id not in self._data:
                if len(self._data) >= self._max:
                    self._evict_oldest()
                self._data[session_id] = SessionMeta(session_id=session_id)
                self._locks[session_id] = threading.Lock()
            meta = self._data[session_id]
            meta.touch()
            return meta

    def exists(self, session_id: str) -> bool:
        with self._store_lock:
            meta = self._data.get(session_id)
            return meta is not None and not meta.is_expired(self._ttl)

    def remove(self, session_id: str) -> None:
        with self._store_lock:
            self._data.pop(session_id, None)
            self._locks.pop(session_id, None)

    def get_lock(self, session_id: str) -> threading.Lock:
        with self._store_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    def active_ids(self) -> list[str]:
        with self._store_lock:
            self._prune_expired()
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._store_lock:
            return len(self._data)

    def clear(self) -> None:
        with self._store_lock:
            self._data.clear()
            self._locks.clear()

    def __iter__(self) -> Iterator[SessionMeta]:
        with self._store_lock:
            return iter(list(self._data.values()))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _prune_expired(self) -> None:
        expired = [sid for sid, m in self._data.items() if m.is_expired(self._ttl)]
        for sid in expired:
            self._data.pop(sid, None)
            self._locks.pop(sid, None)

    def _evict_oldest(self) -> None:
        if not self._data:
            return
        oldest = min(self._data.values(), key=lambda m: m.last_accessed_at)
        self._data.pop(oldest.session_id, None)
        self._locks.pop(oldest.session_id, None)
