"""에이전트 세션을 TTL·최대 개수와 함께 보관한다."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any


class SessionStore:
    """세션 ID별 에이전트 인스턴스를 보관하고, TTL 초과·용량 초과 시 정리한다."""

    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600) -> None:
        self.max_sessions = max(1, max_sessions)
        self.ttl_seconds = max(1, ttl_seconds)
        self._data: dict[str, tuple[Any, float]] = {}

    def _monotonic(self) -> float:
        return time.monotonic()

    def _prune_expired(self) -> None:
        now = self._monotonic()
        dead = [sid for sid, (_, t) in self._data.items() if now - t > self.ttl_seconds]
        for sid in dead:
            del self._data[sid]

    def _evict_oldest(self) -> None:
        if not self._data:
            return
        oldest_sid = min(self._data.items(), key=lambda x: x[1][1])[0]
        del self._data[oldest_sid]

    def get_or_create(
        self,
        session_id: str | None,
        factory: Callable[[], Any],
    ) -> tuple[str, Any]:
        """세션을 조회하거나 새로 만들고, 마지막 접근 시각을 갱신한다."""
        self._prune_expired()
        now = self._monotonic()
        sid = session_id or str(uuid.uuid4())
        if sid in self._data:
            agent, _ = self._data[sid]
            self._data[sid] = (agent, now)
            return sid, agent
        while len(self._data) >= self.max_sessions:
            self._evict_oldest()
        agent = factory()
        self._data[sid] = (agent, now)
        return sid, agent

    def get(self, session_id: str) -> Any | None:
        """세션이 있으면 에이전트를 반환하고 접근 시각을 갱신한다."""
        self._prune_expired()
        if session_id not in self._data:
            return None
        agent, _ = self._data[session_id]
        self._data[session_id] = (agent, self._monotonic())
        return agent

    def reset_agent(self, session_id: str) -> bool:
        """에이전트 대화 이력만 초기화한다. 없으면 False."""
        agent = self.get(session_id)
        if agent is None:
            return False
        reset_fn = getattr(agent, "reset", None)
        if callable(reset_fn):
            reset_fn()
        return True

    def __len__(self) -> int:
        self._prune_expired()
        return len(self._data)

    def clear(self) -> None:
        """모든 세션을 제거한다 (종료 시 정리용)."""
        self._data.clear()
