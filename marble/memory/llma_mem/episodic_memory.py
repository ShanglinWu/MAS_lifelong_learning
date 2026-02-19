"""Episodic memory store for LLMA-Mem."""

import threading
from typing import List, Optional

from .schemas import EpisodicRecord


class EpisodicMemory:
    """Thread-safe episodic memory storing sequential experience records."""

    def __init__(self):
        self._records: List[EpisodicRecord] = []
        self._lock = threading.Lock()

    def add_episode(self, record: EpisodicRecord) -> None:
        """Add an episodic record."""
        with self._lock:
            self._records.append(record)

    def get_episodes(self) -> List[EpisodicRecord]:
        """Return all episodes."""
        with self._lock:
            return list(self._records)

    def get_episodes_by_agent(self, agent_id: str) -> List[EpisodicRecord]:
        """Return episodes for a specific agent."""
        with self._lock:
            return [r for r in self._records if r.agent_id == agent_id]

    def get_recent(self, n: int = 5) -> List[EpisodicRecord]:
        """Return the most recent n episodes."""
        with self._lock:
            return list(self._records[-n:])

    def count(self) -> int:
        """Return total number of episodes."""
        with self._lock:
            return len(self._records)
