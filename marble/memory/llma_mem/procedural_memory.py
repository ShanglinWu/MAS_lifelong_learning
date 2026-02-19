"""Procedural memory store for LLMA-Mem."""

import threading
import time
from typing import Dict, List, Optional

from .schemas import ProceduralRecord


class ProceduralMemory:
    """Thread-safe procedural memory storing learned strategies and patterns."""

    ALPHA = 1.0

    def __init__(self):
        self._records: Dict[str, ProceduralRecord] = {}
        self._lock = threading.Lock()

    def add_procedure(self, record: ProceduralRecord) -> None:
        """Add or overwrite a procedural record."""
        now = time.time()
        if record.timestamp_created <= 0:
            record.timestamp_created = now
        record.timestamp_updated = now
        if not record.timestamps:
            record.timestamps.append(now)
        record.compute_confidence(self.ALPHA)
        with self._lock:
            self._records[record.procedure_id] = record

    def get_procedure(self, procedure_id: str) -> Optional[ProceduralRecord]:
        """Retrieve a procedure by ID."""
        with self._lock:
            return self._records.get(procedure_id)

    def get_all_procedures(self) -> List[ProceduralRecord]:
        """Return all procedures."""
        with self._lock:
            return list(self._records.values())

    def update_procedure_stats(
        self, procedure_id: str, success: bool
    ) -> None:
        """Update success/failure counts and recompute confidence."""
        with self._lock:
            rec = self._records.get(procedure_id)
            if rec is None:
                return
            if success:
                rec.success_count += 1
            else:
                rec.failure_count += 1
            now = time.time()
            rec.timestamp_updated = now
            rec.timestamps.append(now)
            rec.compute_confidence(self.ALPHA)

    def increment_viewed(
        self, procedure_id: str, current_episode_count: Optional[int] = None
    ) -> None:
        """Increment the viewed_times counter for a procedure."""
        with self._lock:
            rec = self._records.get(procedure_id)
            if rec is not None:
                rec.viewed_times += 1
                rec.last_used = time.time()
                if current_episode_count is not None:
                    rec.last_used_episode = current_episode_count

    def prune(
        self,
        min_confidence: float = 0.3,
        current_episode_count: Optional[int] = None,
        min_unused_episodes: int = 5,
    ) -> int:
        """
        Remove procedures that are low-confidence and stale.

        A procedure is stale when it hasn't been used for at least
        `min_unused_episodes` episodes.
        """
        with self._lock:
            to_remove = []
            for pid, rec in self._records.items():
                if rec.confidence_score >= min_confidence:
                    continue
                if current_episode_count is None:
                    to_remove.append(pid)
                    continue
                last_used_episode = rec.last_used_episode
                if last_used_episode < 0:
                    stale = current_episode_count >= min_unused_episodes
                else:
                    stale = (
                        current_episode_count - last_used_episode
                    ) >= min_unused_episodes
                if stale:
                    to_remove.append(pid)
            for pid in to_remove:
                del self._records[pid]
            return len(to_remove)

    def merge_similar(
        self, pid_a: str, pid_b: str
    ) -> Optional[ProceduralRecord]:
        """Merge two similar procedures, keeping the one with higher confidence."""
        with self._lock:
            a = self._records.get(pid_a)
            b = self._records.get(pid_b)
            if a is None or b is None:
                return None

            # Keep the higher-confidence one, accumulate stats
            if a.confidence_score >= b.confidence_score:
                keeper, removed = a, b
                removed_id = pid_b
            else:
                keeper, removed = b, a
                removed_id = pid_a

            keeper.success_count += removed.success_count
            keeper.failure_count += removed.failure_count
            keeper.source_episodes.extend(removed.source_episodes)
            keeper.viewed_times += removed.viewed_times
            keeper.last_used = max(keeper.last_used, removed.last_used)
            keeper.last_used_episode = max(
                keeper.last_used_episode, removed.last_used_episode
            )
            now = time.time()
            keeper.timestamp_updated = now
            keeper.timestamps.append(now)
            keeper.compute_confidence(self.ALPHA)
            del self._records[removed_id]
            return keeper

    def count(self) -> int:
        """Return total number of procedures."""
        with self._lock:
            return len(self._records)
