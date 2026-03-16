"""
Null memory module — a no-op memory implementation that satisfies both the
SharedMemory and BaseMemory interfaces without storing anything.

Use this when you want to disable memory entirely via the config file:

    memory:
      type: NoMemory
"""

from typing import Any, Dict, List, Optional


class NullMemory:
    """
    A memory class that discards all writes and returns empty / None on reads.

    It implements every method found in both SharedMemory and BaseMemory so it
    can be used as a drop-in replacement for either without raising AttributeErrors.
    """

    # ------------------------------------------------------------------
    # SharedMemory interface
    # ------------------------------------------------------------------

    def update(self, key: str, information: Any) -> None:
        """Accept a write but do nothing."""
        pass

    def retrieve(self, key: str) -> None:
        """Always return None."""
        return None

    def retrieve_all(self) -> Dict[str, Any]:
        """Return an empty dict (SharedMemory style)."""
        return {}

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def retrieve_latest(self) -> None:
        """Always return None."""
        return None

    def get_memory_str(self) -> str:
        """Return an empty string so prompt construction stays intact."""
        return ""

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return "NullMemory (memory disabled)"

    def __repr__(self) -> str:
        return "NullMemory()"
