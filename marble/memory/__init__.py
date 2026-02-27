"""Memory package exports with lazy loading to avoid eager heavy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_memory import BaseMemory
    from .llma_mem import LLMAMemManager, MemoryTopology, MemoryTopologyManager, WerewolfMemoryAdapter
    from .long_term_memory import LongTermMemory
    from .shared_memory import SharedMemory
    from .short_term_memory import ShortTermMemory


def __getattr__(name: str):
    if name == "BaseMemory":
        from .base_memory import BaseMemory as _BaseMemory

        return _BaseMemory
    if name == "SharedMemory":
        from .shared_memory import SharedMemory as _SharedMemory

        return _SharedMemory
    if name == "LongTermMemory":
        from .long_term_memory import LongTermMemory as _LongTermMemory

        return _LongTermMemory
    if name == "ShortTermMemory":
        from .short_term_memory import ShortTermMemory as _ShortTermMemory

        return _ShortTermMemory
    if name in {
        "LLMAMemManager",
        "MemoryTopologyManager",
        "MemoryTopology",
        "WerewolfMemoryAdapter",
    }:
        from .llma_mem import (
            LLMAMemManager,
            MemoryTopology,
            MemoryTopologyManager,
            WerewolfMemoryAdapter,
        )

        mapping = {
            "LLMAMemManager": LLMAMemManager,
            "MemoryTopologyManager": MemoryTopologyManager,
            "MemoryTopology": MemoryTopology,
            "WerewolfMemoryAdapter": WerewolfMemoryAdapter,
        }
        return mapping[name]
    raise AttributeError(name)


__all__ = [
    "BaseMemory",
    "SharedMemory",
    "LongTermMemory",
    "ShortTermMemory",
    "LLMAMemManager",
    "MemoryTopologyManager",
    "MemoryTopology",
    "WerewolfMemoryAdapter",
]
