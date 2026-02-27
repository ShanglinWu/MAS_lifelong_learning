"""Pure helpers for selecting advanced memory mode."""

from __future__ import annotations

from typing import Any, Dict


def resolve_advanced_memory_mode(memory_config: Dict[str, Any]) -> str:
    """Return one of: 'none', 'llmamem', 'amem'.

    Raises:
        ValueError: if both llma_mem and amem are enabled.
    """
    llma_enabled = bool(memory_config.get("llma_mem", {}).get("enabled", False))
    amem_enabled = bool(memory_config.get("amem", {}).get("enabled", False))

    if llma_enabled and amem_enabled:
        raise ValueError(
            "Invalid memory config: only one of memory.llma_mem.enabled and "
            "memory.amem.enabled can be true."
        )

    if llma_enabled:
        return "llmamem"
    if amem_enabled:
        return "amem"
    return "none"
