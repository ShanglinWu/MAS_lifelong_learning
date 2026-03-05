"""MARBLE A-MEM integration package."""

from __future__ import annotations

import importlib
from typing import List

from marble.memory.amem.manager import AMEMManager
from marble.memory.amem.schemas import AMEMTopology
from marble.memory.amem.topology import AMEMTopologyManager


def ensure_amem_dependencies() -> None:
    """Raise ImportError with missing package details if runtime deps are absent."""
    required = [
        "chromadb",
        "sentence_transformers",
    ]
    missing: List[str] = []
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        joined = ", ".join(sorted(missing))
        raise ImportError(
            f"Missing A-MEM runtime dependencies: {joined}. "
            "Install with `poetry install --with amem`."
        )


__all__ = [
    "AMEMManager",
    "AMEMTopology",
    "AMEMTopologyManager",
    "ensure_amem_dependencies",
]
