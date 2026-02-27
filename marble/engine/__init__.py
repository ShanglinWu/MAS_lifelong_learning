from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine


def __getattr__(name: str):
    if name == "Engine":
        from .engine import Engine as _Engine

        return _Engine
    raise AttributeError(name)

__all__ = [
    "Engine",
]
