"""Schemas and enums for MARBLE A-MEM integration."""

from enum import Enum


class AMEMTopology(str, Enum):
    LOCAL = "local"
    SHARED = "shared"
