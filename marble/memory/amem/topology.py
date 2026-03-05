"""Topology manager for MARBLE A-MEM integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from marble.memory.amem.schemas import AMEMTopology

if TYPE_CHECKING:
    from marble.memory.amem_vendor import AgenticMemorySystem


class AMEMTopologyManager:
    """Resolves per-agent or shared A-MEM instances based on topology."""

    def __init__(
        self,
        topology: AMEMTopology,
        embedding_model: str,
        llm_backend: str,
        llm_model: str,
        evolution_enabled: bool,
        evolution_threshold: int,
        collection_prefix: str = "amem",
        api_key: Optional[str] = None,
    ):
        self.topology = topology
        self.embedding_model = embedding_model
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.evolution_enabled = evolution_enabled
        self.evolution_threshold = evolution_threshold
        self.collection_prefix = collection_prefix
        self.api_key = api_key

        self._shared_system: Optional["AgenticMemorySystem"] = None
        self._local_systems: Dict[str, "AgenticMemorySystem"] = {}

    def _build_system(self, collection_name: str) -> "AgenticMemorySystem":
        from marble.memory.amem_vendor import AgenticMemorySystem

        return AgenticMemorySystem(
            model_name=self.embedding_model,
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            evo_threshold=self.evolution_threshold,
            api_key=self.api_key,
            collection_name=collection_name,
            reset_collection=False,
            evolution_enabled=self.evolution_enabled,
        )

    def get_memory_system(self, agent_id: str) -> "AgenticMemorySystem":
        if self.topology == AMEMTopology.SHARED:
            if self._shared_system is None:
                collection_name = f"{self.collection_prefix}_shared"
                self._shared_system = self._build_system(collection_name)
            return self._shared_system

        if agent_id not in self._local_systems:
            collection_name = f"{self.collection_prefix}_{agent_id}"
            self._local_systems[agent_id] = self._build_system(collection_name)
        return self._local_systems[agent_id]
