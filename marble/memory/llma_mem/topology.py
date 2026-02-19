"""Topology manager for LLMA-Mem (local / shared / hybrid)."""

from typing import Dict

from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory
from .schemas import MemoryTopology
from .team_state_memory import TeamStateMemory


class HybridTeamStateMemory:
    """Proxy that writes to both local and shared team-state stores."""

    def __init__(
        self, local_team_state: TeamStateMemory, shared_team_state: TeamStateMemory
    ):
        self._local = local_team_state
        self._shared = shared_team_state

    def update_agent_profile(self, *args, **kwargs):
        local_profile = self._local.update_agent_profile(*args, **kwargs)
        self._shared.update_agent_profile(*args, **kwargs)
        return local_profile

    def update_team_pattern(self, *args, **kwargs):
        local_pattern = self._local.update_team_pattern(*args, **kwargs)
        self._shared.update_team_pattern(*args, **kwargs)
        return local_pattern

    def get_agent_profile(self, agent_id: str):
        return self._local.get_agent_profile(agent_id)

    def get_team_patterns(self):
        local_patterns = self._local.get_team_patterns()
        shared_patterns = self._shared.get_team_patterns()
        return local_patterns + shared_patterns


class MemoryTopologyManager:
    """
    Manages memory instances according to the chosen topology:

    - LOCAL:  each agent gets private episodic, procedural, and team state
    - SHARED: all agents share single instances of each memory type
    - HYBRID: local episodic + shared procedural, local + shared team state
    """

    def __init__(self, topology: MemoryTopology):
        self.topology = topology

        # Shared instances (used by SHARED and HYBRID)
        self._shared_episodic = EpisodicMemory()
        self._shared_procedural = ProceduralMemory()
        self._shared_team_state = TeamStateMemory()

        # Per-agent local instances (used by LOCAL and HYBRID)
        self._local_episodic: Dict[str, EpisodicMemory] = {}
        self._local_procedural: Dict[str, ProceduralMemory] = {}
        self._local_team_state: Dict[str, TeamStateMemory] = {}

    def _ensure_local(self, agent_id: str) -> None:
        """Lazily create local memory instances for an agent."""
        if agent_id not in self._local_episodic:
            self._local_episodic[agent_id] = EpisodicMemory()
        if agent_id not in self._local_procedural:
            self._local_procedural[agent_id] = ProceduralMemory()
        if agent_id not in self._local_team_state:
            self._local_team_state[agent_id] = TeamStateMemory()

    def get_episodic(self, agent_id: str) -> EpisodicMemory:
        """Get episodic memory for an agent based on topology."""
        if self.topology == MemoryTopology.SHARED:
            return self._shared_episodic
        else:
            # LOCAL and HYBRID both use local episodic
            self._ensure_local(agent_id)
            return self._local_episodic[agent_id]

    def get_procedural(self, agent_id: str) -> ProceduralMemory:
        """Get procedural memory for an agent based on topology."""
        if self.topology == MemoryTopology.LOCAL:
            self._ensure_local(agent_id)
            return self._local_procedural[agent_id]
        else:
            # SHARED and HYBRID both use shared procedural
            return self._shared_procedural

    def get_team_state(self, agent_id: str) -> TeamStateMemory:
        """Get team state memory for an agent based on topology."""
        if self.topology == MemoryTopology.LOCAL:
            self._ensure_local(agent_id)
            return self._local_team_state[agent_id]
        if self.topology == MemoryTopology.HYBRID:
            self._ensure_local(agent_id)
            return HybridTeamStateMemory(
                local_team_state=self._local_team_state[agent_id],
                shared_team_state=self._shared_team_state,
            )
        else:
            # SHARED uses shared team state
            return self._shared_team_state
