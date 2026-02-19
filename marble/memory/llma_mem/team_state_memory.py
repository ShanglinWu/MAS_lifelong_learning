"""Team state memory for LLMA-Mem."""

import threading
from typing import Dict, List, Optional

from .schemas import AgentProfile, TeamPattern


class TeamStateMemory:
    """Stores agent profiles and team performance patterns."""

    EMA_BETA = 0.9  # Exponential moving average factor for reliability

    def __init__(self):
        self._profiles: Dict[str, AgentProfile] = {}
        self._team_patterns: List[TeamPattern] = []
        self._lock = threading.Lock()

    def update_agent_profile(
        self,
        agent_id: str,
        task_success: bool,
        specialization: Optional[str] = None,
        collaborator_id: Optional[str] = None,
        task_type: Optional[str] = None,
        performance: Optional[float] = None,
        communication_cost: Optional[float] = None,
    ) -> AgentProfile:
        """Update an agent's profile with task outcome using EMA for reliability."""
        with self._lock:
            if agent_id not in self._profiles:
                self._profiles[agent_id] = AgentProfile(agent_id=agent_id)

            profile = self._profiles[agent_id]
            profile.total_tasks_completed += 1

            # EMA update for reliability: r_new = beta * r_old + (1-beta) * outcome
            outcome_val = 1.0 if task_success else 0.0
            profile.reliability_score = (
                self.EMA_BETA * profile.reliability_score
                + (1 - self.EMA_BETA) * outcome_val
            )

            if specialization and specialization not in profile.specialization_areas:
                profile.specialization_areas.append(specialization)

            if collaborator_id and collaborator_id not in profile.collaboration_history:
                profile.collaboration_history.append(collaborator_id)
            if collaborator_id:
                stats = profile.collaboration_stats.setdefault(
                    collaborator_id,
                    {
                        "interaction_count": 0.0,
                        "success_rate": 0.0,
                        "avg_communication_cost": 0.0,
                    },
                )
                prev_count = stats["interaction_count"]
                stats["interaction_count"] = prev_count + 1.0
                stats["success_rate"] = (
                    (stats["success_rate"] * prev_count) + outcome_val
                ) / max(stats["interaction_count"], 1.0)
                if communication_cost is not None:
                    stats["avg_communication_cost"] = (
                        (
                            stats["avg_communication_cost"] * prev_count
                            + communication_cost
                        )
                        / stats["interaction_count"]
                    )
            if task_type and performance is not None:
                prev = profile.skill_levels.get(task_type, performance)
                profile.skill_levels[task_type] = (
                    self.EMA_BETA * prev + (1 - self.EMA_BETA) * performance
                )

            return profile

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Retrieve an agent's profile."""
        with self._lock:
            return self._profiles.get(agent_id)

    def get_all_profiles(self) -> Dict[str, AgentProfile]:
        """Return all agent profiles."""
        with self._lock:
            return dict(self._profiles)

    def update_team_pattern(
        self,
        team_composition: List[str],
        task_type: str,
        performance: float,
        comm_overhead: float = 0.0,
    ) -> TeamPattern:
        """Update or create a team performance pattern using EMA."""
        sorted_team = sorted(team_composition)
        with self._lock:
            # Find existing pattern with same composition
            for pattern in self._team_patterns:
                if sorted(pattern.team_composition) == sorted_team:
                    # EMA update for performance
                    pattern.avg_performance = (
                        self.EMA_BETA * pattern.avg_performance
                        + (1 - self.EMA_BETA) * performance
                    )
                    pattern.communication_overhead = (
                        self.EMA_BETA * pattern.communication_overhead
                        + (1 - self.EMA_BETA) * comm_overhead
                    )
                    pattern.usage_count += 1
                    if task_type not in pattern.task_types_suited_for:
                        pattern.task_types_suited_for.append(task_type)
                    return pattern

            # Create new pattern
            new_pattern = TeamPattern(
                team_composition=sorted_team,
                task_types_suited_for=[task_type],
                avg_performance=performance,
                communication_overhead=comm_overhead,
                usage_count=1,
            )
            self._team_patterns.append(new_pattern)
            return new_pattern

    def get_team_patterns(self) -> List[TeamPattern]:
        """Return all team patterns."""
        with self._lock:
            return list(self._team_patterns)
