"""
Transactive Memory module for LLMA-Mem framework.

Captures the evolving capabilities and collaboration dynamics of the
multi-agent system. Tracks individual agent competencies and collective
patterns that emerge from repeated collaboration.

Components:
    - agent_profiles: specialization_areas, collaboration_history, total_tasks_completed
    - team_patterns: team_composition, task_types_suited_for, avg_performance,
                     communication_overhead, usage_count

Should be used by providing to planning agent for better task allocation.
"""

import json
import os
from typing import Any, Dict, List, Optional


class TransactiveMemory:
    """
    Transactive memory captures agent capabilities and collaboration dynamics.

    Agent profile α_k = <a_id, S, Γ, H>
    Team pattern φ_m = <C_m, T_suited, ρ̄_m, ω_m>

    Used by the planning agent for task allocation and team formation.
    """

    def __init__(self, agent_id: str = "shared", persist_dir: str = "memory_store") -> None:
        """
        Initialize transactive memory.

        At t=0, transactive memory is empty: T^(0) = ∅

        Args:
            agent_id: Identifier ("shared" for shared topology, or agent-specific).
            persist_dir: Directory for file persistence.
        """
        self.agent_id = agent_id
        self.persist_dir = persist_dir
        self.agent_profiles: Dict[str, Dict[str, Any]] = {}
        self.team_patterns: Dict[str, Dict[str, Any]] = {}
        self._pattern_counter: int = 0
        self._load()

    def _get_file_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.agent_id}_transactive.json")

    def _load(self) -> None:
        """Load transactive memory from persistent storage."""
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                self.agent_profiles = data.get("agent_profiles", {})
                self.team_patterns = data.get("team_patterns", {})
                self._pattern_counter = data.get("pattern_counter", 0)

    def _save(self) -> None:
        """Save transactive memory to persistent storage."""
        os.makedirs(self.persist_dir, exist_ok=True)
        file_path = self._get_file_path()
        data = {
            "agent_profiles": self.agent_profiles,
            "team_patterns": self.team_patterns,
            "pattern_counter": self._pattern_counter,
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _ensure_agent_profile(self, agent_id: str) -> None:
        """Ensure an agent profile exists, creating a default if needed."""
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = {
                "specialization_areas": [],
                "collaboration_history": {},
                "total_tasks_completed": 0,
                "success_count": 0,
            }

    def update_agent_profile(
        self,
        agent_id: str,
        task_success: bool,
        task_type: str = "",
        partner_ids: Optional[List[str]] = None,
        specialization_area: str = "",
    ) -> None:
        """
        Update agent profile after task completion.

        ξ_k = successes_k / total_tasks_k

        Also updates agent's learned specialization areas and
        collaboration history with partners.

        Args:
            agent_id: The agent whose profile to update.
            task_success: Whether the task was successful.
            task_type: Type/category of the task.
            partner_ids: IDs of collaborating agents.
            specialization_area: Area of specialization demonstrated.
        """
        self._ensure_agent_profile(agent_id)
        profile = self.agent_profiles[agent_id]

        # Update total tasks and success count
        profile["total_tasks_completed"] += 1
        if task_success:
            profile["success_count"] += 1

        # Update specialization areas (learned from successful tasks)
        if (
            specialization_area
            and specialization_area not in profile["specialization_areas"]
        ):
            profile["specialization_areas"].append(specialization_area)

        # Update collaboration history with each partner
        if partner_ids:
            for partner_id in partner_ids:
                if partner_id == agent_id:
                    continue
                if partner_id not in profile["collaboration_history"]:
                    profile["collaboration_history"][partner_id] = {
                        "interaction_count": 0,
                        "success_count": 0,
                        "success_rate": 0.0,
                    }
                collab = profile["collaboration_history"][partner_id]
                collab["interaction_count"] += 1
                if task_success:
                    collab["success_count"] += 1
                collab["success_rate"] = (
                    collab["success_count"] / collab["interaction_count"]
                )

        self._save()

    def update_team_pattern(
        self,
        team_composition: List[str],
        task_type: str,
        performance: float,
        communication_overhead: float = 0.0,
        success: bool = True,
    ) -> None:
        """
        Update team patterns after task completion.

        Updates avg_performance (ρ̄_m) and communication_overhead (ω_m)
        for the given team configuration.

        Args:
            team_composition: List of agent IDs in the team.
            task_type: Type of task performed.
            performance: Performance metric for this task.
            communication_overhead: Communication overhead metric.
            success: Whether the task was successful.
        """
        # Create config_id from sorted team composition
        config_id = "_".join(sorted(team_composition))

        if config_id not in self.team_patterns:
            self.team_patterns[config_id] = {
                "team_composition": sorted(team_composition),
                "task_types_suited_for": [],
                "avg_performance": 0.0,
                "communication_overhead": 0.0,
                "usage_count": 0,
                "total_performance": 0.0,
                "total_overhead": 0.0,
            }

        pattern = self.team_patterns[config_id]
        pattern["usage_count"] += 1

        # Update task types suited for (only add on success)
        if task_type and task_type not in pattern["task_types_suited_for"] and success:
            pattern["task_types_suited_for"].append(task_type)

        # Update avg_performance: ρ̄_m
        pattern["total_performance"] += performance
        pattern["avg_performance"] = (
            pattern["total_performance"] / pattern["usage_count"]
        )

        # Update communication_overhead: ω_m
        pattern["total_overhead"] += communication_overhead
        pattern["communication_overhead"] = (
            pattern["total_overhead"] / pattern["usage_count"]
        )

        self._save()

    def get_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent's profile."""
        return self.agent_profiles.get(agent_id)

    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all agent profiles."""
        return self.agent_profiles.copy()

    def get_team_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get all team patterns."""
        return self.team_patterns.copy()

    def is_empty(self) -> bool:
        """Check if transactive memory is empty."""
        return not self.agent_profiles and not self.team_patterns

    def to_str(self) -> str:
        """
        Convert transactive memory to a string representation
        suitable for inclusion in planning prompts.

        Provides the planning agent with agent capabilities and
        team collaboration patterns for better task allocation.

        Returns:
            Formatted string of transactive memory contents.
        """
        if not self.agent_profiles and not self.team_patterns:
            return ""

        parts: List[str] = []

        if self.agent_profiles:
            parts.append("=== Agent Capabilities ===")
            for agent_id, profile in self.agent_profiles.items():
                specs = (
                    ", ".join(profile["specialization_areas"])
                    if profile["specialization_areas"]
                    else "None yet"
                )
                total = profile["total_tasks_completed"]
                success_ct = profile.get("success_count", 0)
                reliability = (
                    f"{success_ct / total:.2f}" if total > 0 else "N/A"
                )
                parts.append(f"Agent {agent_id}:")
                parts.append(f"  Specializations: {specs}")
                parts.append(f"  Tasks Completed: {total}")
                parts.append(f"  Reliability (ξ): {reliability}")

                if profile["collaboration_history"]:
                    collab_strs = []
                    for partner, stats in profile["collaboration_history"].items():
                        collab_strs.append(
                            f"{partner}(interactions={stats['interaction_count']}, "
                            f"success_rate={stats['success_rate']:.2f})"
                        )
                    parts.append(f"  Collaboration: {', '.join(collab_strs)}")

        if self.team_patterns:
            parts.append("\n=== Team Patterns ===")
            for config_id, pattern in self.team_patterns.items():
                parts.append(f"Team {pattern['team_composition']}:")
                suited = (
                    ", ".join(pattern["task_types_suited_for"])
                    if pattern["task_types_suited_for"]
                    else "General"
                )
                parts.append(f"  Suited for: {suited}")
                parts.append(
                    f"  Avg Performance: {pattern['avg_performance']:.2f}"
                )
                parts.append(
                    f"  Communication Overhead: {pattern['communication_overhead']:.2f}"
                )
                parts.append(f"  Usage Count: {pattern['usage_count']}")

        return "\n".join(parts)
