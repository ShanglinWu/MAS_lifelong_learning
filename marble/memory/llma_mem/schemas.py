"""Pydantic data models for the LLMA-Mem framework."""

import uuid
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryTopology(str, Enum):
    """Memory topology configurations."""

    LOCAL = "local"
    SHARED = "shared"
    HYBRID = "hybrid"


class EpisodicRecord(BaseModel):
    """A single episodic memory record capturing one experience."""

    agent_id: str
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = 0.0
    task_description: str = ""
    team_composition: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    outcome: str = ""
    outcome_success: Optional[bool] = None
    outcome_metrics: Dict[str, float] = Field(default_factory=dict)
    outcome_error_info: Optional[str] = None
    context_state: Dict = Field(default_factory=dict)
    lessons_learned: str = ""
    related_procedures: List[str] = Field(default_factory=list)
    collaboration_quality: float = 0.0
    embedding: List[float] = Field(default_factory=list)


class ProceduralRecord(BaseModel):
    """A procedural memory record capturing learned strategies/patterns."""

    agent_id: str
    procedure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamps: List[float] = Field(default_factory=list)
    timestamp_created: float = 0.0
    timestamp_updated: float = 0.0
    title: str = ""
    type: str = ""  # e.g. "strategy", "tactic", "communication_pattern"
    knowledge_content: str = ""
    prerequisites: List[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    avg_performance_gain: float = 0.0
    confidence_score: float = 0.5  # computed: (s+alpha)/(s+f+2*alpha)
    source_episodes: List[str] = Field(default_factory=list)
    last_used: float = 0.0
    last_used_episode: int = -1
    viewed_times: int = 0
    embedding: List[float] = Field(default_factory=list)

    def compute_confidence(self, alpha: float = 1.0) -> float:
        """Compute confidence score: sigma = (s + alpha) / (s + f + 2*alpha)."""
        total = self.success_count + self.failure_count
        self.success_rate = self.success_count / total if total > 0 else 0.0
        self.confidence_score = (self.success_count + alpha) / (
            total + 2 * alpha
        )
        return self.confidence_score


class AgentProfile(BaseModel):
    """Profile tracking an agent's capabilities and history."""

    agent_id: str
    specialization_areas: List[str] = Field(default_factory=list)
    skill_levels: Dict[str, float] = Field(default_factory=dict)
    collaboration_history: List[str] = Field(default_factory=list)
    collaboration_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    reliability_score: float = 0.5
    total_tasks_completed: int = 0


class TeamPattern(BaseModel):
    """A pattern describing team performance on certain task types."""

    team_composition: List[str] = Field(default_factory=list)
    task_types_suited_for: List[str] = Field(default_factory=list)
    avg_performance: float = 0.0
    communication_overhead: float = 0.0
    usage_count: int = 0
