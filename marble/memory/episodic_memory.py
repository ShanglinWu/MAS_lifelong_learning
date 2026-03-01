"""
Episodic Memory module for LLMA-Mem framework.

Stores comprehensive records of individual task experiences, preserving the
full context necessary for pattern extraction and reflection. Each episode
captures actions, outcomes, environmental state, and team dynamics.

Schema per spec:
    agent_id, episode_id, timestamp, task_description,
    team_composition, actions_taken,
    outcome (success, metrics, error_info),
    lessons_learned, related_procedures, collaboration_quality
"""

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from marble.llms.text_embedding import text_embedding


class EpisodicMemory:
    """
    Episodic memory stores comprehensive records of individual task experiences.

    Each episode e_i = <a_id, t, τ, C, A, o, c, L, P_rel>

    Retrieval uses multi-factor scoring:
        score(m, q) = λ_r * recency(m) + λ_v * relevance(m, q) + λ_i * importance(m)
    where for episodic memory, m is task_description.
    """

    EMBEDDING_MODEL = "bedrock/amazon.titan-embed-text-v2:0"

    def __init__(self, agent_id: str, persist_dir: str = "memory_store") -> None:
        """
        Initialize episodic memory.

        At t=0, episodic memory is empty: E^(0) = ∅

        Args:
            agent_id: Agent identifier (or "shared" for shared topology).
            persist_dir: Directory for file persistence.
        """
        self.agent_id = agent_id
        self.persist_dir = persist_dir
        self.episodes: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []  # Stored as lists for JSON serialization
        self._episode_counter: int = 0
        self._load()

    def _get_file_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.agent_id}_episodic.json")

    def _load(self) -> None:
        """Load episodic memory from persistent storage."""
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                self.episodes = data.get("episodes", [])
                self.embeddings = data.get("embeddings", [])
                self._episode_counter = data.get("episode_counter", len(self.episodes))

    def _save(self) -> None:
        """Save episodic memory to persistent storage."""
        os.makedirs(self.persist_dir, exist_ok=True)
        file_path = self._get_file_path()
        data = {
            "episodes": self.episodes,
            "embeddings": self.embeddings,
            "episode_counter": self._episode_counter,
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_episode(
        self,
        agent_id: str,
        task_description: str,
        team_composition: List[str],
        actions_taken: List[Dict[str, Any]],
        outcome: Dict[str, Any],
        context: str = "",
        lessons_learned: Optional[List[str]] = None,
        related_procedures: Optional[List[str]] = None,
        collaboration_quality: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Add a new episode to episodic memory.

        Episodic Addition: E^(t+1) = E^(t) ∪ {e_new}

        Args:
            agent_id: The primary agent identifier for this episode.
            task_description: Description of the task.
            team_composition: List of agent IDs involved.
            actions_taken: Sequence of actions taken.
            outcome: {"success": bool, "metrics": dict, "error_info": str|None}
            context: Environmental context and state.
            lessons_learned: Extracted insights.
            related_procedures: Linked procedure IDs.
            collaboration_quality: Team performance score.

        Returns:
            The created episode dict.
        """
        episode: Dict[str, Any] = {
            "agent_id": agent_id,
            "episode_id": self._episode_counter,
            "timestamp": self._episode_counter,  # Sequential for recency formula
            "task_description": task_description,
            "team_composition": team_composition,
            "actions_taken": actions_taken,
            "outcome": {
                "success": outcome.get("success", False),
                "metrics": outcome.get("metrics", {}),
                "error_info": outcome.get("error_info", None),
            },
            "context": context,
            "lessons_learned": lessons_learned if lessons_learned is not None else [],
            "related_procedures": related_procedures if related_procedures is not None else [],
            "collaboration_quality": collaboration_quality,
        }
        self.episodes.append(episode)

        # Compute embedding of task_description (for episodic, m is task_description)
        emb = text_embedding(model=self.EMBEDDING_MODEL, input=task_description)
        self.embeddings.append(emb)

        self._episode_counter += 1
        self._save()
        return episode

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        current_time: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve the most relevant episodes using multi-factor scoring.

        score(m, q) = λ_r * recency(m) + λ_v * relevance(m, q) + λ_i * importance(m)

        Where:
            λ_r = 0.2, λ_v = 0.4, λ_i = 0.4
            recency(m) = exp(t_m - t_current)
            relevance(m, q) = cosine_similarity(embed(m), embed(q))
            importance(e) based on outcome success

        Args:
            query: The query string (current task description).
            top_k: Number of top memories to retrieve.
            current_time: Current sequential timestamp (defaults to episode counter).

        Returns:
            Top-k episodes sorted by score, or None if empty.
        """
        if not self.episodes:
            return None

        if current_time is None:
            current_time = self._episode_counter

        # Compute query embedding
        query_emb = text_embedding(model=self.EMBEDDING_MODEL, input=query)
        query_emb_array = np.array(query_emb).reshape(1, -1)

        scored_episodes: List[tuple] = []
        for i, episode in enumerate(self.episodes):
            # recency(m) = exp(t_m - t_current)
            recency = math.exp(episode["timestamp"] - current_time)

            # relevance(m, q) = cosine_similarity(embed(task_description), embed(query))
            ep_emb = np.array(self.embeddings[i]).reshape(1, -1)
            relevance = float(sklearn_cosine_similarity(ep_emb, query_emb_array)[0][0])

            # importance for episodic: based on outcome success
            importance = 1.0 if episode["outcome"].get("success", False) else 0.3

            # Multi-factor scoring with paper weights
            score = 0.2 * recency + 0.4 * relevance + 0.4 * importance
            scored_episodes.append((score, episode))

        # Sort by score descending and take top-k
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in scored_episodes[:top_k]]
        return results if results else None

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all episodes."""
        return self.episodes.copy()

    def get_recent(self, n: int = 3) -> List[Dict[str, Any]]:
        """Return the n most recent episodes."""
        return self.episodes[-n:] if self.episodes else []

    def count(self) -> int:
        """Return the number of stored episodes."""
        return len(self.episodes)

    def is_empty(self) -> bool:
        """Check if episodic memory is empty."""
        return len(self.episodes) == 0

    def get_lessons_with_embeddings(self) -> List[tuple]:
        """
        Get embeddings of lessons_learned for each episode.
        Used during consolidation for clustering.

        Returns:
            List of (episode, embedding_ndarray) tuples for episodes with lessons.
        """
        result = []
        for ep in self.episodes:
            lessons_text = " ".join(ep.get("lessons_learned", []))
            if lessons_text.strip():
                emb = text_embedding(model=self.EMBEDDING_MODEL, input=lessons_text)
                result.append((ep, np.array(emb)))
        return result
