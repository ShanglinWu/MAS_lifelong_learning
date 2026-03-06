"""
Procedural Memory module for LLMA-Mem framework.

Abstracts reusable strategies, skills, and domain-specific procedures from
episodic experiences. Unlike raw episodes, procedural knowledge represents
generalized patterns that have demonstrated reliability across multiple instances.

Schema per spec:
    agent_id, procedure_id, timestamp_created, timestamp_updated,
    title, knowledge_content,
    success_count, failure_count, success_rate,
    avg_performance_gain, source_episodes, viewed_times
"""

import json
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from marble.llms.text_embedding import text_embedding


class ProceduralMemory:
    """
    Procedural memory stores reusable strategies abstracted from episodic experiences.

    Each procedure p_j = <a_id, t_c, t_u, title, κ, π, s, f, E_src>

    Success rate: ρ_j = s_j / (s_j + f_j + ε)

    Retrieval uses:
        score(m, q) = relevance(m, q) + importance(m)
    where for procedural memory, m is title + knowledge_content.
        importance(p_j) = ρ_j  (success rate)
    """

    EMBEDDING_MODEL = "bedrock/amazon.titan-embed-text-v2:0"
    EPSILON: float = 1e-6  # Small constant to prevent division by zero

    def __init__(self, agent_id: str, persist_dir: str = "memory_store") -> None:
        """
        Initialize procedural memory.

        At t=0, procedural memory is empty: P^(0) = ∅

        Args:
            agent_id: Agent identifier (or "shared" for shared topology).
            persist_dir: Directory for file persistence.
        """
        self.agent_id = agent_id
        self.persist_dir = persist_dir
        self.procedures: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self._procedure_counter: int = 0
        self._load()

    def _get_file_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.agent_id}_procedural.json")

    def _load(self) -> None:
        """Load procedural memory from persistent storage."""
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                self.procedures = data.get("procedures", [])
                self.embeddings = data.get("embeddings", [])
                self._procedure_counter = data.get(
                    "procedure_counter", len(self.procedures)
                )

    def _save(self) -> None:
        """Save procedural memory to persistent storage."""
        os.makedirs(self.persist_dir, exist_ok=True)
        file_path = self._get_file_path()
        data = {
            "procedures": self.procedures,
            "embeddings": self.embeddings,
            "procedure_counter": self._procedure_counter,
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_procedure(
        self,
        title: str,
        knowledge_content: str,
        source_episodes: List[int],
        success_count: int = 0,
        failure_count: int = 0,
    ) -> Dict[str, Any]:
        """
        Add a new procedure to procedural memory.

        Args:
            title: Short descriptive title for the procedure.
            knowledge_content: The generalized strategy/skill content.
            source_episodes: List of episode IDs this was extracted from.
            success_count: Initial success count.
            failure_count: Initial failure count.

        Returns:
            The created procedure dict.
        """
        success_rate = success_count / (success_count + failure_count + self.EPSILON)

        procedure: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "procedure_id": self._procedure_counter,
            "timestamp_created": self._procedure_counter,
            "timestamp_updated": self._procedure_counter,
            "title": title,
            "knowledge_content": knowledge_content,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "avg_performance_gain": 0.0,
            "source_episodes": source_episodes,
            "viewed_times": 0,
        }
        self.procedures.append(procedure)

        # Compute embedding (title + knowledge_content as per spec)
        embed_text = f"{title} {knowledge_content}"
        emb = text_embedding(model=self.EMBEDDING_MODEL, input=embed_text)
        self.embeddings.append(emb)

        self._procedure_counter += 1
        self._save()
        return procedure

    def update_procedure_outcome(self, procedure_id: int, success: bool) -> None:
        """
        Update success/failure counts for a procedure (Bayesian update).

        s_j^(t+1) = s_j^(t) + 1_success
        f_j^(t+1) = f_j^(t) + 1_failure
        ρ_j = s_j / (s_j + f_j + ε)

        Args:
            procedure_id: The procedure to update.
            success: Whether the task using this procedure succeeded.
        """
        for proc in self.procedures:
            if proc["procedure_id"] == procedure_id:
                if success:
                    proc["success_count"] += 1
                else:
                    proc["failure_count"] += 1
                # Update success rate: ρ_j = s_j / (s_j + f_j + ε)
                proc["success_rate"] = proc["success_count"] / (
                    proc["success_count"] + proc["failure_count"] + self.EPSILON
                )
                proc["timestamp_updated"] = self._procedure_counter
                self._save()
                return

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        current_time: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve the most relevant procedures using:
            score(m, q) = relevance(m, q) + importance(m)
        where:
            relevance(m, q) = cosine_similarity(embed(title+content), embed(query))
            importance(p_j) = ρ_j (success rate)

        Args:
            query: The query string (current task description).
            top_k: Number of top procedures to retrieve.
            current_time: Current sequential timestamp.

        Returns:
            Top-k procedures sorted by score, or None if empty.
        """
        if not self.procedures:
            return None

        # Compute query embedding
        query_emb = text_embedding(model=self.EMBEDDING_MODEL, input=query)
        query_emb_array = np.array(query_emb).reshape(1, -1)

        scored_procedures: List[tuple] = []
        for i, proc in enumerate(self.procedures):
            # relevance(m, q) = cosine_similarity(embed(title+content), embed(query))
            proc_emb = np.array(self.embeddings[i]).reshape(1, -1)
            relevance = float(
                sklearn_cosine_similarity(proc_emb, query_emb_array)[0][0]
            )

            # importance(p_j) = ρ_j (purely success rate)
            importance = proc["success_rate"]

            # Requested scoring: no recency, no weighting
            score = relevance + importance
            scored_procedures.append((score, proc))

        scored_procedures.sort(key=lambda x: x[0], reverse=True)

        # Increment viewed_times for retrieved procedures
        results: List[Dict[str, Any]] = []
        for _, proc in scored_procedures[:top_k]:
            proc["viewed_times"] += 1
            results.append(proc)
        self._save()

        return results if results else None

    def is_empty(self) -> bool:
        """Check if procedural memory has no procedures."""
        return len(self.procedures) == 0

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all procedures."""
        return self.procedures.copy()

    def count(self) -> int:
        """Return the number of stored procedures."""
        return len(self.procedures)

    def get_existing_source_sets(self) -> List[Set[int]]:
        """Get the set of source episode IDs for each existing procedure."""
        return [set(p["source_episodes"]) for p in self.procedures]

    def prune_overlap(self) -> None:
        """
        Prune overlap procedures: if two procedures exist where one's source
        episodes are included in another's source episodes, delete the smaller one.
        """
        to_remove: set = set()
        for i in range(len(self.procedures)):
            if i in to_remove:
                continue
            src_i = set(self.procedures[i]["source_episodes"])
            for j in range(len(self.procedures)):
                if i == j or j in to_remove:
                    continue
                src_j = set(self.procedures[j]["source_episodes"])
                if src_i < src_j:  # src_i is proper subset of src_j → remove i
                    to_remove.add(i)
                    break
                elif src_j < src_i:  # src_j is proper subset of src_i → remove j
                    to_remove.add(j)

        if to_remove:
            self.procedures = [
                p for idx, p in enumerate(self.procedures) if idx not in to_remove
            ]
            self.embeddings = [
                e for idx, e in enumerate(self.embeddings) if idx not in to_remove
            ]
            self._save()
