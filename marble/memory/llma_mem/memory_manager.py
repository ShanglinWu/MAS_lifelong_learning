"""Main LLMA-Mem manager interface for agents."""

import os
import re
import time
from typing import List, Optional

from marble.llms.text_embedding import DEFAULT_EMBEDDING_MODEL, text_embedding

from .consolidation import consolidate
from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory
from .retrieval import retrieve_top_k
from .schemas import EpisodicRecord, ProceduralRecord
from .team_state_memory import TeamStateMemory
from .topology import MemoryTopologyManager


class LLMAMemManager:
    """
    Main interface for agents to interact with LLMA-Mem.

    Wraps episodic, procedural, and team state memory with
    retrieval, recording, and consolidation capabilities.
    """

    def __init__(
        self,
        agent_id: str,
        topology_manager: MemoryTopologyManager,
        consolidation_interval: int = 2,
        enable_auto_embedding: bool = True,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.agent_id = agent_id
        self.topology_manager = topology_manager
        self.consolidation_interval = consolidation_interval
        self.enable_auto_embedding = enable_auto_embedding
        self.embedding_model = embedding_model

        # Resolve memory instances from topology
        self.episodic: EpisodicMemory = topology_manager.get_episodic(agent_id)
        self.procedural: ProceduralMemory = topology_manager.get_procedural(agent_id)
        self.team_state: TeamStateMemory = topology_manager.get_team_state(agent_id)
        self._last_retrieved_procedure_ids: List[str] = []

    def _can_embed(self) -> bool:
        """Best-effort check before attempting Bedrock embedding calls."""
        if not self.enable_auto_embedding:
            return False
        # Support both standard AWS credentials and bearer-token based Bedrock auth.
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            return True
        return bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )

    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text, returning [] on failure."""
        if not text.strip() or not self._can_embed():
            return []
        try:
            return text_embedding(model=self.embedding_model, input=text)
        except Exception:
            return []

    def record_episode(
        self,
        task_description: str,
        actions_taken: List[str],
        outcome: str,
        team_composition: Optional[List[str]] = None,
        context_state: Optional[dict] = None,
        lessons_learned: str = "",
        collaboration_quality: float = 0.5,
        embedding: Optional[List[float]] = None,
        task_success: Optional[bool] = None,
        used_procedure_ids: Optional[List[str]] = None,
    ) -> EpisodicRecord:
        """
        Record an episode and trigger consolidation if needed.
        Also updates team state memory.
        """
        if not embedding:
            embedding_text = (
                f"Task: {task_description}\n"
                f"Actions: {'; '.join(actions_taken)}\n"
                f"Outcome: {outcome}\n"
                f"Lessons: {lessons_learned}"
            )
            embedding = self._embed_text(embedding_text)

        inferred_success = (
            task_success
            if task_success is not None
            else self._infer_task_success(outcome, context_state or {})
        )

        record = EpisodicRecord(
            agent_id=self.agent_id,
            timestamp=time.time(),
            task_description=task_description,
            team_composition=team_composition or [],
            actions_taken=actions_taken,
            outcome=outcome,
            context_state=context_state or {},
            lessons_learned=lessons_learned,
            collaboration_quality=collaboration_quality,
            embedding=embedding,
            outcome_success=inferred_success,
        )

        self.episodic.add_episode(record)

        # Update team state
        self.team_state.update_agent_profile(
            agent_id=self.agent_id,
            task_success=inferred_success,
            task_type=task_description[:50],
            performance=1.0 if inferred_success else 0.0,
        )
        if team_composition:
            self.team_state.update_team_pattern(
                team_composition=team_composition,
                task_type=task_description[:50],
                performance=1.0 if inferred_success else 0.0,
            )

        # Update procedural success/failure counts for strategies the agent just used.
        procedure_ids = (
            list(used_procedure_ids)
            if used_procedure_ids is not None
            else list(self._last_retrieved_procedure_ids)
        )
        for pid in procedure_ids:
            self.procedural.update_procedure_stats(pid, inferred_success)
        self._last_retrieved_procedure_ids = []

        # Trigger consolidation
        consolidate(
            episodic=self.episodic,
            procedural=self.procedural,
            agent_id=self.agent_id,
            trigger_interval=self.consolidation_interval,
        )

        return record

    def retrieve_relevant_memories(
        self,
        query_embedding: List[float],
        query_team: Optional[List[str]] = None,
        k: int = 5,
    ) -> List:
        """
        Multi-factor retrieval across episodic and procedural memories.
        Returns list of (record, score) tuples.
        """
        self._last_retrieved_procedure_ids = []
        all_records = []
        all_records.extend(self.episodic.get_episodes())
        all_records.extend(self.procedural.get_all_procedures())

        # Increment viewed_times for procedural records that are retrieved
        results = retrieve_top_k(
            records=all_records,
            query_embedding=query_embedding,
            query_team=query_team or [],
            k=k,
        )

        for record, _ in results:
            if isinstance(record, ProceduralRecord):
                self.procedural.increment_viewed(
                    record.procedure_id,
                    current_episode_count=self.episodic.count(),
                )
                self._last_retrieved_procedure_ids.append(record.procedure_id)

        return results

    def get_memory_context_str(
        self,
        query_embedding: Optional[List[float]] = None,
        query_text: str = "",
        query_team: Optional[List[str]] = None,
        k: int = 5,
    ) -> str:
        """
        Get a formatted string of relevant memories for prompt injection.
        Replaces BaseMemory.get_memory_str().
        """
        parts = []
        self._last_retrieved_procedure_ids = []

        if query_embedding is None and query_text:
            query_embedding = self._embed_text(query_text)

        if query_embedding:
            results = self.retrieve_relevant_memories(
                query_embedding=query_embedding,
                query_team=query_team,
                k=k,
            )

            if results:
                parts.append("=== Strategic Memory (LLMA-Mem) ===")
                for record, score in results:
                    if isinstance(record, EpisodicRecord):
                        parts.append(
                            f"[Episode] {record.task_description} -> "
                            f"{record.outcome} "
                            f"(lessons: {record.lessons_learned}) "
                            f"[score: {score:.2f}]"
                        )
                    elif isinstance(record, ProceduralRecord):
                        parts.append(
                            f"[Procedure] {record.title}: "
                            f"{record.knowledge_content} "
                            f"(confidence: {record.confidence_score:.2f}) "
                            f"[score: {score:.2f}]"
                        )
        else:
            # Without embeddings, return recent episodes
            recent = self.episodic.get_recent(k)
            if recent:
                parts.append("=== Recent Memory (LLMA-Mem) ===")
                for ep in recent:
                    parts.append(
                        f"[Episode] {ep.task_description} -> {ep.outcome}"
                    )

            procedures = self.procedural.get_all_procedures()
            if procedures:
                # Sort by confidence and take top entries
                top_procs = sorted(
                    procedures,
                    key=lambda p: p.confidence_score,
                    reverse=True,
                )[:k]
                parts.append("=== Known Strategies ===")
                for proc in top_procs:
                    parts.append(
                        f"[Strategy] {proc.title}: {proc.knowledge_content} "
                        f"(confidence: {proc.confidence_score:.2f})"
                    )

        # Add team state info
        profile = self.team_state.get_agent_profile(self.agent_id)
        if profile:
            parts.append(
                f"=== Agent Profile ===\n"
                f"Reliability: {profile.reliability_score:.2f}, "
                f"Tasks completed: {profile.total_tasks_completed}"
            )

        return "\n".join(parts) if parts else ""

    def _infer_task_success(self, outcome: str, context_state: dict) -> bool:
        """Best-effort success inference for non-structured outcomes."""
        explicit = context_state.get("task_success")
        if isinstance(explicit, bool):
            return explicit
        explicit = context_state.get("success")
        if isinstance(explicit, bool):
            return explicit

        text = (outcome or "").strip().lower()
        if not text:
            return False

        fail_re = re.compile(
            r"\b(fail|failed|failure|error|exception|invalid|timeout|crash|traceback)\b"
        )
        success_re = re.compile(
            r"\b(success|succeeded|done|completed|resolved|fixed|passed)\b"
        )
        if fail_re.search(text):
            return False
        if success_re.search(text):
            return True
        return len(text) > 20
