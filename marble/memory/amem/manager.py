"""MARBLE-facing A-MEM manager with LLMA-Mem-compatible agent interface."""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from marble.memory.amem.topology import AMEMTopologyManager


class AMEMManager:
    """Adapter exposing the interface consumed by BaseAgent."""

    def __init__(
        self,
        agent_id: str,
        topology_manager: AMEMTopologyManager,
        retrieval_k: int = 5,
    ):
        self.agent_id = agent_id
        self.topology_manager = topology_manager
        self.retrieval_k = max(1, int(retrieval_k))
        self.system = topology_manager.get_memory_system(agent_id)

    def _infer_task_success(self, outcome: str, context_state: dict) -> bool:
        explicit = context_state.get("task_success")
        if isinstance(explicit, bool):
            return explicit
        explicit = context_state.get("success")
        if isinstance(explicit, bool):
            return explicit

        text = (outcome or "").strip().lower()
        if not text:
            return False

        fail_re = re.compile(r"\b(fail|failed|failure|error|exception|invalid|timeout|crash|traceback)\b")
        success_re = re.compile(r"\b(success|succeeded|done|completed|resolved|fixed|passed)\b")
        if fail_re.search(text):
            return False
        if success_re.search(text):
            return True
        return False

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
    ) -> str:
        """Store one episode in A-MEM.

        Extra parameters are accepted for LLMA-Mem compatibility and ignored where not used.
        """
        _ = (collaboration_quality, embedding, used_procedure_ids)
        context = context_state or {}
        inferred_success = (
            task_success if task_success is not None else self._infer_task_success(outcome, context)
        )

        payload = {
            "agent_id": self.agent_id,
            "task_description": task_description,
            "actions_taken": actions_taken,
            "outcome": outcome,
            "team_composition": team_composition or [],
            "context_state": context,
            "lessons_learned": lessons_learned,
            "task_success": inferred_success,
        }

        tags = ["episode", "success" if inferred_success else "failure"]
        return self.system.add_note(
            content=json.dumps(payload, ensure_ascii=True),
            tags=tags,
            context=str(task_description)[:120] if task_description else "Task Episode",
            category="AgentEpisode",
        )

    def get_memory_context_str(
        self,
        query_embedding: Optional[List[float]] = None,
        query_text: str = "",
        query_team: Optional[List[str]] = None,
        k: int = 5,
    ) -> str:
        """Return formatted retrieved memories for prompt injection."""
        _ = (query_embedding, query_team)
        top_k = max(1, int(k) if isinstance(k, int) and k > 0 else self.retrieval_k)
        query = query_text.strip() if query_text else "recent successful strategies"

        rows = self.system.search_agentic(query, k=top_k)
        if not rows:
            return ""

        parts = ["=== Strategic Memory (A-MEM) ==="]
        for row in rows:
            row_type = "Neighbor" if row.get("is_neighbor") else "Note"
            content = str(row.get("content", "")).replace("\n", " ")
            if len(content) > 300:
                content = content[:297] + "..."
            line = f"[{row_type}] {content}"

            tags = row.get("tags", [])
            if isinstance(tags, list) and tags:
                line += f" (tags: {', '.join(str(t) for t in tags[:5])})"

            score = row.get("score")
            if isinstance(score, (int, float)):
                line += f" [score: {score:.2f}]"

            parts.append(line)

        return "\n".join(parts)
