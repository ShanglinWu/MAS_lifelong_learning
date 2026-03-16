"""
LLMA-Mem: LifeLong Multi-Agent Memory Framework.

A unified memory framework designed to support lifelong learning across diverse
multi-agent topologies. Consists of three interdependent memory modules:
    - Episodic Memory: task experience records
    - Procedural Memory: reusable strategies abstracted from episodes
    - Transactive Memory: agent capabilities and collaboration dynamics

Supports three memory topology configurations:
    - local:  each agent maintains private E, P, T
    - shared: centralized memory pool E_shared, P_shared, T_shared
    - hybrid: local E + local T + shared P + shared T

Memory lifecycle:
    1. Initialization  (all empty at t=0)
    2. Retrieval       (hierarchical: procedural first, fall back to episodic)
    3. Update          (episodic addition, procedural update, transactive update)
    4. Consolidation   (every N=3 episodes, extract procedures from episodes)
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from marble.llms.model_prompting import model_prompting
from marble.llms.text_embedding import text_embedding
from marble.memory.episodic_memory import EpisodicMemory
from marble.memory.procedural_memory import ProceduralMemory
from marble.memory.transactive_memory import TransactiveMemory


class LLMAMem:
    """
    LLMA-Mem: LifeLong Multi-Agent Memory.

    Orchestrates episodic, procedural, and transactive memory modules.
    Implements the full memory lifecycle: initialization, retrieval, update,
    and consolidation.

    Also provides backward-compatible interface (get_memory_str, update,
    retrieve_latest) so it can serve as a drop-in replacement for BaseMemory.
    """

    EMBEDDING_MODEL = "bedrock/amazon.titan-embed-text-v2:0"
    DEFAULT_CONSOLIDATION_INTERVAL = 3  # N = 3 episodes

    def __init__(
        self,
        agent_id: str,
        episodic: Optional[EpisodicMemory] = None,
        procedural: Optional[ProceduralMemory] = None,
        transactive: Optional[TransactiveMemory] = None,
        local_transactive: Optional[TransactiveMemory] = None,
        llm_model: str = "",
        persist_dir: str = "memory_store",
        consolidation_interval: int = 3,
    ) -> None:
        """
        Initialize LLMA-Mem.

        At system initialization (t=0), all memory stores are empty:
        E^(0) = ∅, P^(0) = ∅, T^(0) = ∅

        Args:
            agent_id: Agent identifier.
            episodic: EpisodicMemory instance (creates new if None).
            procedural: ProceduralMemory instance (creates new if None).
            transactive: TransactiveMemory instance (creates new if None).
            local_transactive: Local transactive memory (hybrid topology only).
            llm_model: LLM model for consolidation and lesson extraction.
            persist_dir: Directory for file persistence.
            consolidation_interval: Episodes between consolidations (N=3).
        """
        self.agent_id = agent_id
        self.llm_model = llm_model
        self.persist_dir = persist_dir
        self.consolidation_interval = consolidation_interval

        # Initialize memory modules (empty at t=0 if not pre-existing)
        self.episodic = episodic or EpisodicMemory(agent_id, persist_dir)
        self.procedural = procedural or ProceduralMemory(agent_id, persist_dir)
        self.transactive = transactive or TransactiveMemory(agent_id, persist_dir)
        # local_transactive: only used in hybrid topology (T_i^local)
        self.local_transactive = local_transactive

        # State for current task lifecycle
        self._current_actions: List[Dict[str, Any]] = []
        self._current_task: str = ""
        self._last_retrieved_procedures: List[Dict[str, Any]] = []
        self._episodes_since_consolidation: int = 0

        # Load metadata
        self._load_meta()

    # ==================================================================
    # Persistence helpers
    # ==================================================================

    def _get_meta_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.agent_id}_llma_meta.json")

    def _load_meta(self) -> None:
        meta_path = self._get_meta_path()
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self._episodes_since_consolidation = meta.get(
                    "episodes_since_consolidation", 0
                )

    def _save_meta(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        meta_path = self._get_meta_path()
        meta = {
            "episodes_since_consolidation": self._episodes_since_consolidation,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ==================================================================
    # Memory Retrieval
    # ==================================================================

    def set_task_context(self, task: str) -> None:
        """
        Set the current task context for retrieval and reset action tracking.

        Called at the start of each task execution.

        Args:
            task: The current task description.
        """
        self._current_task = task
        self._current_actions = []
        self._last_retrieved_procedures = []

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant memories following hierarchical logic:

        - If procedural memory is empty → retrieve from episodic memory
        - If procedural memory is not empty → retrieve from procedural memory
        - If the retrieve of procedural memory is None → retrieve from episodic

        Top-k memories with highest scores are returned as context string.

        Args:
            query: Query string (current task description).
            top_k: Number of top memories to retrieve.

        Returns:
            Formatted string of retrieved memories.
        """
        if self.procedural.is_empty():
            # Procedural empty → retrieve from episodic
            episodes = self.episodic.retrieve(query, top_k=top_k)
            if episodes:
                return self._format_episodic_results(episodes)
            return ""
        else:
            # Procedural not empty → try procedural first
            procedures = self.procedural.retrieve(query, top_k=top_k)
            if procedures is not None:
                self._last_retrieved_procedures = procedures
                return self._format_procedural_results(procedures)
            else:
                # Procedural retrieval returned None → fall back to episodic
                episodes = self.episodic.retrieve(query, top_k=top_k)
                if episodes:
                    return self._format_episodic_results(episodes)
                return ""

    def _format_episodic_results(self, episodes: List[Dict[str, Any]]) -> str:
        """Format episodic memory results into a prompt-ready string."""
        parts = ["[Past Experiences - for background context only, do not change your assigned tool actions]"]
        for ep in episodes:
            parts.append(f"- Similar past task: {ep['task_description'][:500]}")
            outcome_str = (
                "succeeded" if ep["outcome"].get("success", False) else "had issues"
            )
            parts.append(f"  Result: {outcome_str}")
            if ep.get("lessons_learned"):
                lessons = "; ".join(ep["lessons_learned"][:3])
                parts.append(f"  Takeaway: {lessons}")
        return "\n".join(parts)

    def _format_procedural_results(self, procedures: List[Dict[str, Any]]) -> str:
        """Format procedural memory results into a prompt-ready string."""
        parts = ["[Relevant Procedures & Strategies]"]
        for proc in procedures:
            parts.append(
                f"- {proc['title']} (success rate: {proc['success_rate']:.2f})"
            )
            parts.append(f"  Strategy: {proc['knowledge_content'][:300]}")
        return "\n".join(parts)

    # ==================================================================
    # Memory Update
    # ==================================================================

    def record_action(self, action_info: Dict[str, Any]) -> None:
        """
        Record an action taken during the current task (for episode creation).

        Args:
            action_info: Dictionary describing the action taken.
        """
        self._current_actions.append(action_info)

    def update_after_task(
        self,
        task_description: str,
        team_composition: List[str],
        outcome: Dict[str, Any],
        context: str = "",
        task_type: str = "",
    ) -> None:
        """
        Update memory after task completion. Handles:

        1. Episodic Addition:
           E^(t+1) = E^(t) ∪ {e_new}

        2. Procedural Update (for used procedures):
           s_j^(t+1) = s_j^(t) + 1_success
           f_j^(t+1) = f_j^(t) + 1_failure

        3. Transactive Memory Update:
           ξ_k = successes_k / total_tasks_k
           Also updates specialization areas, ρ and ω in team_patterns.

        4. Triggers consolidation every N episodes (N=3).

        Args:
            task_description: Description of the completed task.
            team_composition: List of agent IDs involved.
            outcome: {"success": bool, "metrics": dict, "error_info": str|None}
            context: Environmental context and state.
            task_type: Type of task for transactive memory.
        """
        success = outcome.get("success", False)

        # Extract lessons learned using LLM
        lessons = self._extract_lessons(
            task_description, self._current_actions, outcome, context
        )

        # Determine related procedures (from last retrieval)
        related_proc_ids = [
            str(p["procedure_id"]) for p in self._last_retrieved_procedures
        ]

        # --- 1. Episodic Addition: E^(t+1) = E^(t) ∪ {e_new} ---
        episode = self.episodic.add_episode(
            agent_id=self.agent_id,
            task_description=task_description,
            team_composition=team_composition,
            actions_taken=self._current_actions.copy(),
            outcome=outcome,
            context=context,
            lessons_learned=lessons,
            related_procedures=related_proc_ids,
            collaboration_quality=outcome.get("metrics", {}).get(
                "collaboration_quality", 0.0
            ),
        )

        # --- 2. Procedural Update ---
        for proc in self._last_retrieved_procedures:
            self.procedural.update_procedure_outcome(proc["procedure_id"], success)

        # --- 3. Transactive Memory Update ---
        # Update agent profile: ξ_k = successes_k / total_tasks_k
        specialization = task_type if success else ""
        self.transactive.update_agent_profile(
            agent_id=self.agent_id,
            task_success=success,
            task_type=task_type,
            partner_ids=team_composition,
            specialization_area=specialization,
        )
        # Also update local transactive if in hybrid topology
        if self.local_transactive is not None:
            self.local_transactive.update_agent_profile(
                agent_id=self.agent_id,
                task_success=success,
                task_type=task_type,
                partner_ids=team_composition,
                specialization_area=specialization,
            )

        # Update team pattern (ρ̄ and ω)
        performance = outcome.get("metrics", {}).get(
            "performance", 1.0 if success else 0.0
        )
        comm_overhead = outcome.get("metrics", {}).get(
            "communication_overhead", 0.0
        )
        self.transactive.update_team_pattern(
            team_composition=team_composition,
            task_type=task_type,
            performance=performance,
            communication_overhead=comm_overhead,
            success=success,
        )
        if self.local_transactive is not None:
            self.local_transactive.update_team_pattern(
                team_composition=team_composition,
                task_type=task_type,
                performance=performance,
                communication_overhead=comm_overhead,
                success=success,
            )

        # Clear current task state
        self._current_actions = []
        self._last_retrieved_procedures = []

        # --- 4. Check if consolidation is needed (every N episodes) ---
        self._episodes_since_consolidation += 1
        if self._episodes_since_consolidation >= self.consolidation_interval:
            self._consolidate()
            self._episodes_since_consolidation = 0
        self._save_meta()

    def _extract_lessons(
        self,
        task_description: str,
        actions: List[Dict[str, Any]],
        outcome: Dict[str, Any],
        context: str = "",
    ) -> List[str]:
        """
        Extract lessons learned from a task experience using LLM.

        Args:
            task_description: Description of the task.
            actions: Actions taken during the task.
            outcome: Task outcome information.

        Returns:
            List of extracted lesson strings.
        """
        if not self.llm_model:
            # Fallback: basic lessons from outcome
            if outcome.get("success", False):
                return [f"Successfully completed task: {task_description[:100]}"]
            else:
                error = outcome.get("error_info", "unknown error")
                return [f"Failed task due to: {error}"]

        actions_str = json.dumps(actions[:5], default=str)[:500]
        outcome_str = json.dumps(outcome, default=str)[:500]

        success = outcome.get("success", False)
        if success:
            focus = (
                "Focus on what strategies and actions led to success "
                "so they can be replicated in future tasks."
            )
        else:
            error_info = outcome.get("error_info", "")
            dim_scores = outcome.get("metrics", {}).get("dimension_scores", {})
            focus = (
                "The task FAILED. Focus on diagnosing the root cause of failure. "
                "Identify what went wrong, which actions were missing or incorrect, "
                "and what should be done differently next time. Please be detailed.\n"
            )
            if dim_scores:
                low_dims = [k for k, v in dim_scores.items() if isinstance(v, (int, float)) and v <= 2]
                if low_dims:
                    focus += f"The lowest scoring dimensions were: {', '.join(low_dims)}.\n"
            if error_info:
                focus += f"Error details: {error_info}\n"

        # Include task summary (what each agent did) when available
        context_section = ""
        if context:
            context_section = f"Task Summary (what each agent did and results):\n{context[:1000]}\n\n"

        prompt = (
            "Based on the following task experience, extract 1-2 concise, actionable lessons learned.\n"
            "Focus on CONCRETE actions: which tool functions should have been called, "
            "what code patterns worked or failed, and what specific steps to take next time.\n"
            "Do NOT suggest vague advice like 'communicate better' or 'provide clearer instructions'.\n"
            "Instead, suggest specific tool calls or coding strategies.\n\n"
            f"Task: {task_description[:500]}\n\n"
            f"{context_section}"
            f"Actions taken: {actions_str}\n"
            f"Outcome: {outcome_str}\n\n"
            f"{focus}\n"
            "Return the lessons as a JSON array of strings. Example:\n"
            '["Lesson 1", "Lesson 2", "Lesson 3"]\n'
        )

        try:
            response = model_prompting(
                llm_model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a reflective learning assistant that extracts "
                            "actionable lessons from task experiences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                return_num=1,
                max_token_num=512,
                temperature=0.7,
                top_p=None,
                stream=None,
            )[0]
            content = response.content if response.content else "[]"
            # Parse JSON array from response
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                lessons = json.loads(content[start : end + 1])
                return lessons if isinstance(lessons, list) else [str(lessons)]
            return [content.strip()]
        except Exception:
            if outcome.get("success", False):
                return [f"Completed: {task_description[:100]}"]
            else:
                return [f"Failed: {task_description[:100]}"]

    # ==================================================================
    # Memory Consolidation (Algorithm 1)
    # ==================================================================

    def _consolidate(self) -> None:
        """
        Procedural Knowledge Extraction (Algorithm 1 from spec):

        1. Cluster episodes by lessons_learned similarity using embedding space.
        2. For each cluster C with |C| >= 2:
           - E_success ← successful episodes in C
           - Create procedure with generalized strategy/skills from E_success
           - Source episodes ← E_success
        3. Prune overlap procedures (if one procedure's source episodes are
           included in another's, delete the smaller one).
        """
        episodes_with_embeddings = self.episodic.get_lessons_with_embeddings()

        if len(episodes_with_embeddings) < 2:
            return

        episodes = [e for e, _ in episodes_with_embeddings]
        embeddings = np.array([emb for _, emb in episodes_with_embeddings])

        # Cluster by lessons_learned similarity using embedding space
        try:
            # Compute distance matrix from cosine similarity
            sim_matrix = sklearn_cosine_similarity(embeddings)
            dist_matrix = 1.0 - sim_matrix
            np.fill_diagonal(dist_matrix, 0)
            # Clamp negative distances to 0
            dist_matrix = np.maximum(dist_matrix, 0)

            n_clusters = max(1, len(embeddings) // 2)
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )
            labels = clustering.fit_predict(dist_matrix)
        except Exception:
            return

        # Group episodes by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for i, label in enumerate(labels):
            label_int = int(label)
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(episodes[i])

        # Get existing source episode sets to avoid duplicates
        existing_source_sets = self.procedural.get_existing_source_sets()

        # For each cluster with |C| >= 2
        for cluster_id, cluster_episodes in clusters.items():
            if len(cluster_episodes) < 2:
                continue

            # E_success ← successful episodes in C
            success_episodes = [
                ep
                for ep in cluster_episodes
                if ep["outcome"].get("success", False)
            ]

            if not success_episodes:
                continue

            source_episode_ids = sorted(
                [ep["episode_id"] for ep in success_episodes]
            )

            # Skip if this exact set of source episodes already exists
            source_set = set(source_episode_ids)
            if any(source_set == existing for existing in existing_source_sets):
                continue

            # Create procedure with generalized strategy/skills from E_success
            title, knowledge = self._generalize_strategy(success_episodes)

            success_count = len(success_episodes)
            failure_count = len(cluster_episodes) - success_count

            self.procedural.add_procedure(
                title=title,
                knowledge_content=knowledge,
                source_episodes=source_episode_ids,
                success_count=success_count,
                failure_count=failure_count,
            )

        # Prune overlap procedures
        self.procedural.prune_overlap()

    def _generalize_strategy(
        self, episodes: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Use LLM to generalize a reusable strategy from successful episodes.

        Args:
            episodes: List of successful episode dicts.

        Returns:
            (title, knowledge_content) tuple.
        """
        if not self.llm_model:
            # Fallback: combine lessons directly
            all_lessons: List[str] = []
            for ep in episodes:
                all_lessons.extend(ep.get("lessons_learned", []))
            title = f"Strategy from {len(episodes)} episodes"
            knowledge = (
                "; ".join(all_lessons) if all_lessons else "Combined experience"
            )
            return title, knowledge

        episodes_str = ""
        for i, ep in enumerate(episodes[:5]):  # Limit for prompt size
            episodes_str += (
                f"Episode {i + 1}:\n"
                f"  Task: {ep['task_description'][:200]}\n"
                f"  Lessons: {ep.get('lessons_learned', [])}\n"
                f"  Outcome: {json.dumps(ep['outcome'], default=str)[:200]}\n\n"
            )

        prompt = (
            "Based on the following successful task experiences, extract a generalized "
            "strategy or skill that can be reused in similar future situations:\n\n"
            f"{episodes_str}\n"
            "Respond in JSON format:\n"
            '{"title": "Short descriptive title", '
            '"knowledge_content": "Detailed reusable strategy description"}\n'
        )

        try:
            response = model_prompting(
                llm_model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract generalized, reusable strategies "
                            "from task experiences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                return_num=1,
                max_token_num=512,
                temperature=0.7,
                top_p=None,
                stream=None,
            )[0]
            content = response.content if response.content else "{}"
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(content[start : end + 1])
                return (
                    data.get("title", "Extracted Strategy"),
                    data.get("knowledge_content", ""),
                )
            return "Extracted Strategy", content.strip()
        except Exception:
            all_lessons = []
            for ep in episodes:
                all_lessons.extend(ep.get("lessons_learned", []))
            return (
                f"Strategy from {len(episodes)} episodes",
                "; ".join(all_lessons),
            )

    # ==================================================================
    # Backward-Compatible Interface (drop-in for BaseMemory)
    # ==================================================================

    def update(self, key: str, information: Any) -> None:
        """
        Backward-compatible update.

        Records the information as an action in the current task's action
        history (used for episodic memory creation at end of task).

        Args:
            key: Key (kept for interface compatibility).
            information: Action/result information to record.
        """
        self._current_actions.append(information)

    def retrieve_latest(self) -> Any:
        """Backward-compatible: retrieve the latest episodic entry."""
        episodes = self.episodic.get_recent(1)
        return episodes[0] if episodes else None

    def retrieve_all(self) -> List[Any]:
        """Backward-compatible: retrieve all episodes."""
        return self.episodic.get_all()

    def get_memory_str(self) -> str:
        """
        Backward-compatible: return relevant memory as a string.

        Uses the current task context for retrieval via the hierarchical
        logic (procedural first, fall back to episodic).

        Returns:
            Formatted string of retrieved memories for prompt inclusion.
        """
        if self._current_task:
            return self.retrieve(self._current_task)

        # If no task context, return recent episodes summary
        recent = self.episodic.get_recent(2)
        if recent:
            return self._format_episodic_results(recent)
        return ""

    def get_transactive_str(self) -> str:
        """
        Get transactive memory as string for planning prompts.

        Returns:
            Formatted string of transactive memory for task allocation.
        """
        return self.transactive.to_str()

    # ==================================================================
    # Topology Factory
    # ==================================================================

    @staticmethod
    def create_for_topology(
        topology: str,
        agent_ids: List[str],
        llm_model: str = "",
        persist_dir: str = "memory_store",
        consolidation_interval: int = 3,
    ) -> Dict[str, "LLMAMem"]:
        """
        Create LLMAMem instances for all agents based on topology configuration.

        Topologies:
        - local:  Each agent has private M_i = <E_i, P_i, T_i>.
                  No direct cross-agent memory access.
        - shared: Centralized pool M_shared = <E_shared, P_shared, T_shared>.
                  All agents share the same memory.
        - hybrid: M_i^hybrid = <E_i^local, T_i^local, P_shared, T_shared>.
                  Local episodic + local transactive, shared procedural + shared transactive.

        Args:
            topology: "local", "shared", or "hybrid".
            agent_ids: List of agent identifiers.
            llm_model: LLM model for consolidation.
            persist_dir: Persistence directory.
            consolidation_interval: Episodes between consolidations (N=3).

        Returns:
            Dict mapping agent_id to LLMAMem instance.
        """
        memories: Dict[str, LLMAMem] = {}

        if topology == "local":
            # Local: each agent has private E, P, T
            for agent_id in agent_ids:
                memories[agent_id] = LLMAMem(
                    agent_id=agent_id,
                    llm_model=llm_model,
                    persist_dir=persist_dir,
                    consolidation_interval=consolidation_interval,
                )

        elif topology == "shared":
            # Shared: centralized E_shared, P_shared, T_shared
            shared_episodic = EpisodicMemory("shared", persist_dir)
            shared_procedural = ProceduralMemory("shared", persist_dir)
            shared_transactive = TransactiveMemory("shared", persist_dir)

            for agent_id in agent_ids:
                memories[agent_id] = LLMAMem(
                    agent_id=agent_id,
                    episodic=shared_episodic,
                    procedural=shared_procedural,
                    transactive=shared_transactive,
                    llm_model=llm_model,
                    persist_dir=persist_dir,
                    consolidation_interval=consolidation_interval,
                )

        elif topology == "hybrid":
            # Hybrid: local E + local T, shared P + shared T
            shared_procedural = ProceduralMemory("shared", persist_dir)
            shared_transactive = TransactiveMemory("shared", persist_dir)

            for agent_id in agent_ids:
                local_episodic = EpisodicMemory(agent_id, persist_dir)
                local_trans = TransactiveMemory(agent_id, persist_dir)

                memories[agent_id] = LLMAMem(
                    agent_id=agent_id,
                    episodic=local_episodic,
                    procedural=shared_procedural,
                    transactive=shared_transactive,
                    local_transactive=local_trans,
                    llm_model=llm_model,
                    persist_dir=persist_dir,
                    consolidation_interval=consolidation_interval,
                )

        else:
            raise ValueError(
                f"Unknown topology: {topology}. Must be 'local', 'shared', or 'hybrid'."
            )

        return memories
