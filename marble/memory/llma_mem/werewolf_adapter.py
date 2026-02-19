"""Adapter bridging LLMA-Mem to WerewolfAgent's interaction pattern."""

from typing import Any, Dict, List, Optional

from .memory_manager import LLMAMemManager


class WerewolfMemoryAdapter:
    """
    Bridges LLMA-Mem to the WerewolfAgent's plain-dict workflow.

    The plain-dict shared_memory in WerewolfEnv is completely untouched;
    this adapter adds an orthogonal memory layer on top.
    """

    def __init__(self, manager: LLMAMemManager):
        self.manager = manager

    def record_game_action(
        self,
        event_type: str,
        result: Dict[str, Any],
        game_state: Optional[Dict[str, Any]] = None,
        team: Optional[List[str]] = None,
    ) -> None:
        """
        Record a game action as an episodic memory.

        Args:
            event_type: The werewolf event type (e.g. "vote_action", "werewolf_action")
            result: The action result dict from the agent's decision
            game_state: Current public game state snapshot
            team: List of agent IDs on the same team
        """
        action_str = result.get("action", "unknown")
        target = result.get("target", "")
        outcome_str = f"{event_type}: {action_str}"
        if target:
            outcome_str += f" -> {target}"

        actions_taken = [f"{event_type}:{action_str}"]
        if target:
            actions_taken.append(f"target:{target}")

        self.manager.record_episode(
            task_description=f"Werewolf game: {event_type}",
            actions_taken=actions_taken,
            outcome=outcome_str,
            team_composition=team or [],
            context_state=game_state or {},
            lessons_learned="",
            collaboration_quality=0.5,
        )

    def get_strategic_context(
        self,
        situation: str = "",
        alive_players: Optional[List[str]] = None,
    ) -> str:
        """
        Get strategic memory context for prompt injection.

        Args:
            situation: Description of current game situation
            alive_players: List of currently alive player IDs

        Returns:
            Formatted string of relevant strategic memories
        """
        return self.manager.get_memory_context_str(
            query_text=situation,
            query_team=alive_players or [],
            k=3,
        )
