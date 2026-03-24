"""Local LLMA-Mem quickstart that runs without external APIs."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marble.memory import LLMAMem


def fake_embedding(model: str, input: str) -> list[float]:
    lowered = input.lower()
    return [
        float("api" in lowered or "error" in lowered),
        float("cache" in lowered or "latency" in lowered),
        float("database" in lowered or "query" in lowered),
        float(len(lowered.split())),
    ]


def main() -> None:
    persist_dir = Path("memory_store/demo")
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    with patch("marble.memory.episodic_memory.text_embedding", side_effect=fake_embedding), patch(
        "marble.memory.procedural_memory.text_embedding",
        side_effect=fake_embedding,
    ):
        team = LLMAMem.create_for_topology(
            topology="shared",
            agent_ids=["planner", "analyst"],
            persist_dir=str(persist_dir),
            consolidation_interval=99,
        )
        planner = team["planner"]

        planner.set_task_context("Diagnose the API timeout regression.")
        planner.record_action(
            {"tool": "inspect_logs", "result": "Timeout spikes appeared after cache invalidation."}
        )
        planner.update_after_task(
            task_description="Diagnose the API timeout regression.",
            team_composition=["planner", "analyst"],
            outcome={
                "success": True,
                "metrics": {"performance": 0.92, "communication_overhead": 0.15},
            },
            context="Cache miss rate increased after the rollout.",
            task_type="incident-response",
        )

        analyst = team["analyst"]
        analyst.set_task_context("Investigate the API timeout regression.")
        memory_str = analyst.get_memory_str()

    print("Retrieved memory context:")
    print(memory_str or "<empty>")
    print("\nPersisted files:")
    for path in sorted(persist_dir.glob("*")):
        print(f"- {path}")


if __name__ == "__main__":
    main()
