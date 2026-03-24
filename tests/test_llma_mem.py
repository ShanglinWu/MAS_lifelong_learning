from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from marble.memory import EpisodicMemory, LLMAMem, ProceduralMemory, TransactiveMemory


def fake_embedding(model: str, input: str) -> list[float]:
    lowered = input.lower()
    return [
        float("api" in lowered or "error" in lowered),
        float("cache" in lowered or "latency" in lowered),
        float("database" in lowered or "query" in lowered),
        float(len(lowered.split())),
    ]


class TestLLMAMem(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_topology_factory_shared_reuses_backing_stores(self) -> None:
        memories = LLMAMem.create_for_topology(
            topology="shared",
            agent_ids=["a1", "a2"],
            persist_dir=str(self.tmp_path),
            consolidation_interval=99,
        )

        self.assertIs(memories["a1"].episodic, memories["a2"].episodic)
        self.assertIs(memories["a1"].procedural, memories["a2"].procedural)
        self.assertIs(memories["a1"].transactive, memories["a2"].transactive)

    def test_update_after_task_persists_episode_and_transactive_state(self) -> None:
        with patch(
            "marble.memory.episodic_memory.text_embedding",
            side_effect=fake_embedding,
        ):
            memory = LLMAMem(
                agent_id="agent-1",
                persist_dir=str(self.tmp_path),
                consolidation_interval=99,
            )
            memory.set_task_context("Diagnose the API timeout regression.")
            memory.record_action(
                {"tool": "inspect_logs", "result": "Observed cache misses."}
            )
            memory.update_after_task(
                task_description="Diagnose the API timeout regression.",
                team_composition=["agent-1", "agent-2"],
                outcome={
                    "success": True,
                    "metrics": {"performance": 0.9, "communication_overhead": 0.2},
                },
                context="Cache invalidation caused latency spikes.",
                task_type="incident-response",
            )

        self.assertEqual(memory.episodic.count(), 1)
        profile = memory.transactive.get_agent_profile("agent-1")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["total_tasks_completed"], 1)
        self.assertEqual(profile["success_count"], 1)
        self.assertTrue((self.tmp_path / "agent-1_episodic.json").exists())
        self.assertTrue((self.tmp_path / "agent-1_transactive.json").exists())

    def test_retrieval_prefers_procedural_memory_when_available(self) -> None:
        with patch(
            "marble.memory.procedural_memory.text_embedding",
            side_effect=fake_embedding,
        ):
            procedural = ProceduralMemory(
                agent_id="agent-1", persist_dir=str(self.tmp_path)
            )
            procedural.add_procedure(
                title="Investigate API latency with cache metrics",
                knowledge_content=(
                    "Inspect cache miss rate before checking downstream services."
                ),
                source_episodes=[0, 1],
                success_count=3,
                failure_count=0,
            )
            memory = LLMAMem(
                agent_id="agent-1",
                episodic=EpisodicMemory("agent-1", str(self.tmp_path)),
                procedural=procedural,
                transactive=TransactiveMemory("agent-1", str(self.tmp_path)),
                persist_dir=str(self.tmp_path),
                consolidation_interval=99,
            )
            memory.set_task_context(
                "Why is the API latency higher after the cache rollout?"
            )
            memory_str = memory.get_memory_str()

        self.assertIn("[Relevant Procedures & Strategies]", memory_str)
        self.assertIn("cache miss rate", memory_str)


if __name__ == "__main__":
    unittest.main()
