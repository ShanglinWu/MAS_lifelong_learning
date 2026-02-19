from marble.memory.llma_mem import LLMAMemManager, MemoryTopology, MemoryTopologyManager
from marble.memory.llma_mem.schemas import ProceduralRecord


def test_consolidation_extracts_procedure_without_crashing() -> None:
    manager = LLMAMemManager(
        agent_id="agent_1",
        topology_manager=MemoryTopologyManager(MemoryTopology.LOCAL),
        consolidation_interval=10,
        enable_auto_embedding=False,
    )

    for _ in range(10):
        manager.record_episode(
            task_description="diagnose issue",
            actions_taken=["inspect logs", "restart service"],
            outcome="success",
            team_composition=["agent_1", "agent_2"],
            embedding=[0.1, 0.2, 0.3],
        )

    assert manager.procedural.count() >= 1


def test_query_embedding_drives_retrieval_and_context() -> None:
    manager = LLMAMemManager(
        agent_id="agent_1",
        topology_manager=MemoryTopologyManager(MemoryTopology.LOCAL),
        enable_auto_embedding=False,
    )

    manager.record_episode(
        task_description="task-a",
        actions_taken=["a1"],
        outcome="success",
        team_composition=["agent_1"],
        embedding=[1.0, 0.0],
    )
    manager.record_episode(
        task_description="task-b",
        actions_taken=["b1"],
        outcome="failure",
        team_composition=["agent_1"],
        embedding=[0.0, 1.0],
    )

    results = manager.retrieve_relevant_memories(
        query_embedding=[1.0, 0.0], query_team=["agent_1"], k=1
    )
    assert len(results) == 1
    top_record, _ = results[0]
    assert top_record.task_description == "task-a"

    context = manager.get_memory_context_str(
        query_embedding=[1.0, 0.0], query_team=["agent_1"], k=1
    )
    assert "Strategic Memory (LLMA-Mem)" in context


def test_hybrid_team_state_keeps_local_profile_isolated() -> None:
    topology = MemoryTopologyManager(MemoryTopology.HYBRID)
    manager_a = LLMAMemManager(
        agent_id="agent_a",
        topology_manager=topology,
        enable_auto_embedding=False,
    )
    manager_b = LLMAMemManager(
        agent_id="agent_b",
        topology_manager=topology,
        enable_auto_embedding=False,
    )

    manager_a.record_episode(
        task_description="task-a",
        actions_taken=["a"],
        outcome="success",
        team_composition=["agent_a", "agent_b"],
        embedding=[1.0, 0.0],
        task_success=True,
    )

    profile_a_seen_by_a = manager_a.team_state.get_agent_profile("agent_a")
    profile_a_seen_by_b = manager_b.team_state.get_agent_profile("agent_a")

    assert profile_a_seen_by_a is not None
    assert profile_a_seen_by_b is None


def test_retrieved_procedure_is_updated_after_task_feedback() -> None:
    manager = LLMAMemManager(
        agent_id="agent_1",
        topology_manager=MemoryTopologyManager(MemoryTopology.LOCAL),
        enable_auto_embedding=False,
    )
    proc = ProceduralRecord(
        agent_id="agent_1",
        title="use checklist",
        type="strategy",
        knowledge_content="Always validate assumptions.",
        embedding=[1.0, 0.0],
    )
    manager.procedural.add_procedure(proc)

    _ = manager.get_memory_context_str(
        query_embedding=[1.0, 0.0],
        query_team=["agent_1"],
        k=1,
    )
    manager.record_episode(
        task_description="task-c",
        actions_taken=["step-1"],
        outcome="completed successfully",
        embedding=[1.0, 0.0],
        task_success=True,
    )

    updated = manager.procedural.get_procedure(proc.procedure_id)
    assert updated is not None
    assert updated.success_count == 1
    assert updated.viewed_times >= 1


def test_prune_requires_staleness_and_low_confidence() -> None:
    manager = LLMAMemManager(
        agent_id="agent_1",
        topology_manager=MemoryTopologyManager(MemoryTopology.LOCAL),
        consolidation_interval=1,
        enable_auto_embedding=False,
    )
    proc = ProceduralRecord(
        agent_id="agent_1",
        title="fragile strategy",
        type="strategy",
        knowledge_content="Not reliable",
        success_count=0,
        failure_count=20,
    )
    manager.procedural.add_procedure(proc)

    # Mark as recently used at episode 8; at episode 10 it should not be pruned for K=5.
    manager.procedural.increment_viewed(proc.procedure_id, current_episode_count=8)
    pruned_now = manager.procedural.prune(
        min_confidence=0.3,
        current_episode_count=10,
        min_unused_episodes=5,
    )
    assert pruned_now == 0

    pruned_later = manager.procedural.prune(
        min_confidence=0.3,
        current_episode_count=14,
        min_unused_episodes=5,
    )
    assert pruned_later == 1


def test_success_inference_handles_failure_keywords() -> None:
    manager = LLMAMemManager(
        agent_id="agent_1",
        topology_manager=MemoryTopologyManager(MemoryTopology.LOCAL),
        enable_auto_embedding=False,
    )
    ep = manager.record_episode(
        task_description="task-d",
        actions_taken=["step-1"],
        outcome="Execution failed with timeout error",
        embedding=[0.0, 1.0],
    )
    assert ep.outcome_success is False
