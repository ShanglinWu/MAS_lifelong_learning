import json

from marble.memory.amem.manager import AMEMManager


class _FakeMemorySystem:
    def __init__(self) -> None:
        self.added_notes = []
        self.search_rows = []

    def add_note(self, content: str, **kwargs):
        self.added_notes.append({"content": content, "kwargs": kwargs})
        return "fake-note-id"

    def search_agentic(self, query: str, k: int = 5):
        _ = query
        return self.search_rows[:k]


class _FakeTopologyManager:
    def __init__(self, system: _FakeMemorySystem) -> None:
        self.system = system

    def get_memory_system(self, agent_id: str) -> _FakeMemorySystem:
        _ = agent_id
        return self.system


def test_record_episode_stores_note_payload() -> None:
    system = _FakeMemorySystem()
    manager = AMEMManager(
        agent_id="agent_1",
        topology_manager=_FakeTopologyManager(system),
        retrieval_k=5,
    )

    note_id = manager.record_episode(
        task_description="Diagnose latency spike",
        actions_taken=["inspect logs", "check metrics"],
        outcome="Issue resolved successfully",
        context_state={"service": "api"},
        task_success=True,
    )

    assert note_id == "fake-note-id"
    assert len(system.added_notes) == 1

    payload_raw = system.added_notes[0]["content"]
    payload = json.loads(payload_raw)
    assert payload["agent_id"] == "agent_1"
    assert payload["task_description"] == "Diagnose latency spike"
    assert payload["task_success"] is True

    tags = system.added_notes[0]["kwargs"]["tags"]
    assert "episode" in tags
    assert "success" in tags


def test_get_memory_context_str_is_deterministic() -> None:
    system = _FakeMemorySystem()
    system.search_rows = [
        {
            "id": "n1",
            "content": "{\"task\":\"t1\",\"outcome\":\"ok\"}",
            "tags": ["episode", "success"],
            "score": 0.12,
            "is_neighbor": False,
        },
        {
            "id": "n2",
            "content": "{\"task\":\"t2\",\"outcome\":\"retry\"}",
            "tags": ["episode", "failure"],
            "is_neighbor": True,
        },
    ]

    manager = AMEMManager(
        agent_id="agent_1",
        topology_manager=_FakeTopologyManager(system),
        retrieval_k=2,
    )

    context = manager.get_memory_context_str(query_text="debug timeout", k=2)
    assert "Strategic Memory (A-MEM)" in context
    assert "[Note]" in context
    assert "[Neighbor]" in context
    assert "tags: episode, success" in context
