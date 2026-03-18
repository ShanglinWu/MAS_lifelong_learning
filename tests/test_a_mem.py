from types import SimpleNamespace
from unittest.mock import patch

from marble.engine.engine import Engine
from marble.memory import AMemMemory, SharedMemory


def _fake_embedding(model: str, input: str) -> list[float]:
    lower = input.lower()
    return [
        1.0 if "weather" in lower else 0.0,
        1.0 if "sleep" in lower else 0.0,
        float(len(lower.split())),
    ]


def test_a_mem_memory_retrieves_relevant_notes() -> None:
    memory = AMemMemory(agent_id="agent_1", llm_model="")

    with patch("marble.memory.a_mem.text_embedding", side_effect=_fake_embedding):
        memory.update("agent_1", {"result": "It is cold weather today."})
        memory.update("agent_1", {"result": "I want to sleep early."})
        memory.set_task_context("What's the weather like today?")

        results = memory.retrieve("weather today", top_k=1)
        memory_str = memory.get_memory_str()

    assert results[0]["content"].find("weather") != -1
    assert "[Agentic Memory Baseline]" in memory_str
    assert "weather" in memory_str


def test_a_mem_memory_links_related_notes() -> None:
    memory = AMemMemory(agent_id="agent_1", llm_model="", link_threshold=0.6)

    with patch("marble.memory.a_mem.text_embedding", side_effect=_fake_embedding):
        memory.update("agent_1", "weather forecast says rain")
        memory.update("agent_1", "weather report says cold front")

    notes = memory.retrieve_all()
    assert notes[1]["links"]
    assert notes[1]["links"][0] == notes[0]["note_id"]


def test_engine_initializes_a_mem_memory_per_agent() -> None:
    engine = Engine.__new__(Engine)
    engine.agents = [
        SimpleNamespace(agent_id="agent_1", llm="model-a", memory=None),
        SimpleNamespace(agent_id="agent_2", llm="model-b", memory=None),
    ]
    engine.config = SimpleNamespace(llm="shared-model")
    engine.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)

    memory = engine._initialize_memory({"type": "AMemMemory"})

    assert isinstance(memory, SharedMemory)
    assert isinstance(engine.agents[0].memory, AMemMemory)
    assert engine.agents[0].memory.llm_model == "model-a"
    assert isinstance(engine.agents[1].memory, AMemMemory)
