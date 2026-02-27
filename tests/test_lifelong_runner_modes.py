from scripts.lifelong.run_lifelong_experiment import apply_memory_mode_config


def _base_config():
    return {"memory": {"type": "SharedMemory"}}


def test_apply_memory_mode_config_baseline() -> None:
    data = _base_config()
    apply_memory_mode_config(
        data=data,
        mode="baseline",
        consolidation_interval=3,
        llma_topology="local",
        amem_topology="local",
        amem_evolution_enabled=True,
        amem_evolution_threshold=100,
        amem_embedding_model="all-MiniLM-L6-v2",
        amem_llm_backend="openai",
        amem_llm_model="gpt-4o-mini",
        amem_retrieval_k=5,
    )

    assert data["memory"]["llma_mem"]["enabled"] is False
    assert data["memory"]["amem"]["enabled"] is False


def test_apply_memory_mode_config_llmamem() -> None:
    data = _base_config()
    apply_memory_mode_config(
        data=data,
        mode="llmamem",
        consolidation_interval=7,
        llma_topology="hybrid",
        amem_topology="local",
        amem_evolution_enabled=False,
        amem_evolution_threshold=50,
        amem_embedding_model="all-MiniLM-L6-v2",
        amem_llm_backend="openai",
        amem_llm_model="gpt-4o-mini",
        amem_retrieval_k=3,
    )

    llma = data["memory"]["llma_mem"]
    amem = data["memory"]["amem"]
    assert llma["enabled"] is True
    assert llma["topology"] == "hybrid"
    assert llma["consolidation_interval"] == 7
    assert amem["enabled"] is False


def test_apply_memory_mode_config_amem() -> None:
    data = _base_config()
    apply_memory_mode_config(
        data=data,
        mode="amem",
        consolidation_interval=2,
        llma_topology="local",
        amem_topology="shared",
        amem_evolution_enabled=True,
        amem_evolution_threshold=123,
        amem_embedding_model="all-MiniLM-L6-v2",
        amem_llm_backend="openai",
        amem_llm_model="gpt-4o-mini",
        amem_retrieval_k=9,
    )

    llma = data["memory"]["llma_mem"]
    amem = data["memory"]["amem"]
    assert llma["enabled"] is False
    assert amem["enabled"] is True
    assert amem["topology"] == "shared"
    assert amem["evolution_enabled"] is True
    assert amem["evolution_threshold"] == 123
    assert amem["embedding_model"] == "all-MiniLM-L6-v2"
    assert amem["llm_backend"] == "openai"
    assert amem["llm_model"] == "gpt-4o-mini"
    assert amem["retrieval_k"] == 9
