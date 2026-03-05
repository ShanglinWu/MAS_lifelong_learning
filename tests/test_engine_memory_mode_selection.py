from marble.engine.memory_selection import resolve_advanced_memory_mode


def test_resolve_memory_mode_none_enabled() -> None:
    mode = resolve_advanced_memory_mode(
        {"llma_mem": {"enabled": False}, "amem": {"enabled": False}}
    )
    assert mode == "none"


def test_resolve_memory_mode_llma_only() -> None:
    mode = resolve_advanced_memory_mode(
        {"llma_mem": {"enabled": True}, "amem": {"enabled": False}}
    )
    assert mode == "llmamem"


def test_resolve_memory_mode_amem_only() -> None:
    mode = resolve_advanced_memory_mode(
        {"llma_mem": {"enabled": False}, "amem": {"enabled": True}}
    )
    assert mode == "amem"


def test_resolve_memory_mode_rejects_dual_enable() -> None:
    try:
        resolve_advanced_memory_mode(
            {"llma_mem": {"enabled": True}, "amem": {"enabled": True}}
        )
    except ValueError as exc:
        assert "only one" in str(exc)
    else:
        raise AssertionError("Expected ValueError when both llma_mem and amem are enabled")
