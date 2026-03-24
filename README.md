# LLMA-Mem

This repository has been reduced to the LLMA-Mem code only.

LLMA-Mem is a lifelong multi-agent memory framework with three coordinated memory layers:

- `EpisodicMemory`: stores full task experiences.
- `ProceduralMemory`: stores reusable strategies extracted from past work.
- `TransactiveMemory`: stores agent capabilities and team collaboration patterns.

The package entrypoint is `marble.memory.LLMAMem`.

## What was removed

The previous repo mixed LLMA-Mem with MultiAgentBench datasets, MARBLE environments, AMem code, generated experiment outputs, and benchmark scripts. Those parts were removed so the repo is focused on LLMA-Mem and easier to run directly.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install poetry
poetry install
```

If you want to use real embedding and LLM backends, set the environment variables required by your LiteLLM provider before running your own experiments.

## Quickstart

Run the local demo:

```bash
poetry run python examples/llma_mem_quickstart.py
```

The demo is plug and play:

- it does not call external APIs,
- it uses mocked embeddings,
- it writes sample memory artifacts to `memory_store/demo/`,
- it shows how one agent writes memory and another retrieves it.

## General Usage Guideline

1. Create one `LLMAMem` instance per agent, or use `LLMAMem.create_for_topology(...)` for `local`, `shared`, or `hybrid` memory layouts.
2. Call `set_task_context(task_description, agent_profile=...)` at the start of each task.
3. Record intermediate actions with `record_action(...)` or `update(...)`.
4. Retrieve prompt-ready memory context with `get_memory_str()`.
5. After the task finishes, call `update_after_task(...)` with:
   - `task_description`
   - `team_composition`
   - `outcome`
   - optional `context`
   - optional `task_type`
6. Persist and reload memory by pointing future runs to the same `persist_dir`.

## Minimal Example

```python
from marble.memory import LLMAMem

team = LLMAMem.create_for_topology(
    topology="shared",
    agent_ids=["planner", "analyst"],
    persist_dir="memory_store/my_run",
    consolidation_interval=5,
)

planner = team["planner"]
planner.set_task_context("Diagnose the API timeout regression.")
planner.record_action({"tool": "inspect_logs", "result": "Timeouts started after cache rollout."})

planner.update_after_task(
    task_description="Diagnose the API timeout regression.",
    team_composition=["planner", "analyst"],
    outcome={"success": True, "metrics": {"performance": 0.9}},
    context="Cache invalidation caused a miss-rate spike.",
    task_type="incident-response",
)
```

## Testing

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The included test suite covers:

- topology creation,
- episode persistence,
- transactive-memory updates,
- procedural-memory retrieval.

## Repo Layout

```text
marble/
  llms/
  memory/
examples/
tests/
README.md
pyproject.toml
```
