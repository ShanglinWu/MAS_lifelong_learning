<div align="center">
  <h1>LLMA-Mem</h1>
  <p><strong>Lifelong Multi-Agent Memory</strong></p>
  <p>A focused Python package for persistent episodic, procedural, and transactive memory in multi-agent systems.</p>
</div>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#figures">Figures</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#usage">Usage</a> •
  <a href="#testing">Testing</a>
</p>

## Overview

LLMA-Mem gives multi-agent workflows a memory system that survives beyond a single run.

It combines three memory layers:

| Layer | Purpose |
| --- | --- |
| `EpisodicMemory` | Stores full task experiences and outcomes |
| `ProceduralMemory` | Stores reusable strategies distilled from prior work |
| `TransactiveMemory` | Stores agent capability profiles and collaboration patterns |

The main entrypoint is `llmamem.memory.LLMAMem`.

## Figures

The repository includes the original LLMA-Mem figures as PDF assets:

| Figure | Description |
| --- | --- |
| [Architecture_topology.pdf](images/Architecture_topology.pdf) | System structure and supported memory topologies |
| [lifecycle.pdf](images/lifecycle.pdf) | Retrieval, update, and consolidation lifecycle |

If your Markdown viewer supports PDF preview, open them directly from the links above. On GitHub, they will open in the repository file viewer.

## Why This Repo

This repository is intentionally trimmed down to LLMA-Mem only.

- No MultiAgentBench datasets
- No MARBLE environment framework
- No benchmark scripts or generated experiment outputs
- No Poetry-based workflow

The result is a smaller, cleaner package that is easier to install, test, and plug into your own agent stack.

## Installation

Set up a plain Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

If you want real provider-backed LLM and embedding calls, install the optional dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quickstart

Run the local demo:

```bash
python3 examples/llma_mem_quickstart.py
```

What the demo does:

- runs without external APIs
- mocks embeddings locally
- writes sample artifacts into `memory_store/demo/`
- shows one agent writing memory and another retrieving it

## Usage

### Recommended flow

1. Create one `LLMAMem` instance per agent, or use `LLMAMem.create_for_topology(...)`.
2. Choose a topology: `local`, `shared`, or `hybrid`.
3. Call `set_task_context(...)` before each task.
4. Record actions with `record_action(...)` or `update(...)`.
5. Pull prompt-ready memory with `get_memory_str()`.
6. Finalize the task with `update_after_task(...)`.
7. Reuse the same `persist_dir` across runs to keep memory persistent.

### Minimal example

```python
from llmamem.memory import LLMAMem

team = LLMAMem.create_for_topology(
    topology="shared",
    agent_ids=["planner", "analyst"],
    persist_dir="memory_store/my_run",
    consolidation_interval=5,
)

planner = team["planner"]
planner.set_task_context("Diagnose the API timeout regression.")
planner.record_action(
    {"tool": "inspect_logs", "result": "Timeouts started after cache rollout."}
)

planner.update_after_task(
    task_description="Diagnose the API timeout regression.",
    team_composition=["planner", "analyst"],
    outcome={"success": True, "metrics": {"performance": 0.9}},
    context="Cache invalidation caused a miss-rate spike.",
    task_type="incident-response",
)
```

### Topologies

| Topology | Behavior |
| --- | --- |
| `local` | Each agent has private episodic, procedural, and transactive memory |
| `shared` | All agents share the same memory stores |
| `hybrid` | Local episodic memory with shared higher-level memory structures |

## Testing

Run the built-in tests with standard library tooling:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The current tests cover:

- shared-topology initialization
- episode persistence
- transactive-memory updates
- procedural-memory retrieval priority

## Repository Layout

```text
llmamem/
  llms/
  memory/
examples/
images/
tests/
README.md
requirements.txt
setup.py
```

## Notes

- `requirements.txt` is optional and only needed for real LiteLLM-backed calls.
- The package imports cleanly without those provider dependencies for local demo/testing.
- Generated memory files are written under `memory_store/` and are ignored by git.
