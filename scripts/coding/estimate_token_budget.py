#!/usr/bin/env python3
"""
Rough token budget estimator for MARBLE coding run outputs.

This is an offline estimator over generated task JSONL files.
It does not require model/provider telemetry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def est_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def iter_task_files(result_dir: Path) -> Iterable[Path]:
    if result_dir.is_file() and result_dir.suffix == ".jsonl":
        yield result_dir
        return
    for p in sorted(result_dir.glob("task_*.jsonl")):
        if p.is_file():
            yield p


def estimate_file(path: Path) -> Dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    records = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records += 1
            data = json.loads(line)

            task_text = str(data.get("task", ""))
            input_tokens += est_tokens(task_text)

            for it in data.get("iterations", []) or []:
                task_assignments = it.get("task_assignments", {}) or {}
                for prompt in task_assignments.values():
                    input_tokens += est_tokens(str(prompt))

                task_results = it.get("task_results", []) or []
                for tr in task_results:
                    result_text = tr.get("result", "") if isinstance(tr, dict) else ""
                    output_tokens += est_tokens(str(result_text))

                summary = it.get("summary", "")
                output_tokens += est_tokens(str(summary))

    return {
        "records": records,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate input/output token budget from task JSONL outputs."
    )
    parser.add_argument(
        "path",
        help="Path to a task_*.jsonl file or a directory containing task_*.jsonl files.",
    )
    args = parser.parse_args()

    root = Path(args.path)
    files = list(iter_task_files(root))
    if not files:
        print(f"No task_*.jsonl files found under: {root}")
        return 1

    total_in = 0
    total_out = 0
    total_records = 0

    print("file,records,input_tokens,output_tokens,total_tokens")
    for f in files:
        stats = estimate_file(f)
        total_records += stats["records"]
        total_in += stats["input_tokens"]
        total_out += stats["output_tokens"]
        print(
            f"{f},{stats['records']},{stats['input_tokens']},"
            f"{stats['output_tokens']},{stats['total_tokens']}"
        )

    print(
        f"TOTAL,{total_records},{total_in},{total_out},{total_in + total_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

