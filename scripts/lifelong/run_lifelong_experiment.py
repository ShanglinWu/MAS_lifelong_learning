#!/usr/bin/env python3
"""
Run lifelong experiments and compute AP/AIP with task-specific score adapters.

Formulas (from LLMAMem paper):
  AP_t = (1 / t) * sum_{i=1..t} J_i
  AIP  = (1 / T) * sum_{t=1..T} AP_t
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


TASK_TYPES = {"coding", "research", "db", "minecraft"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lifelong experiments (baseline vs LLMA-Mem)."
    )
    parser.add_argument("--model", default="google.gemma-3-4b-it")
    parser.add_argument("--task_type", choices=sorted(TASK_TYPES), default="coding")
    parser.add_argument("--task_start", type=int, default=1)
    parser.add_argument("--task_end", type=int, default=10)
    parser.add_argument("--timeout_sec", type=int, default=1200)
    parser.add_argument(
        "--config_dir",
        default="marble/configs/coding_configs",
        help="Directory containing experiment configs.",
    )
    parser.add_argument(
        "--config_pattern",
        default="config_{task_id}.yaml",
        help="Config filename pattern inside config_dir. Supports {task_id}.",
    )
    parser.add_argument(
        "--result_root",
        default="marble/result/lifelong",
        help="Output directory for run/eval artifacts.",
    )
    parser.add_argument(
        "--modes",
        default="baseline,llmamem",
        help="Comma-separated modes. Supported: baseline,llmamem",
    )
    parser.add_argument("--consolidation_interval", type=int, default=2)
    parser.add_argument(
        "--num_agents",
        type=int,
        default=0,
        help="If > 0, keep only first N agents and prune relationships accordingly.",
    )
    parser.add_argument(
        "--llma_topology",
        choices=["local", "shared", "hybrid"],
        default="local",
        help="LLMA-Mem topology to use in llmamem mode.",
    )
    parser.add_argument(
        "--use_poetry",
        action="store_true",
        default=False,
        help="Run tasks via `poetry run python` (default: disabled).",
    )
    parser.add_argument(
        "--db_eval_model",
        default="gpt-4o-mini",
        help="LLM model used for DB root-cause extraction (batch_eval-compatible).",
    )
    return parser.parse_args()


def _mean_non_negative(values: Any) -> float:
    if not isinstance(values, list):
        return -1.0
    valid = [float(v) for v in values if isinstance(v, (int, float)) and v >= 0]
    if not valid:
        return -1.0
    return sum(valid) / len(valid)


def _collect_numeric_values(obj: Any) -> List[float]:
    vals: List[float] = []
    if isinstance(obj, (int, float)):
        vals.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            vals.extend(_collect_numeric_values(v))
    elif isinstance(obj, list):
        for v in obj:
            vals.extend(_collect_numeric_values(v))
    return vals


def _score_from_code_quality(code_quality: Dict[str, Any]) -> float:
    keys = ["instruction_following", "executability", "consistency", "quality"]
    vals: List[float] = []
    for key in keys:
        val = code_quality.get(key)
        if isinstance(val, (int, float)):
            vals.append(float(val))
    if not vals:
        return -1.0
    return (sum(vals) / len(vals)) * 20.0


def _score_from_task_evaluation_generic(task_eval: Any) -> float:
    if isinstance(task_eval, (int, float)):
        return float(task_eval) * 20.0
    nums = _collect_numeric_values(task_eval)
    if nums:
        return (sum(nums) / len(nums)) * 20.0
    return -1.0


def extract_task_scores_coding(summary: Dict[str, Any]) -> Tuple[float, float, float]:
    ts = -1.0
    code_quality = summary.get("code_quality")
    if isinstance(code_quality, dict):
        ts = _score_from_code_quality(code_quality)
    else:
        ts = _score_from_task_evaluation_generic(summary.get("task_evaluation"))

    planning = _mean_non_negative(summary.get("planning_scores", []))
    communication = _mean_non_negative(summary.get("communication_scores", []))
    cs = -1.0
    if planning >= 0 and communication >= 0:
        cs = ((planning + communication) / 2.0) * 20.0
    elif planning >= 0:
        cs = planning * 20.0
    elif communication >= 0:
        cs = communication * 20.0

    if ts >= 0 and cs >= 0:
        j = (ts + cs) / 2.0
    elif ts >= 0:
        j = ts
    elif cs >= 0:
        j = cs
    else:
        j = -1.0
    return ts, cs, j


def extract_task_scores_research(summary: Dict[str, Any]) -> Tuple[float, float, float]:
    # task_evaluation is usually a dict, e.g. innovation/safety/feasibility
    ts = _score_from_task_evaluation_generic(summary.get("task_evaluation"))
    planning = _mean_non_negative(summary.get("planning_scores", []))
    communication = _mean_non_negative(summary.get("communication_scores", []))
    cs = -1.0
    if planning >= 0 and communication >= 0:
        cs = ((planning + communication) / 2.0) * 20.0
    elif planning >= 0:
        cs = planning * 20.0
    elif communication >= 0:
        cs = communication * 20.0
    j = (ts + cs) / 2.0 if (ts >= 0 and cs >= 0) else ts
    return ts, cs, j if j >= 0 else -1.0


def _db_extract_predicted_labels(predicted_text: str, db_eval_model: str) -> List[str]:
    try:
        import litellm
        from litellm.utils import trim_messages
    except Exception:
        return []

    prompt = (
        f"{predicted_text}\n\n"
        "From the text above, please identify the two predicted root causes of the issue.\n\n"
        "Please print each of them in the form they appear in two separate lines."
        "I have a very rudimentary system, so if it is not in the exact form, it will crash."
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        completion = litellm.completion(
            model=db_eval_model,
            messages=trim_messages(
                messages, model=db_eval_model, max_tokens=int(16384 * 0.6)
            ),
            max_tokens=512,
            temperature=0.0,
        )
        response = completion.choices[0].message.content.strip()
        return [label.strip() for label in response.split("\n") if label.strip()]
    except Exception:
        return []


def extract_task_scores_db(
    summary: Dict[str, Any], db_eval_model: str
) -> Tuple[float, float, float]:
    """
    Batch-eval compatible DB task score:
    - Use LLM to extract two predicted root causes from task_evaluation.predicted
    - Compare against task_evaluation.root_cause
    - TS = accuracy * 100
    """
    task_eval = summary.get("task_evaluation")
    if not isinstance(task_eval, dict):
        return -1.0, -1.0, -1.0

    gold_labels = task_eval.get("root_cause", [])
    predicted = task_eval.get("predicted", "")
    if not isinstance(gold_labels, list) or not isinstance(predicted, str):
        return -1.0, -1.0, -1.0

    predicted_labels = _db_extract_predicted_labels(predicted, db_eval_model)
    if not predicted_labels:
        return -1.0, -1.0, -1.0

    match_count = sum(g in predicted_labels for g in gold_labels)
    ts = (match_count / len(gold_labels) * 100.0) if gold_labels else 0.0
    # Keep DB output aligned with old script semantics: report task score only.
    return ts, -1.0, ts


def extract_task_scores_minecraft(summary: Dict[str, Any]) -> Tuple[float, float, float]:
    # engine writes task_evaluation = block_hit_rate*5, so TS = *20
    ts = _score_from_task_evaluation_generic(summary.get("task_evaluation"))
    return ts, -1.0, ts


def extract_task_scores(
    summary: Dict[str, Any], task_type: str, db_eval_model: str
) -> Tuple[float, float, float]:
    if task_type == "coding":
        return extract_task_scores_coding(summary)
    if task_type == "research":
        return extract_task_scores_research(summary)
    if task_type == "db":
        return extract_task_scores_db(summary, db_eval_model)
    if task_type == "minecraft":
        return extract_task_scores_minecraft(summary)
    return extract_task_scores_coding(summary)


def compute_ap_aip(js: List[float]) -> Tuple[List[float], float]:
    valid_js: List[float] = []
    ap_curve: List[float] = []
    running_sum = 0.0
    for j in js:
        if j < 0:
            ap_curve.append(ap_curve[-1] if ap_curve else -1.0)
            continue
        valid_js.append(j)
        running_sum += j
        ap_curve.append(running_sum / len(valid_js))

    valid_aps = [ap for ap in ap_curve if ap >= 0]
    aip = (sum(valid_aps) / len(valid_aps)) if valid_aps else -1.0
    return ap_curve, aip


def maybe_plot(
    out_png: Path,
    task_ids: List[int],
    mode_to_values: Dict[str, List[float]],
    y_label: str,
    title: str,
) -> Tuple[str, str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(9, 5))
        has_series = False
        for mode, values in mode_to_values.items():
            points = [(t, v) for t, v in zip(task_ids, values) if v >= 0]
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.plot(xs, ys, marker="o", label=mode)
            has_series = True

        plt.xlabel("Task Index")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(alpha=0.3)
        if has_series:
            plt.legend()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        return str(out_png), ""
    except Exception as exc:
        return "", f"plot_skipped: {exc}"


def load_last_jsonl_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except Exception:
        return {}


def load_latest_minecraft_block_hit_rate(repo_root: Path) -> float:
    score_path = repo_root / "data" / "score.json"
    if not score_path.exists():
        return -1.0
    try:
        payload = json.loads(score_path.read_text(encoding="utf-8"))
    except Exception:
        return -1.0
    if not isinstance(payload, list) or not payload:
        return -1.0
    last = payload[-1]
    if not isinstance(last, dict):
        return -1.0
    val = last.get("block_hit_rate")
    if not isinstance(val, (int, float)):
        return -1.0
    return float(val) * 100.0


def run_one_task(
    repo_root: Path,
    marble_dir: Path,
    src_config: Path,
    tmp_config: Path,
    output_path_rel_to_marble: str,
    model: str,
    llma_mem_enabled: bool,
    consolidation_interval: int,
    timeout_sec: int,
    use_poetry: bool,
    num_agents: int,
    llma_topology: str,
) -> subprocess.CompletedProcess[str]:
    data = yaml.safe_load(src_config.read_text(encoding="utf-8"))
    data["llm"] = model
    metrics = data.setdefault("metrics", {})
    metrics["evaluate_llm"] = model

    if num_agents > 0 and isinstance(data.get("agents"), list):
        kept_agents = data["agents"][:num_agents]
        data["agents"] = kept_agents
        kept_ids = {
            a.get("agent_id")
            for a in kept_agents
            if isinstance(a, dict) and a.get("agent_id")
        }
        if isinstance(data.get("relationships"), list):
            data["relationships"] = [
                rel
                for rel in data["relationships"]
                if isinstance(rel, list)
                and len(rel) >= 2
                and rel[0] in kept_ids
                and rel[1] in kept_ids
            ]

    memory = data.setdefault("memory", {})
    llma_mem = memory.setdefault("llma_mem", {})
    llma_mem["enabled"] = bool(llma_mem_enabled)
    if llma_mem_enabled:
        llma_mem["topology"] = llma_topology
        llma_mem["consolidation_interval"] = consolidation_interval
        llma_mem.setdefault("auto_embedding", True)
        llma_mem.setdefault("embedding_model", "amazon.titan-embed-text-v2:0")

    output = data.setdefault("output", {})
    output["format"] = "jsonl"
    output["file_path"] = output_path_rel_to_marble

    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    workspace = marble_dir / "workspace"
    for name in ("solution.py", "advices.json", "error.json"):
        p = workspace / name
        if p.exists():
            p.unlink()

    run_env = os.environ.copy()
    existing_pythonpath = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = (
        f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    )

    cmd = [sys.executable, "main.py", "--config_path", str(tmp_config)]
    if use_poetry:
        cmd = ["poetry", "run", "python", "main.py", "--config_path", str(tmp_config)]

    return subprocess.run(
        cmd,
        cwd=str(marble_dir),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
        env=run_env,
    )


def main() -> int:
    args = parse_args()
    use_poetry = args.use_poetry
    repo_root = Path(__file__).resolve().parents[2]
    marble_dir = repo_root / "marble"
    config_dir = repo_root / args.config_dir
    result_root = repo_root / args.result_root
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    supported = {"baseline", "llmamem"}
    for mode in modes:
        if mode not in supported:
            raise ValueError(f"Unsupported mode: {mode}. Supported: {sorted(supported)}")

    result_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = result_root / "_tmp_configs"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    global_rows: List[Dict[str, Any]] = []
    mode_to_ap: Dict[str, List[float]] = {}
    task_ids = list(range(args.task_start, args.task_end + 1))

    for mode in modes:
        llma_mem_enabled = mode == "llmamem"
        mode_dir = result_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        js: List[float] = []
        mode_rows: List[Dict[str, Any]] = []

        for task_id in task_ids:
            src_config = config_dir / args.config_pattern.format(task_id=task_id)
            if not src_config.exists():
                mode_rows.append(
                    {
                        "mode": mode,
                        "task_id": task_id,
                        "status": "missing_config",
                        "ts": -1.0,
                        "cs": -1.0,
                        "j": -1.0,
                        "returncode": -1,
                    }
                )
                js.append(-1.0)
                continue

            output_abs = result_root / mode / f"task_{task_id}.jsonl"
            output_abs.parent.mkdir(parents=True, exist_ok=True)
            if output_abs.exists():
                output_abs.unlink()

            tmp_config = tmp_dir / f"{mode}_config_{task_id}.yaml"
            try:
                proc = run_one_task(
                    repo_root=repo_root,
                    marble_dir=marble_dir,
                    src_config=src_config,
                    tmp_config=tmp_config,
                    output_path_rel_to_marble=str(output_abs),
                    model=args.model,
                    llma_mem_enabled=llma_mem_enabled,
                    consolidation_interval=args.consolidation_interval,
                    timeout_sec=args.timeout_sec,
                    use_poetry=use_poetry,
                    num_agents=args.num_agents,
                    llma_topology=args.llma_topology,
                )
                summary = load_last_jsonl_record(output_abs)
                ts, cs, j = extract_task_scores(
                    summary, args.task_type, args.db_eval_model
                )
                status = "ok" if proc.returncode == 0 else "run_failed"
                row = {
                    "mode": mode,
                    "task_id": task_id,
                    "status": status,
                    "ts": round(ts, 4),
                    "cs": round(cs, 4),
                    "j": round(j, 4),
                    "returncode": proc.returncode,
                }
                if proc.returncode != 0:
                    row["stderr_tail"] = proc.stderr[-500:]
                mode_rows.append(row)
                js.append(j)
            except subprocess.TimeoutExpired:
                ts = cs = j = -1.0
                status = "timeout"
                if args.task_type == "minecraft":
                    ts = load_latest_minecraft_block_hit_rate(repo_root)
                    if ts >= 0:
                        j = ts
                        status = "timeout_scored"
                mode_rows.append(
                    {
                        "mode": mode,
                        "task_id": task_id,
                        "status": status,
                        "ts": round(ts, 4) if ts >= 0 else -1.0,
                        "cs": round(cs, 4) if cs >= 0 else -1.0,
                        "j": round(j, 4) if j >= 0 else -1.0,
                        "returncode": -2,
                    }
                )
                js.append(j)

        ap_curve, aip = compute_ap_aip(js)
        mode_to_ap[mode] = ap_curve

        for row, ap in zip(mode_rows, ap_curve):
            row["ap_t"] = round(ap, 4) if ap >= 0 else -1.0
            row["aip"] = round(aip, 4) if aip >= 0 else -1.0
        global_rows.extend(mode_rows)

        summary_path = mode_dir / "lifelong_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "task_type": args.task_type,
                    "model": args.model,
                    "num_agents": args.num_agents if args.num_agents > 0 else "default",
                    "task_range": [args.task_start, args.task_end],
                    "aip": round(aip, 6) if aip >= 0 else -1.0,
                    "ap_curve": [round(x, 6) if x >= 0 else -1.0 for x in ap_curve],
                    "rows": mode_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    csv_path = result_root / "lifelong_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "mode",
            "task_id",
            "status",
            "ts",
            "cs",
            "j",
            "ap_t",
            "aip",
            "returncode",
            "stderr_tail",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in global_rows:
            writer.writerow(row)

    mode_to_j: Dict[str, List[float]] = {}
    for mode in modes:
        rows = [r for r in global_rows if r["mode"] == mode]
        rows = sorted(rows, key=lambda r: int(r["task_id"]))
        mode_to_j[mode] = [
            float(r["j"]) if isinstance(r.get("j"), (int, float)) else -1.0 for r in rows
        ]

    plot_path, plot_error = maybe_plot(
        result_root / "lifelong_ap_curve.png",
        task_ids,
        mode_to_ap,
        "AP_t",
        f"Lifelong AP Curve ({args.task_type})",
    )
    j_plot_path, j_plot_error = maybe_plot(
        result_root / "lifelong_j_curve.png",
        task_ids,
        mode_to_j,
        "J",
        f"Lifelong J Curve ({args.task_type})",
    )

    final_report = {
        "task_type": args.task_type,
        "model": args.model,
        "num_agents": args.num_agents if args.num_agents > 0 else "default",
        "task_range": [args.task_start, args.task_end],
        "modes": modes,
        "csv": str(csv_path),
        "plot_ap": plot_path,
        "plot_ap_error": plot_error,
        "plot_j": j_plot_path,
        "plot_j_error": j_plot_error,
    }
    (result_root / "report.json").write_text(
        json.dumps(final_report, indent=2), encoding="utf-8"
    )

    print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
