"""
Analyze a MARBLE development_output.jsonl file and report per-task and
aggregate averages for:

  TS  = avg(executability, instruction_following, consistency) * 50
  CS  = avg(planning_scores + communication_scores, skip -1) * 20  AP_t = (1/t) * sum(TS_1 ... TS_t)   [running average of TS up to task t]
  AIP  = (1/T) * sum(AP_1 ... AP_T)   [mean of all running averages]  Input tokens, Output tokens (from token_usage)

  Failed tasks (all iterations have empty task_results) are listed but
  excluded from all averages.

Usage:
    python scripts/analyze_output.py <path_to_jsonl>
    python scripts/analyze_output.py marble/result/nomem/development_output_llama8b.jsonl
"""

import json
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_avg(values: List[Any]) -> Optional[float]:
    """Return the mean of numeric values, excluding -1. None if nothing left."""
    filtered = [v for v in values if isinstance(v, (int, float)) and v != -1]
    return sum(filtered) / len(filtered) if filtered else None


def _get_dim_scores(record: Dict[str, Any]) -> List[int]:
    """Return the list of dimension score values from task_evaluation."""
    te = record.get("task_evaluation")
    if not isinstance(te, dict):
        return []
    keys = ["executability", "instruction_following", "consistency", "quality"]
    return [te[k] for k in keys if k in te and isinstance(te[k], (int, float))]


def is_all_ones(record: Dict[str, Any]) -> bool:
    """True when every dimension score is 1 (complete failure)."""
    vals = _get_dim_scores(record)
    return bool(vals) and all(v == 1 for v in vals)


def is_failed(record: Dict[str, Any]) -> bool:
    """
    A task is considered failed if:
      - every iteration has an empty task_results list, OR
      - all dimension scores are 1.
    """
    iterations = record.get("iterations", [])
    if not iterations:
        return True
    if all(not iter_data.get("task_results") for iter_data in iterations):
        return True
    if is_all_ones(record):
        return True
    return False


def compute_ts(record: Dict[str, Any]) -> Optional[float]:
    """
    TS = avg(executability, instruction_following, consistency, quality) * 20.
    Returns None if task_evaluation is missing or all three fields are absent.
    """
    te = record.get("task_evaluation")
    if not isinstance(te, dict):
        return None
    keys = ["executability", "instruction_following", "consistency", "quality"]
    values = [te[k] for k in keys if k in te and isinstance(te[k], (int, float))]
    if not values:
        return None
    return (sum(values) / len(values)) * 20


def compute_cs(record: Dict[str, Any]) -> Optional[float]:
    """
    CS = avg(planning_scores + communication_scores, skip -1) * 20.
    Returns None if no valid scores exist.
    """
    all_scores = record.get("planning_scores", []) + record.get("communication_scores", [])
    avg = safe_avg(all_scores)
    return avg * 20 if avg is not None else None


def compute_tokens(record: Dict[str, Any]):
    """Return (prompt_tokens, completion_tokens) or (0, 0) if missing."""
    tu = record.get("token_usage", {})
    if not isinstance(tu, dict):
        return 0, 0
    return tu.get("prompt_tokens", 0) or 0, tu.get("completion_tokens", 0) or 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(path: str) -> None:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: JSON parse error — {e}")

    if not records:
        print("No valid records found.")
        return

    print(f"{'='*88}")
    print(f"  File : {path}")
    print(f"  Tasks: {len(records)}")
    print(f"{'='*88}")
    header = f"{'Task':>5}  {'Status':>6}  {'TS':>8}  {'CS':>8}  {'AP_t':>8}  {'Input tok':>12}  {'Output tok':>12}"
    print(header)
    print("-" * len(header))

    per_task_ts: List[float] = []
    per_task_cs: List[float] = []
    per_task_input: List[int] = []
    per_task_output: List[int] = []
    per_task_ap: List[float] = []   # AP_t for each passed task in order
    failed_count = 0

    for i, rec in enumerate(records, 1):
        failed = is_failed(rec)
        status = "FAIL" if failed else "OK"

        ts = compute_ts(rec)
        cs = compute_cs(rec)
        inp, out = compute_tokens(rec)

        if failed:
            ts_str = cs_str = ap_str = "SKIP"
            failed_count += 1
        else:
            ts_str = f"{ts:.2f}" if ts is not None else "N/A"
            cs_str = f"{cs:.2f}" if cs is not None else "N/A"

            if ts is not None:
                per_task_ts.append(ts)
            if cs is not None:
                per_task_cs.append(cs)
            per_task_input.append(inp)
            per_task_output.append(out)

            # AP_t = running average of TS over all passed tasks seen so far
            if per_task_ts:
                ap_t = (sum(per_task_ts) / len(per_task_ts))
                per_task_ap.append(ap_t)
                ap_str = f"{ap_t:.2f}"
            else:
                ap_str = "N/A"

        print(f"{i:>5}  {status:>6}  {ts_str:>8}  {cs_str:>8}  {ap_str:>8}  {inp:>12,}  {out:>12,}")

    print("-" * len(header))

    avg_ts  = sum(per_task_ts) / len(per_task_ts) if per_task_ts else None
    avg_cs  = sum(per_task_cs) / len(per_task_cs) if per_task_cs else None
    avg_inp = sum(per_task_input) / len(per_task_input) if per_task_input else 0
    avg_out = sum(per_task_output) / len(per_task_output) if per_task_output else 0
    aip     = sum(per_task_ap) / len(per_task_ap) if per_task_ap else None

    avg_ts_str = f"{avg_ts:.4f}" if avg_ts is not None else "N/A"
    avg_cs_str = f"{avg_cs:.4f}" if avg_cs is not None else "N/A"
    aip_str    = f"{aip:.4f}"    if aip    is not None else "N/A"

    passed = len(records) - failed_count
    print(f"{'AVG':>5}  {'':>6}  {avg_ts_str:>8}  {avg_cs_str:>8}  {aip_str:>8}  {avg_inp:>12.1f}  {avg_out:>12.1f}")
    print(f"{'='*88}")
    print()
    print("Summary")
    print(f"  TS  = avg(executability + instruction_following + consistency + quality) × 20")
    print(f"  CS  = avg(planning_scores + communication_scores, skip -1) × 20")
    print(f"  AP_t = (1/t) * sum(TS_1 .. TS_t)   [running avg of TS across first t passed tasks]")
    print(f"  AIP  = (1/T) * sum(AP_1 .. AP_T)   [mean of all AP_t values]")
    print(f"  Tasks with all 1s are marked FAIL")
    print(f"  Avg TS                         : {avg_ts_str}")
    print(f"  Avg CS                         : {avg_cs_str}")
    print(f"  AIP                            : {aip_str}")
    print(f"  Avg input tokens  (per task)   : {avg_inp:.1f}")
    print(f"  Avg output tokens (per task)   : {avg_out:.1f}")
    print(f"  Passed / Total                 : {passed} / {len(records)}")
    print(f"  Failed (all 1s or no output)   : {failed_count}")
    print(f"  Tasks with valid TS            : {len(per_task_ts)} / {passed}")
    print(f"  Tasks with valid CS            : {len(per_task_cs)} / {passed}")
    print(f"{'='*88}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_output.py <path_to_jsonl>")
        sys.exit(1)
    analyze(sys.argv[1])
