#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREP_SCRIPT="${ROOT_DIR}/scripts/research/utils/prepare_amem_research_config.py"
MAIN_SCRIPT="${ROOT_DIR}/marble/main.py"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESEARCH_JSONL="${RESEARCH_JSONL:-${ROOT_DIR}/multiagentbench/research/research_main.jsonl}"

MODEL_NAME="${MODEL_NAME:-bedrock/converse/qwen.qwen3-32b-v1:0}"
EVALUATE_LLM="${EVALUATE_LLM:-bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bedrock/amazon.titan-embed-text-v2:0}"
TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-100}"
TASK_IDS="${TASK_IDS:-}"
RUN_TAG="${RUN_TAG:-amem_research_$(date -u +%Y%m%dT%H%M%SZ)}"
TASK_PARALLELISM="${TASK_PARALLELISM:-4}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TEAM_SIZE="${TEAM_SIZE:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-3}"
MAX_MEMORY_CONTEXT="${MAX_MEMORY_CONTEXT:-5}"
LINK_THRESHOLD="${LINK_THRESHOLD:-0.72}"

if [[ "${TEAM_SIZE}" != "3" ]]; then
    echo "This runner currently enforces TEAM_SIZE=3." >&2
    exit 1
fi

if [[ "${MAX_ITERATIONS}" != "3" ]]; then
    echo "This runner currently enforces MAX_ITERATIONS=3." >&2
    exit 1
fi

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${ROOT_DIR}/marble/configs/test_config_research_generated_amem/${safe_model_name}/${RUN_TAG}}"
SOURCE_CONFIG_DIR="${SOURCE_CONFIG_DIR:-${ROOT_DIR}/marble/configs/test_config_research_runtime.${safe_model_name}.${RUN_TAG}}"
FINAL_OUTPUT_REL="result/research/${safe_model_name}/amem/${RUN_TAG}.jsonl"
FINAL_OUTPUT_ABS="${ROOT_DIR}/marble/${FINAL_OUTPUT_REL}"
TASK_OUTPUT_DIR_REL="result/research/${safe_model_name}/amem/${RUN_TAG}/tasks"
TASK_OUTPUT_DIR_ABS="${ROOT_DIR}/marble/${TASK_OUTPUT_DIR_REL}"

cleanup() {
    if [[ -n "${TEMP_SOURCE_CONFIG_DIR:-}" && -d "${TEMP_SOURCE_CONFIG_DIR}" ]]; then
        rm -rf "${TEMP_SOURCE_CONFIG_DIR}"
    fi
}
trap cleanup EXIT

load_bedrock_token_from_env() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "Missing env file: ${ENV_FILE}" >&2
        return 1
    fi

    local token_line token
    token_line="$(grep -m1 '^AWS_BEARER_TOKEN_BEDROCK=' "${ENV_FILE}" || true)"
    token="${token_line#AWS_BEARER_TOKEN_BEDROCK=}"
    token="${token%\"}"
    token="${token#\"}"
    token="${token%\'}"
    token="${token#\'}"
    token="$(printf '%s' "${token}" | tr -d '\r\n')"

    if [[ -z "${token}" ]]; then
        echo "AWS_BEARER_TOKEN_BEDROCK is empty or missing in ${ENV_FILE}" >&2
        return 1
    fi

    export AWS_BEARER_TOKEN_BEDROCK="${token}"
}

task_sequence() {
    if [[ -n "${TASK_IDS}" ]]; then
        echo "${TASK_IDS}" | tr ',' '\n' | sed '/^\s*$/d'
    else
        seq "${TASK_START}" "${TASK_END}"
    fi
}

prepare_source_configs() {
    if [[ -d "${SOURCE_CONFIG_DIR}" ]] && find "${SOURCE_CONFIG_DIR}" -maxdepth 1 -type f -name 'task_*.yaml' | grep -q .; then
        return 0
    fi

    if [[ ! -f "${RESEARCH_JSONL}" ]]; then
        echo "Research benchmark JSONL does not exist: ${RESEARCH_JSONL}" >&2
        exit 1
    fi

    TEMP_SOURCE_CONFIG_DIR="${SOURCE_CONFIG_DIR}"
    mkdir -p "${SOURCE_CONFIG_DIR}"

    "${PYTHON_BIN}" - "${RESEARCH_JSONL}" "${SOURCE_CONFIG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from ruamel.yaml import YAML

input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

defaults = {
    "coordinate_mode": "graph",
    "environment": {
        "max_iterations": 3,
        "name": "Research Collaboration Environment",
        "type": "Research",
    },
    "llm": "gpt-3.5-turbo",
    "memory": {"type": "BaseMemory"},
    "metrics": {"evaluate_llm": "gpt-4o"},
    "output": {"file_path": "result/discussion_output.jsonl"},
}


def fill_defaults(data):
    if data.get("coordinate_mode", "") == "":
        data["coordinate_mode"] = defaults["coordinate_mode"]
    if data.get("llm", "") == "":
        data["llm"] = defaults["llm"]

    for key in ["environment", "memory", "output"]:
        if isinstance(data.get(key), dict):
            for sub_key, default_val in defaults[key].items():
                if data[key].get(sub_key, "") == "":
                    data[key][sub_key] = default_val
        else:
            data[key] = dict(defaults[key])

    if isinstance(data.get("metrics"), dict):
        if data["metrics"].get("evaluate_llm", "") == "":
            data["metrics"]["evaluate_llm"] = defaults["metrics"]["evaluate_llm"]
    else:
        data["metrics"] = dict(defaults["metrics"])

    return data


yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

output_dir.mkdir(parents=True, exist_ok=True)

with input_path.open("r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        data = fill_defaults(json.loads(line))
        task_id = data.get("task_id")
        if task_id is None:
            continue
        output_path = output_dir / f"task_{task_id}.yaml"
        with output_path.open("w", encoding="utf-8") as outfile:
            yaml.dump(data, outfile)
PY
}

recover_task_outputs_from_merged() {
    if find "${TASK_OUTPUT_DIR_ABS}" -maxdepth 1 -type f -name 'task_*.jsonl' | grep -q .; then
        return 0
    fi

    if [[ ! -f "${FINAL_OUTPUT_ABS}" ]]; then
        return 0
    fi

    "${PYTHON_BIN}" - "${SOURCE_CONFIG_DIR}" "${FINAL_OUTPUT_ABS}" "${TASK_OUTPUT_DIR_ABS}" <<'PY'
import json
import os
import re
import sys

from ruamel.yaml import YAML

source_dir, merged_path, tasks_dir = sys.argv[1:4]
yaml = YAML(typ="safe")

def normalize_task(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

task_to_id = {}
for name in sorted(os.listdir(source_dir)):
    match = re.fullmatch(r"task_(\d+)\.yaml", name)
    if not match:
        continue
    task_id = int(match.group(1))
    path = os.path.join(source_dir, name)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f)
    task_text = normalize_task((cfg or {}).get("task", {}).get("content", ""))
    if task_text:
        task_to_id[task_text] = task_id

recovered = 0
with open(merged_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        obj = json.loads(line)
        task_id = obj.get("task_id")
        if task_id is None:
            task_id = task_to_id.get(normalize_task(obj.get("task", "")))
        if task_id is None:
            continue
        shard_path = os.path.join(tasks_dir, f"task_{task_id}.jsonl")
        with open(shard_path, "w", encoding="utf-8") as shard:
            shard.write(line + "\n")
        recovered += 1

print(f"[amem-research] recovered {recovered} task shard(s) from merged output", file=sys.stderr)
PY
}

pending_task_sequence() {
    while IFS= read -r task_id; do
        if [[ "${SKIP_COMPLETED}" == "1" && -f "${TASK_OUTPUT_DIR_ABS}/task_${task_id}.jsonl" ]]; then
            echo "[amem-research] Skipping completed task ${task_id}" >&2
            continue
        fi
        echo "${task_id}"
    done < <(task_sequence)
}

run_task() {
    local task_id="$1"
    local source_config="${SOURCE_CONFIG_DIR}/task_${task_id}.yaml"
    local target_config="${GENERATED_CONFIG_DIR}/task_${task_id}.yaml"
    local config_rel="./configs/test_config_research_generated_amem/${safe_model_name}/${RUN_TAG}/task_${task_id}.yaml"
    local output_rel="${TASK_OUTPUT_DIR_REL}/task_${task_id}.jsonl"
    local output_abs="${ROOT_DIR}/marble/${output_rel}"

    if [[ ! -f "${source_config}" ]]; then
        echo "Source config missing for task ${task_id}: ${source_config}" >&2
        return 1
    fi

    mkdir -p "$(dirname "${output_abs}")" "${GENERATED_CONFIG_DIR}"
    rm -f "${output_abs}"

    "${PYTHON_BIN}" "${PREP_SCRIPT}" \
        --source-config "${source_config}" \
        --target-config "${target_config}" \
        --llm-model "${MODEL_NAME}" \
        --evaluate-llm "${EVALUATE_LLM}" \
        --agent-count "${TEAM_SIZE}" \
        --output-path "${output_rel}" \
        --embedding-model "${EMBEDDING_MODEL}" \
        --retrieval-top-k "${RETRIEVAL_TOP_K}" \
        --max-memory-context "${MAX_MEMORY_CONTEXT}" \
        --link-threshold "${LINK_THRESHOLD}" \
        --max-iterations "${MAX_ITERATIONS}"

    load_bedrock_token_from_env

    (
        cd "${ROOT_DIR}/marble"
        PYTHONPATH=.. "${PYTHON_BIN}" "${MAIN_SCRIPT}" --config_path "${config_rel}"
    )

    "${PYTHON_BIN}" - "${output_abs}" "${task_id}" <<'PY'
import json
import sys

path = sys.argv[1]
task_id = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f if line.strip()]
with open(path, "w", encoding="utf-8") as f:
    for line in lines:
        obj = json.loads(line)
        obj["task_id"] = task_id
        f.write(json.dumps(obj) + "\n")
PY

    echo "[amem-research] Task ${task_id} completed."
}

merge_task_outputs() {
    mkdir -p "$(dirname "${FINAL_OUTPUT_ABS}")"
    rm -f "${FINAL_OUTPUT_ABS}"

    local task_output
    local task_file
    for task_output in "${TASK_OUTPUT_DIR_ABS}"/task_*.jsonl; do
        if [[ ! -e "${task_output}" ]]; then
            continue
        fi
        task_file="$(basename "${task_output}")"
        printf '%s\n' "${task_file}"
    done | sort -V | while IFS= read -r task_file; do
        task_output="${TASK_OUTPUT_DIR_ABS}/${task_file}"
        cat "${task_output}" >> "${FINAL_OUTPUT_ABS}"
    done

    echo "[amem-research] Aggregated task outputs into ${FINAL_OUTPUT_ABS}"
}

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file does not exist: ${ENV_FILE}" >&2
    exit 1
fi

prepare_source_configs
mkdir -p "$(dirname "${FINAL_OUTPUT_ABS}")" "${TASK_OUTPUT_DIR_ABS}"
recover_task_outputs_from_merged

echo "Starting A-Mem research run"
echo "Model: ${MODEL_NAME}"
echo "Run tag: ${RUN_TAG}"
echo "Source configs: ${SOURCE_CONFIG_DIR}"
echo "Generated configs: ${GENERATED_CONFIG_DIR}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Evaluation model: ${EVALUATE_LLM}"
echo "Task parallelism: ${TASK_PARALLELISM}"
echo "Skip completed: ${SKIP_COMPLETED}"
echo "Team size: ${TEAM_SIZE}"
echo "Max iterations: ${MAX_ITERATIONS}"
if [[ -n "${TASK_IDS}" ]]; then
    echo "Task IDs: ${TASK_IDS}"
else
    echo "Task range: ${TASK_START}-${TASK_END}"
fi

failures=0
active_jobs=0
while IFS= read -r task_id; do
    run_task "${task_id}" &
    active_jobs=$((active_jobs + 1))

    if (( active_jobs >= TASK_PARALLELISM )); then
        if ! wait -n; then
            failures=$((failures + 1))
        fi
        active_jobs=$((active_jobs - 1))
    fi
done < <(pending_task_sequence)

while (( active_jobs > 0 )); do
    if ! wait -n; then
        failures=$((failures + 1))
    fi
    active_jobs=$((active_jobs - 1))
done

merge_task_outputs

if (( failures > 0 )); then
    echo "A-Mem research run completed with ${failures} task failure(s)." >&2
    exit 1
fi

echo "A-Mem research run completed successfully."
