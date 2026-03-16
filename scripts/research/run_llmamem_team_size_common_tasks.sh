#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_simulation.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME="${MODEL_NAME:-bedrock/converse/qwen.qwen3-32b-v1:0}"
EVALUATE_MODEL="${EVALUATE_MODEL:-bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
RESEARCH_JSONL="${RESEARCH_JSONL:-${ROOT_DIR}/multiagentbench/research/research_main.jsonl}"
APPEND_RESULTS="${APPEND_RESULTS:-1}"
TEAM_SIZES="${TEAM_SIZES:-1,3,5,7}"
CONSOLIDATION_INTERVAL="${CONSOLIDATION_INTERVAL:-3}"
TASK_PARALLELISM="${TASK_PARALLELISM:-1}"
TASK_IDS="${TASK_IDS:-10,11,15,18,20,26,29,30,51,56,59,64,77,78,92,100}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-llmamem_common16}"

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"
TEMP_CONFIG_DIR="$(mktemp -d "${ROOT_DIR}/marble/configs/test_config_research_common.${safe_model_name}.XXXXXX")"

cleanup() {
    if [[ -d "${TEMP_CONFIG_DIR}" ]]; then
        rm -rf "${TEMP_CONFIG_DIR}"
    fi
}
trap cleanup EXIT

if [[ ! -x "${BASE_RUNNER}" ]]; then
    echo "Base runner is not executable: ${BASE_RUNNER}" >&2
    exit 1
fi

if [[ ! -f "${RESEARCH_JSONL}" ]]; then
    echo "Research benchmark JSONL does not exist: ${RESEARCH_JSONL}" >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file does not exist: ${ENV_FILE}" >&2
    exit 1
fi

echo "Preparing common-task configs"
echo "Model: ${MODEL_NAME}"
echo "Team sizes: ${TEAM_SIZES}"
echo "Task IDs: ${TASK_IDS}"
echo "Consolidation interval: ${CONSOLIDATION_INTERVAL}"
echo "Task parallelism: ${TASK_PARALLELISM}"
echo "Temporary config dir: ${TEMP_CONFIG_DIR}"

"${PYTHON_BIN}" - "${RESEARCH_JSONL}" "${TEMP_CONFIG_DIR}" "${TASK_IDS}" <<'PY'
import json
import sys
from pathlib import Path

from ruamel.yaml import YAML

input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
task_ids = [int(x.strip()) for x in sys.argv[3].split(",") if x.strip()]
task_id_set = set(task_ids)

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
    data["coordinate_mode"] = "graph"
    return data


yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

output_dir.mkdir(parents=True, exist_ok=True)
seen = set()

with input_path.open("r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        data = fill_defaults(json.loads(line))
        task_id = data.get("task_id")
        if task_id in task_id_set:
            output_path = output_dir / f"task_{task_id}.yaml"
            with output_path.open("w", encoding="utf-8") as outfile:
                yaml.dump(data, outfile)
            seen.add(task_id)

missing = sorted(task_id_set - seen)
if missing:
    raise SystemExit(f"Missing task IDs in benchmark: {missing}")
PY

IFS=',' read -r -a TEAM_SIZE_ARRAY <<< "${TEAM_SIZES}"

for team_size in "${TEAM_SIZE_ARRAY[@]}"; do
    team_size="$(echo "${team_size}" | xargs)"
    if [[ -z "${team_size}" ]]; then
        continue
    fi
    if ! [[ "${team_size}" =~ ^[0-9]+$ ]]; then
        echo "Invalid team size: ${team_size}" >&2
        exit 1
    fi

    echo "Running LLMAMem team-size condition: ${team_size}"

    WORKERS=llmamem \
    MODEL_NAME="${MODEL_NAME}" \
    EVALUATE_MODEL="${EVALUATE_MODEL}" \
    AGENT_COUNT="${team_size}" \
    ENV_FILE="${ENV_FILE}" \
    APPEND_RESULTS="${APPEND_RESULTS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BASE_CONFIG_MODE=dir \
    CONFIG_DIR="${TEMP_CONFIG_DIR}" \
    CONFIG_GLOB='task_*.yaml' \
    GENERATED_CONFIG_DIR="${ROOT_DIR}/marble/configs/test_config_research_generated/common_${safe_model_name}_team${team_size}" \
    TASK_PARALLELISM="${TASK_PARALLELISM}" \
    LLMAMEM_TASK_PARALLELISM="${TASK_PARALLELISM}" \
    CONSOLIDATION_INTERVAL="${CONSOLIDATION_INTERVAL}" \
    RUN_TAG="${RUN_TAG_PREFIX}_team${team_size}" \
    "${BASE_RUNNER}"
done

echo "Completed common-task LLMAMem runs for team sizes: ${TEAM_SIZES}"
