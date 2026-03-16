#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_CONFIG="${ROOT_DIR}/marble/configs/coding_config/coding_config_bedrock_converse_deepseek.v3.2_llmamem.yaml"
UPDATE_SCRIPT="${ROOT_DIR}/scripts/coding/utils/update_coding_config.py"
MAIN_SCRIPT="${ROOT_DIR}/marble/main.py"

MODEL_NAME="${MODEL_NAME:-bedrock/converse/deepseek.v3.2}"
TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-100}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
APPEND_RESULTS="${APPEND_RESULTS:-0}"
TOPOLOGIES="${TOPOLOGIES:-shared,hybrid}"
CONSOLIDATION_INTERVALS="${CONSOLIDATION_INTERVALS:-2,5,10,20}"
RUN_TAG="${RUN_TAG:-ablation_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"
CONFIG_BASE_DIR="${ROOT_DIR}/marble/configs/coding_config/generated_ablations/${RUN_TAG}"

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

prepare_condition_config() {
    local topology="$1"
    local interval="$2"
    local config_path="$3"
    local workspace_rel="$4"
    local persist_rel="$5"
    local output_rel="$6"

    mkdir -p "$(dirname "${config_path}")"
    cp "${BASE_CONFIG}" "${config_path}"

    "${PYTHON_BIN}" - "${config_path}" "${workspace_rel}" "${persist_rel}" "${output_rel}" "${topology}" "${interval}" "${MODEL_NAME}" <<'PY'
import sys
from ruamel.yaml import YAML

(
    config_path,
    workspace_rel,
    persist_rel,
    output_rel,
    topology,
    interval,
    model_name,
) = sys.argv[1:]

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

cfg["llm"] = model_name
cfg["environment"]["workspace_dir"] = workspace_rel
cfg["memory"]["type"] = "LLMAMem"
cfg["memory"]["topology"] = topology
cfg["memory"]["persist_dir"] = persist_rel
cfg["memory"]["consolidation_interval"] = int(interval)
cfg["output"]["file_path"] = output_rel

with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)
PY
}

run_condition() {
    local topology="$1"
    local interval="$2"
    local condition_name="${topology}_ci${interval}"
    local config_path="${CONFIG_BASE_DIR}/${condition_name}.yaml"
    local config_rel="./configs/coding_config/generated_ablations/${RUN_TAG}/${condition_name}.yaml"
    local workspace_rel="workspace_${safe_model_name}/${RUN_TAG}/${condition_name}"
    local workspace_abs="${ROOT_DIR}/marble/${workspace_rel}"
    local persist_rel="memory_store/${safe_model_name}/${RUN_TAG}/${condition_name}"
    local persist_abs="${ROOT_DIR}/marble/${persist_rel}"
    local output_rel="result/${safe_model_name}/${RUN_TAG}/${condition_name}/development_output.jsonl"
    local output_abs="${ROOT_DIR}/marble/${output_rel}"
    local solution_log_dir="${ROOT_DIR}/marble/logs/${safe_model_name}/${RUN_TAG}/${condition_name}"

    mkdir -p "${workspace_abs}" "${persist_abs}" "$(dirname "${output_abs}")" "${solution_log_dir}"
    if [[ "${APPEND_RESULTS}" != "1" ]]; then
        rm -f "${output_abs}"
    fi

    prepare_condition_config "${topology}" "${interval}" "${config_path}" "${workspace_rel}" "${persist_rel}" "${output_rel}"

    for id in $(seq "${TASK_START}" "${TASK_END}"); do
        echo "[${condition_name}] Processing task ID=${id}..."

        rm -rf "${workspace_abs:?}"/*
        "${PYTHON_BIN}" "${UPDATE_SCRIPT}" --benchmark_id "${id}" --config_path "${config_path}" --llm_model "${MODEL_NAME}"
        load_bedrock_token_from_env

        (
            cd "${ROOT_DIR}/marble"
            PYTHONPATH=.. "${PYTHON_BIN}" "${MAIN_SCRIPT}" --config_path "${config_rel}"
        )

        if [[ -f "${workspace_abs}/solution.py" ]]; then
            cp "${workspace_abs}/solution.py" "${solution_log_dir}/solution_${id}.py"
        else
            echo "[${condition_name}] warning: solution.py missing for task ${id}" >&2
        fi

        echo "[${condition_name}] Task ${id} completed."
    done

    echo "[${condition_name}] All tasks completed."
}

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Base config does not exist: ${BASE_CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file does not exist: ${ENV_FILE}" >&2
    exit 1
fi

IFS=',' read -r -a TOPOLOGY_ARRAY <<< "${TOPOLOGIES}"
IFS=',' read -r -a INTERVAL_ARRAY <<< "${CONSOLIDATION_INTERVALS}"

for topology in "${TOPOLOGY_ARRAY[@]}"; do
    topology="$(echo "${topology}" | xargs)"
    case "${topology}" in
        local|shared|hybrid) ;;
        *)
            echo "Invalid topology: ${topology}. Use local, shared, or hybrid." >&2
            exit 1
            ;;
    esac
done

for interval in "${INTERVAL_ARRAY[@]}"; do
    interval="$(echo "${interval}" | xargs)"
    if ! [[ "${interval}" =~ ^[0-9]+$ ]]; then
        echo "Invalid consolidation interval: ${interval}" >&2
        exit 1
    fi
done

echo "Starting LLMA-Mem coding ablation"
echo "Model: ${MODEL_NAME}"
echo "Task range: ${TASK_START}-${TASK_END}"
echo "Topologies: ${TOPOLOGIES}"
echo "Consolidation intervals: ${CONSOLIDATION_INTERVALS}"
echo "Run tag: ${RUN_TAG}"
echo "Generated configs: ${CONFIG_BASE_DIR}"
echo "Append results: ${APPEND_RESULTS}"

pids=()
for topology in "${TOPOLOGY_ARRAY[@]}"; do
    topology="$(echo "${topology}" | xargs)"
    for interval in "${INTERVAL_ARRAY[@]}"; do
        interval="$(echo "${interval}" | xargs)"
        run_condition "${topology}" "${interval}" &
        pids+=("$!")
    done
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "At least one ablation condition failed." >&2
    exit 1
fi

echo "All ablation conditions completed successfully."
