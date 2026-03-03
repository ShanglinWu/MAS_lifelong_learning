#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_CONFIG="${ROOT_DIR}/marble/configs/coding_config/coding_config.yaml"
UPDATE_SCRIPT="${ROOT_DIR}/scripts/coding/utils/update_coding_config.py"
MAIN_SCRIPT="${ROOT_DIR}/marble/main.py"

MODEL_NAME="${MODEL_NAME:-bedrock/converse/qwen.qwen3-next-80b-a3b}"
TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-100}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"

WORKERS=("nomem" "sharedmem" "llmamem")

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

memory_type_for_worker() {
    local worker="$1"
    case "${worker}" in
        nomem) echo "NoMemory" ;;
        sharedmem) echo "SharedMemory" ;;
        llmamem) echo "LLMAMem" ;;
        *)
            echo "Unknown worker: ${worker}" >&2
            exit 1
            ;;
    esac
}

prepare_worker_config() {
    local worker="$1"
    local memory_type="$2"
    local config_path="$3"
    local workspace_rel="$4"
    local persist_rel="$5"
    local output_rel="$6"

    cp "${BASE_CONFIG}" "${config_path}"

    python - "${config_path}" "${memory_type}" "${workspace_rel}" "${persist_rel}" "${output_rel}" <<'PY'
import sys
from ruamel.yaml import YAML

config_path, memory_type, workspace_rel, persist_rel, output_rel = sys.argv[1:]

yaml = YAML()
yaml.preserve_quotes = True

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

cfg["environment"]["workspace_dir"] = workspace_rel
cfg["memory"]["type"] = memory_type

if memory_type == "LLMAMem":
    cfg["memory"]["topology"] = "shared"
    cfg["memory"]["persist_dir"] = persist_rel
    cfg["memory"]["consolidation_interval"] = 3
else:
    cfg["memory"].pop("topology", None)
    cfg["memory"].pop("persist_dir", None)
    cfg["memory"].pop("consolidation_interval", None)

cfg["output"]["file_path"] = output_rel

with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)
PY
}

run_worker() {
    local worker="$1"
    local memory_type="$2"

    local config_path="${ROOT_DIR}/marble/configs/coding_config/coding_config_${worker}.yaml"
    local config_rel="./configs/coding_config/coding_config_${worker}.yaml"
    local workspace_rel="workspace_${worker}"
    local workspace_abs="${ROOT_DIR}/marble/${workspace_rel}"
    local persist_rel="memory_store/${worker}"
    local persist_abs="${ROOT_DIR}/marble/${persist_rel}"
    local output_rel="result/${worker}/development_output_${safe_model_name}.jsonl"
    local output_abs="${ROOT_DIR}/marble/${output_rel}"
    local solution_log_dir="${ROOT_DIR}/marble/logs/${safe_model_name}/${worker}"

    mkdir -p "${solution_log_dir}" "$(dirname "${output_abs}")" "${workspace_abs}" "${persist_abs}"
    : > "${output_abs}"

    prepare_worker_config "${worker}" "${memory_type}" "${config_path}" "${workspace_rel}" "${persist_rel}" "${output_rel}"

    for id in $(seq "${TASK_START}" "${TASK_END}"); do
        echo "[${worker}] Processing task ID=${id}..."

        rm -rf "${workspace_abs:?}"/*
        python "${UPDATE_SCRIPT}" --benchmark_id "${id}" --config_path "${config_path}" --llm_model "${MODEL_NAME}"
        load_bedrock_token_from_env

        (
            cd "${ROOT_DIR}/marble"
            PYTHONPATH=.. python "${MAIN_SCRIPT}" --config_path "${config_rel}"
        )

        if [[ -f "${workspace_abs}/solution.py" ]]; then
            cp "${workspace_abs}/solution.py" "${solution_log_dir}/solution_${id}.py"
        else
            echo "[${worker}] warning: solution.py missing for task ${id}" >&2
        fi

        echo "[${worker}] Task ${id} completed."
    done

    echo "[${worker}] All tasks completed."
}

echo "Starting workers for model: ${MODEL_NAME}"
echo "Task range: ${TASK_START}-${TASK_END}"
echo "Env file: ${ENV_FILE}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file does not exist: ${ENV_FILE}" >&2
    exit 1
fi

pids=()
for worker in "${WORKERS[@]}"; do
    run_worker "${worker}" "$(memory_type_for_worker "${worker}")" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "At least one worker failed." >&2
    exit 1
fi

echo "All workers completed successfully."
