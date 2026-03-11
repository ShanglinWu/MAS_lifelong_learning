#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_DIR="${CONFIG_DIR:-${ROOT_DIR}/marble/configs/test_config_minecraft}"
CONFIG_GLOB="${CONFIG_GLOB:-*.yaml}"
MAIN_SCRIPT="${ROOT_DIR}/marble/main.py"

MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
EVALUATE_MODEL="${EVALUATE_MODEL:-gpt-3.5-turbo}"
EVALUATE_PROVIDER="${EVALUATE_PROVIDER:-openai}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
APPEND_RESULTS="${APPEND_RESULTS:-1}"
TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-0}"

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"

if [[ -n "${WORKERS:-}" ]]; then
    IFS=',' read -r -a WORKERS <<< "${WORKERS}"
else
    WORKERS=("sharedmem")
fi

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

load_env_file() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "Missing env file: ${ENV_FILE}" >&2
        return 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
}

collect_configs() {
    find "${CONFIG_DIR}" -maxdepth 1 -type f -name "${CONFIG_GLOB}" | sort
}

prepare_config() {
    local base_config="$1"
    local worker="$2"
    local memory_type="$3"
    local config_path="$4"
    local output_rel="$5"
    local persist_rel="$6"

    cp "${base_config}" "${config_path}"

    python - "${config_path}" "${MODEL_NAME}" "${EVALUATE_MODEL}" "${EVALUATE_PROVIDER}" "${memory_type}" "${output_rel}" "${persist_rel}" <<'PY'
import sys
from ruamel.yaml import YAML

config_path, model_name, evaluate_model, evaluate_provider, memory_type, output_rel, persist_rel = sys.argv[1:]

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

cfg["llm"] = model_name
cfg.setdefault("metrics", {})
cfg["metrics"]["evaluate_llm"] = {
    "model": evaluate_model,
    "provider": evaluate_provider,
}
cfg.setdefault("memory", {})
cfg["memory"]["type"] = memory_type
if memory_type == "LLMAMem":
    cfg["memory"]["topology"] = "local"
    cfg["memory"]["persist_dir"] = persist_rel
    cfg["memory"]["consolidation_interval"] = 5
else:
    cfg["memory"].pop("topology", None)
    cfg["memory"].pop("persist_dir", None)
    cfg["memory"].pop("consolidation_interval", None)
cfg.setdefault("output", {})
cfg["output"]["file_path"] = output_rel

with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)
PY
}

echo "Starting Minecraft simulations"
echo "Config directory: ${CONFIG_DIR}"
echo "Config glob: ${CONFIG_GLOB}"
echo "Model: ${MODEL_NAME}"
echo "Evaluation model: ${EVALUATE_MODEL}"
echo "Evaluation provider: ${EVALUATE_PROVIDER}"
echo "Env file: ${ENV_FILE}"
echo "Workers: ${WORKERS[*]}"
echo "Task range: start=${TASK_START}, end=${TASK_END:-all}"
echo "Append results: ${APPEND_RESULTS}"

if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "Config directory does not exist: ${CONFIG_DIR}" >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file does not exist: ${ENV_FILE}" >&2
    exit 1
fi

mapfile -t CONFIGS < <(collect_configs)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
    echo "No configs found in ${CONFIG_DIR} matching ${CONFIG_GLOB}" >&2
    exit 1
fi

failed=0
for worker in "${WORKERS[@]}"; do
    memory_type="$(memory_type_for_worker "${worker}")"
    index=0

    for base_config in "${CONFIGS[@]}"; do
        index=$((index + 1))

        if (( index < TASK_START )); then
            continue
        fi
        if (( TASK_END > 0 && index > TASK_END )); then
            break
        fi

        base_name="$(basename "${base_config}")"
        task_name="${base_name%.yaml}"
        config_name="${task_name}_${safe_model_name}_${worker}.yaml"

        worker_config_abs="${ROOT_DIR}/marble/configs/test_config_minecraft/${config_name}"
        worker_config_rel="./configs/test_config_minecraft/${config_name}"
        output_rel="result/minecraft/${safe_model_name}/${worker}/${task_name}.jsonl"
        output_abs="${ROOT_DIR}/marble/${output_rel}"
        persist_rel="memory_store/minecraft/${safe_model_name}/${worker}/${task_name}"
        persist_abs="${ROOT_DIR}/marble/${persist_rel}"

        mkdir -p "$(dirname "${output_abs}")" "${persist_abs}"
        if [[ "${APPEND_RESULTS}" != "1" ]]; then
            : > "${output_abs}"
        fi

        prepare_config "${base_config}" "${worker}" "${memory_type}" "${worker_config_abs}" "${output_rel}" "${persist_rel}"
        load_env_file

        echo "[${worker}] Running ${task_name}..."
        if ! (
            cd "${ROOT_DIR}/marble"
            PYTHONPATH=.. python "${MAIN_SCRIPT}" --config_path "${worker_config_rel}"
        ); then
            echo "[${worker}] Failed: ${task_name}" >&2
            failed=1
        else
            echo "[${worker}] Completed: ${task_name}"
        fi
    done
done

if [[ "${failed}" -ne 0 ]]; then
    echo "At least one Minecraft task failed." >&2
    exit 1
fi

echo "All Minecraft tasks completed successfully."
