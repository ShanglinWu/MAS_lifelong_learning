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

# Comma-separated worker list, e.g.:
#   WORKERS_CSV=amem
#   WORKERS_CSV=nomem,sharedmem,llmamem,amem
WORKERS_CSV="${WORKERS_CSV:-nomem,sharedmem,llmamem}"
IFS=',' read -r -a WORKERS <<< "${WORKERS_CSV}"

AMEM_TOPOLOGY="${AMEM_TOPOLOGY:-local}"
AMEM_EVOLUTION_ENABLED="${AMEM_EVOLUTION_ENABLED:-true}"
AMEM_EVOLUTION_THRESHOLD="${AMEM_EVOLUTION_THRESHOLD:-100}"
AMEM_EMBEDDING_MODEL="${AMEM_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
AMEM_LLM_BACKEND="${AMEM_LLM_BACKEND:-bedrock}"
AMEM_LLM_MODEL="${AMEM_LLM_MODEL:-bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0}"
AMEM_RETRIEVAL_K="${AMEM_RETRIEVAL_K:-5}"

load_bedrock_token_from_env() {
    validate_token() {
        local token="$1"
        if [[ -z "${token}" ]]; then
            echo "AWS_BEARER_TOKEN_BEDROCK is empty." >&2
            return 1
        fi
        if [[ "${token}" == *" "* || "${token}" == *$'\t'* ]]; then
            echo "AWS_BEARER_TOKEN_BEDROCK contains whitespace; token is likely malformed." >&2
            return 1
        fi
        # Common copy/paste mistake: using a presigned query string instead of the API key.
        if [[ "${token}" == *"Action=CallWithBearerToken"* || "${token}" == *"&X-Amz-"* ]]; then
            echo "AWS_BEARER_TOKEN_BEDROCK looks like a presigned query string, not a Bedrock API key." >&2
            return 1
        fi
        return 0
    }

    local shell_token file_token token_line token
    shell_token="${AWS_BEARER_TOKEN_BEDROCK:-}"
    file_token=""

    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "Missing env file: ${ENV_FILE}" >&2
        return 1
    fi

    token_line="$(grep -m1 '^AWS_BEARER_TOKEN_BEDROCK=' "${ENV_FILE}" || true)"
    file_token="${token_line#AWS_BEARER_TOKEN_BEDROCK=}"
    file_token="${file_token%\"}"
    file_token="${file_token#\"}"
    file_token="${file_token%\'}"
    file_token="${file_token#\'}"
    file_token="$(printf '%s' "${file_token}" | tr -d '\r\n')"

    # Prefer ENV_FILE token so stale shell exports do not silently break runs.
    if [[ -n "${shell_token}" && -n "${file_token}" && "${shell_token}" != "${file_token}" ]]; then
        if [[ -z "${BEDROCK_TOKEN_MISMATCH_WARNED:-}" ]]; then
            echo "AWS_BEARER_TOKEN_BEDROCK differs between shell env and ${ENV_FILE}; using ${ENV_FILE} value." >&2
            export BEDROCK_TOKEN_MISMATCH_WARNED=1
        fi
    fi

    if [[ -n "${file_token}" ]]; then
        token="${file_token}"
    elif [[ -n "${shell_token}" ]]; then
        token="${shell_token}"
    else
        echo "AWS_BEARER_TOKEN_BEDROCK is empty or missing in ${ENV_FILE} and shell env." >&2
        return 1
    fi

    validate_token "${token}" || return 1
    export AWS_BEARER_TOKEN_BEDROCK="${token}"

    # Export AWS_REGION_NAME so LiteLLM targets the correct region.
    if [[ -z "${AWS_REGION_NAME:-}" ]]; then
        local region_line region
        region_line="$(grep -m1 '^AWS_REGION_NAME=' "${ENV_FILE}" || true)"
        region="${region_line#AWS_REGION_NAME=}"
        region="${region%\"}"
        region="${region#\"}"
        region="${region%\'}"
        region="${region#\'}"
        region="$(printf '%s' "${region}" | tr -d '\r\n')"
        if [[ -n "${region}" ]]; then
            export AWS_REGION_NAME="${region}"
        fi
    fi
}

memory_type_for_worker() {
    local worker="$1"
    case "${worker}" in
        nomem) echo "NoMemory" ;;
        sharedmem) echo "SharedMemory" ;;
        llmamem) echo "LLMAMem" ;;
        amem) echo "SharedMemory" ;;
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
    local amem_topology="$7"
    local amem_evolution_enabled="$8"
    local amem_evolution_threshold="$9"
    local amem_embedding_model="${10}"
    local amem_llm_backend="${11}"
    local amem_llm_model="${12}"
    local amem_retrieval_k="${13}"

    cp "${BASE_CONFIG}" "${config_path}"

    python - "${config_path}" "${memory_type}" "${workspace_rel}" "${persist_rel}" "${output_rel}" "${worker}" "${amem_topology}" "${amem_evolution_enabled}" "${amem_evolution_threshold}" "${amem_embedding_model}" "${amem_llm_backend}" "${amem_llm_model}" "${amem_retrieval_k}" <<'PY'
import sys
from ruamel.yaml import YAML

(
    config_path,
    memory_type,
    workspace_rel,
    persist_rel,
    output_rel,
    worker,
    amem_topology,
    amem_evolution_enabled,
    amem_evolution_threshold,
    amem_embedding_model,
    amem_llm_backend,
    amem_llm_model,
    amem_retrieval_k,
) = sys.argv[1:]

yaml = YAML()
yaml.preserve_quotes = True

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

cfg["environment"]["workspace_dir"] = workspace_rel
cfg["memory"]["type"] = memory_type

# Always disable advanced memory backends for this demo script unless a
# dedicated worker/mode enables them explicitly.
cfg.setdefault("memory", {})
cfg["memory"].setdefault("amem", {})
cfg["memory"].setdefault("llma_mem", {})
cfg["memory"]["amem"]["enabled"] = False
cfg["memory"]["llma_mem"]["enabled"] = False

if memory_type == "LLMAMem":
    cfg["memory"]["topology"] = "shared"
    cfg["memory"]["persist_dir"] = persist_rel
    cfg["memory"]["consolidation_interval"] = 3
else:
    cfg["memory"].pop("topology", None)
    cfg["memory"].pop("persist_dir", None)
    cfg["memory"].pop("consolidation_interval", None)

if worker == "amem":
    cfg["memory"]["amem"]["enabled"] = True
    cfg["memory"]["amem"]["topology"] = amem_topology
    cfg["memory"]["amem"]["evolution_enabled"] = (
        str(amem_evolution_enabled).strip().lower() == "true"
    )
    cfg["memory"]["amem"]["evolution_threshold"] = int(amem_evolution_threshold)
    cfg["memory"]["amem"]["embedding_model"] = amem_embedding_model
    cfg["memory"]["amem"]["llm_backend"] = amem_llm_backend
    cfg["memory"]["amem"]["llm_model"] = amem_llm_model
    cfg["memory"]["amem"]["retrieval_k"] = int(amem_retrieval_k)

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

    prepare_worker_config "${worker}" "${memory_type}" "${config_path}" "${workspace_rel}" "${persist_rel}" "${output_rel}" \
        "${AMEM_TOPOLOGY}" "${AMEM_EVOLUTION_ENABLED}" "${AMEM_EVOLUTION_THRESHOLD}" "${AMEM_EMBEDDING_MODEL}" \
        "${AMEM_LLM_BACKEND}" "${AMEM_LLM_MODEL}" "${AMEM_RETRIEVAL_K}"

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
echo "Workers: ${WORKERS[*]}"
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
