#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_DIR="${CONFIG_DIR:-${ROOT_DIR}/marble/configs/test_config_research}"
CONFIG_GLOB="${CONFIG_GLOB:-profile_[0-9]*.yaml}"
MAIN_SCRIPT="${ROOT_DIR}/marble/main.py"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${ROOT_DIR}/marble/configs/test_config_research_generated}"
RESEARCH_JSONL="${RESEARCH_JSONL:-${ROOT_DIR}/multiagentbench/research/research_main.jsonl}"
BASE_CONFIG_MODE="${BASE_CONFIG_MODE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME="${MODEL_NAME:-gpt-3.5-turbo}"
EVALUATE_MODEL="${EVALUATE_MODEL:-bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
AGENT_COUNT="${AGENT_COUNT:-3}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
APPEND_RESULTS="${APPEND_RESULTS:-1}"
CONSOLIDATION_INTERVAL="${CONSOLIDATION_INTERVAL:-5}"
RUN_TAG="${RUN_TAG:-}"
PROFILE_START="${PROFILE_START:-${TASK_START:-1}}"
PROFILE_END="${PROFILE_END:-${TASK_END:-0}}"
TASK_PARALLELISM="${TASK_PARALLELISM:-5}"
NOMEM_TASK_PARALLELISM="${NOMEM_TASK_PARALLELISM:-${TASK_PARALLELISM}}"
SHAREDMEM_TASK_PARALLELISM="${SHAREDMEM_TASK_PARALLELISM:-${TASK_PARALLELISM}}"
LLMAMEM_TASK_PARALLELISM="${LLMAMEM_TASK_PARALLELISM:-1}"

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"
safe_run_tag="$(echo "${RUN_TAG}" | tr '/:' '__' | tr -cd '[:alnum:]_.-')"
ACTIVE_CONFIG_DIR="${CONFIG_DIR}"
ACTIVE_CONFIG_GLOB="${CONFIG_GLOB}"
TEMP_BASE_CONFIG_DIR=""
CONFIGS=()

if [[ -n "${WORKERS:-}" ]]; then
    IFS=',' read -r -a WORKERS <<< "${WORKERS}"
else
    WORKERS=("nomem" "sharedmem" "llmamem")
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

prepare_worker_config() {
    local base_config="$1"
    local worker="$2"
    local memory_type="$3"
    local config_path="$4"
    local output_rel="$5"
    local persist_rel="$6"

    cp "${base_config}" "${config_path}"

    "${PYTHON_BIN}" - "${config_path}" "${MODEL_NAME}" "${EVALUATE_MODEL}" "${AGENT_COUNT}" "${memory_type}" "${output_rel}" "${persist_rel}" "${CONSOLIDATION_INTERVAL}" <<'PY'
import sys
from ruamel.yaml import YAML

(
    config_path,
    model_name,
    evaluate_model,
    agent_count,
    memory_type,
    output_rel,
    persist_rel,
    consolidation_interval,
) = sys.argv[1:]
agent_count = int(agent_count)
consolidation_interval = int(consolidation_interval)

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

cfg["llm"] = model_name
cfg.setdefault("metrics", {})
cfg["metrics"]["evaluate_llm"] = evaluate_model

agents = cfg.get("agents", [])
selected_agent_ids = {agent["agent_id"] for agent in agents[:agent_count]}
cfg["agents"] = agents[:agent_count]

relationships = cfg.get("relationships", [])
cfg["relationships"] = [
    relation
    for relation in relationships
    if len(relation) >= 2
    and relation[0] in selected_agent_ids
    and relation[1] in selected_agent_ids
]

cfg.setdefault("memory", {})
cfg["memory"]["type"] = memory_type

if memory_type == "LLMAMem":
    cfg["memory"]["topology"] = "local"
    cfg["memory"]["persist_dir"] = persist_rel
    cfg["memory"]["consolidation_interval"] = consolidation_interval
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

collect_configs() {
    find "${ACTIVE_CONFIG_DIR}" -maxdepth 1 -type f -name "${ACTIVE_CONFIG_GLOB}" | sort -V
}

validate_profile_range() {
    local total_configs="$1"

    if ! [[ "${PROFILE_START}" =~ ^[0-9]+$ ]] || ! [[ "${PROFILE_END}" =~ ^[0-9]+$ ]]; then
        echo "PROFILE_START/TASK_START and PROFILE_END/TASK_END must be non-negative integers." >&2
        exit 1
    fi

    if (( PROFILE_START < 1 )); then
        echo "PROFILE_START/TASK_START must be >= 1." >&2
        exit 1
    fi

    if (( PROFILE_END != 0 && PROFILE_END < PROFILE_START )); then
        echo "PROFILE_END/TASK_END must be 0 or >= PROFILE_START/TASK_START." >&2
        exit 1
    fi

    if (( PROFILE_START > total_configs )); then
        echo "Requested start index ${PROFILE_START}, but only ${total_configs} base task config(s) exist in ${ACTIVE_CONFIG_DIR}." >&2
        exit 1
    fi

    if (( PROFILE_END > total_configs )); then
        echo "Requested end index ${PROFILE_END}, but only ${total_configs} base task config(s) exist in ${ACTIVE_CONFIG_DIR}." >&2
        exit 1
    fi
}

cleanup_temp_base_configs() {
    if [[ -n "${TEMP_BASE_CONFIG_DIR}" && -d "${TEMP_BASE_CONFIG_DIR}" ]]; then
        rm -rf "${TEMP_BASE_CONFIG_DIR}"
    fi
}

prepare_base_configs() {
    local mode="${BASE_CONFIG_MODE}"

    if [[ "${mode}" == "auto" ]]; then
        if [[ -f "${RESEARCH_JSONL}" ]]; then
            mode="jsonl"
        else
            mode="dir"
        fi
    fi

    case "${mode}" in
        jsonl)
            if [[ ! -f "${RESEARCH_JSONL}" ]]; then
                echo "Research benchmark JSONL does not exist: ${RESEARCH_JSONL}" >&2
                exit 1
            fi

            TEMP_BASE_CONFIG_DIR="$(mktemp -d "${ROOT_DIR}/marble/configs/test_config_research_runtime.${safe_model_name}.XXXXXX")"
            "${PYTHON_BIN}" - "${RESEARCH_JSONL}" "${TEMP_BASE_CONFIG_DIR}" <<'PY'
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

    if isinstance(data.get("metrics"), dict) and data["metrics"].get("evaluate_llm", "") == "":
        data["metrics"]["evaluate_llm"] = defaults["metrics"]["evaluate_llm"]

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
        if not isinstance(task_id, int):
            raise ValueError(f"Invalid or missing task_id in record: {line[:120]}")
        output_path = output_dir / f"task_{task_id}.yaml"
        with output_path.open("w", encoding="utf-8") as outfile:
            yaml.dump(data, outfile)
PY

            ACTIVE_CONFIG_DIR="${TEMP_BASE_CONFIG_DIR}"
            ACTIVE_CONFIG_GLOB="task_[0-9]*.yaml"
            ;;
        dir)
            ACTIVE_CONFIG_DIR="${CONFIG_DIR}"
            ACTIVE_CONFIG_GLOB="${CONFIG_GLOB}"
            ;;
        *)
            echo "Unknown BASE_CONFIG_MODE: ${BASE_CONFIG_MODE}. Expected auto, jsonl, or dir." >&2
            exit 1
            ;;
    esac
}

run_worker() {
    local worker="$1"
    local memory_type="$2"
    local max_parallelism
    local running_jobs=0
    local failed=0

    case "${worker}" in
        nomem) max_parallelism="${NOMEM_TASK_PARALLELISM}" ;;
        sharedmem) max_parallelism="${SHAREDMEM_TASK_PARALLELISM}" ;;
        llmamem) max_parallelism="${LLMAMEM_TASK_PARALLELISM}" ;;
        *) max_parallelism=1 ;;
    esac

    if ! [[ "${max_parallelism}" =~ ^[0-9]+$ ]] || (( max_parallelism < 1 )); then
        echo "Invalid task parallelism for ${worker}: ${max_parallelism}" >&2
        return 1
    fi

    run_single_profile() {
        local base_config="$1"
        local failed=0

        local base_name profile_name config_name
        base_name="$(basename "${base_config}")"
        profile_name="${base_name%.yaml}"
        config_name="${profile_name}_${safe_model_name}_${worker}.yaml"

        local worker_config_dir="${GENERATED_CONFIG_DIR}/${safe_model_name}/${worker}"
        local worker_config_abs="${worker_config_dir}/${config_name}"
        local worker_config_rel="${worker_config_abs}"
        local output_rel
        local output_abs
        local persist_rel
        local persist_abs

        if [[ -n "${safe_run_tag}" ]]; then
            output_rel="result/research/${safe_model_name}/${worker}/${safe_run_tag}/${profile_name}.jsonl"
        else
            output_rel="result/research/${safe_model_name}/${worker}/${profile_name}.jsonl"
        fi
        output_abs="${ROOT_DIR}/marble/${output_rel}"

        if [[ "${worker}" == "llmamem" ]]; then
            if [[ -n "${safe_run_tag}" ]]; then
                persist_rel="memory_store/research/${safe_model_name}/${worker}/${safe_run_tag}"
            else
                persist_rel="memory_store/research/${safe_model_name}/${worker}"
            fi
        else
            if [[ -n "${safe_run_tag}" ]]; then
                persist_rel="memory_store/research/${safe_model_name}/${worker}/${safe_run_tag}/${profile_name}"
            else
                persist_rel="memory_store/research/${safe_model_name}/${worker}/${profile_name}"
            fi
        fi
        persist_abs="${ROOT_DIR}/marble/${persist_rel}"

        mkdir -p "${worker_config_dir}" "$(dirname "${output_abs}")" "${persist_abs}"
        if [[ "${APPEND_RESULTS}" != "1" ]]; then
            : > "${output_abs}"
        fi

        prepare_worker_config "${base_config}" "${worker}" "${memory_type}" "${worker_config_abs}" "${output_rel}" "${persist_rel}"
        load_bedrock_token_from_env

        echo "[${worker}] Running ${profile_name}..."
        if ! (
            cd "${ROOT_DIR}/marble"
            PYTHONPATH=.. "${PYTHON_BIN}" "${MAIN_SCRIPT}" --config_path "${worker_config_rel}"
        ); then
            echo "[${worker}] Failed: ${profile_name}" >&2
            failed=1
        else
            echo "[${worker}] Completed: ${profile_name}"
        fi
        return "${failed}"
    }

    for (( index = PROFILE_START; index <= ${#CONFIGS[@]}; index++ )); do
        if (( PROFILE_END > 0 && index > PROFILE_END )); then
            break
        fi

        if (( max_parallelism == 1 )); then
            if ! run_single_profile "${CONFIGS[index-1]}"; then
                failed=1
            fi
            continue
        fi

        run_single_profile "${CONFIGS[index-1]}" &
        running_jobs=$((running_jobs + 1))

        if (( running_jobs >= max_parallelism )); then
            if ! wait -n; then
                failed=1
            fi
            running_jobs=$((running_jobs - 1))
        fi
    done

    while (( running_jobs > 0 )); do
        if ! wait -n; then
            failed=1
        fi
        running_jobs=$((running_jobs - 1))
    done

    return "${failed}"
}

echo "Starting research simulations"
echo "Base config mode: ${BASE_CONFIG_MODE}"
echo "Config directory: ${CONFIG_DIR}"
echo "Config glob: ${CONFIG_GLOB}"
echo "Research benchmark JSONL: ${RESEARCH_JSONL}"
echo "Generated config directory: ${GENERATED_CONFIG_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Evaluation model: ${EVALUATE_MODEL}"
echo "Agent count: ${AGENT_COUNT}"
echo "Env file: ${ENV_FILE}"
echo "Profile range: start=${PROFILE_START}, end=${PROFILE_END:-all}"
echo "Append results: ${APPEND_RESULTS}"
echo "Task parallelism: nomem=${NOMEM_TASK_PARALLELISM}, sharedmem=${SHAREDMEM_TASK_PARALLELISM}, llmamem=${LLMAMEM_TASK_PARALLELISM}"

if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "Config directory does not exist: ${CONFIG_DIR}" >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file does not exist: ${ENV_FILE}" >&2
    exit 1
fi

trap cleanup_temp_base_configs EXIT

prepare_base_configs
echo "Active config directory: ${ACTIVE_CONFIG_DIR}"
echo "Active config glob: ${ACTIVE_CONFIG_GLOB}"

mapfile -t CONFIGS < <(collect_configs)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
    echo "No configs found in ${ACTIVE_CONFIG_DIR} matching ${ACTIVE_CONFIG_GLOB}" >&2
    exit 1
fi

validate_profile_range "${#CONFIGS[@]}"

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
    echo "At least one research worker failed." >&2
    exit 1
fi

echo "All research workers completed successfully."
