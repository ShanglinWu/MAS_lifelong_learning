#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_DIR="${CONFIG_DIR:-${ROOT_DIR}/marble/configs/test_config_database}"
CONFIG_GLOB="${CONFIG_GLOB:-*.yaml}"
MAIN_SCRIPT="${ROOT_DIR}/marble/main.py"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${ROOT_DIR}/marble/configs/test_config_database_generated}"

MODEL_NAME="${MODEL_NAME:-gpt-3.5-turbo}"
EVALUATE_MODEL="${EVALUATE_MODEL:-bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
AGENT_COUNT="${AGENT_COUNT:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
APPEND_RESULTS="${APPEND_RESULTS:-1}"
CASE_START="${CASE_START:-1}"
CASE_END="${CASE_END:-0}"
TASK_PARALLELISM="${TASK_PARALLELISM:-5}"
NOMEM_TASK_PARALLELISM="${NOMEM_TASK_PARALLELISM:-${TASK_PARALLELISM}}"
SHAREDMEM_TASK_PARALLELISM="${SHAREDMEM_TASK_PARALLELISM:-${TASK_PARALLELISM}}"
LLMAMEM_TASK_PARALLELISM="${LLMAMEM_TASK_PARALLELISM:-1}"

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"
CONFIGS=()

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
    find "${CONFIG_DIR}" -maxdepth 1 -type f -name "${CONFIG_GLOB}" | sort -V
}

validate_case_range() {
    local total_configs="$1"

    if ! [[ "${CASE_START}" =~ ^[0-9]+$ ]] || ! [[ "${CASE_END}" =~ ^[0-9]+$ ]]; then
        echo "CASE_START and CASE_END must be non-negative integers." >&2
        exit 1
    fi

    if (( CASE_START < 1 )); then
        echo "CASE_START must be >= 1." >&2
        exit 1
    fi

    if (( CASE_END != 0 && CASE_END < CASE_START )); then
        echo "CASE_END must be 0 or >= CASE_START." >&2
        exit 1
    fi

    if (( CASE_START > total_configs )); then
        echo "Requested start index ${CASE_START}, but only ${total_configs} base case config(s) exist in ${CONFIG_DIR}." >&2
        exit 1
    fi

    if (( CASE_END > total_configs )); then
        echo "Requested end index ${CASE_END}, but only ${total_configs} base case config(s) exist in ${CONFIG_DIR}." >&2
        exit 1
    fi
}

prepare_config() {
    local base_config="$1"
    local memory_type="$2"
    local config_path="$3"
    local output_rel="$4"
    local persist_rel="$5"

    cp "${base_config}" "${config_path}"

    python - "${config_path}" "${MODEL_NAME}" "${EVALUATE_MODEL}" "${AGENT_COUNT}" "${MAX_ITERATIONS}" "${memory_type}" "${output_rel}" "${persist_rel}" <<'PY'
import sys
from ruamel.yaml import YAML

config_path, model_name, evaluate_model, agent_count, max_iterations, memory_type, output_rel, persist_rel = sys.argv[1:]
agent_count = int(agent_count)
max_iterations = int(max_iterations)

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.load(f)

cfg["llm"] = model_name
cfg.setdefault("metrics", {})
cfg["metrics"]["evaluate_llm"] = evaluate_model
cfg.setdefault("environment", {})
cfg["environment"]["max_iterations"] = max_iterations

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

    run_single_case() {
        local base_config="$1"
        local failed=0
        local base_name case_name config_name
        local worker_config_dir worker_config_abs worker_config_rel
        local output_rel output_abs persist_rel persist_abs

        base_name="$(basename "${base_config}")"
        case_name="${base_name%.yaml}"
        config_name="${case_name}_${safe_model_name}_${worker}.yaml"

        worker_config_dir="${GENERATED_CONFIG_DIR}/${safe_model_name}/${worker}"
        worker_config_abs="${worker_config_dir}/${config_name}"
        worker_config_rel="./configs/test_config_database_generated/${safe_model_name}/${worker}/${config_name}"
        output_rel="result/database/${safe_model_name}/${worker}/${case_name}.json"
        output_abs="${ROOT_DIR}/marble/${output_rel}"
        persist_rel="memory_store/database/${safe_model_name}/${worker}/${case_name}"
        persist_abs="${ROOT_DIR}/marble/${persist_rel}"

        mkdir -p "${worker_config_dir}" "$(dirname "${output_abs}")" "${persist_abs}"
        if [[ "${APPEND_RESULTS}" != "1" ]]; then
            : > "${output_abs}"
        fi

        prepare_config "${base_config}" "${memory_type}" "${worker_config_abs}" "${output_rel}" "${persist_rel}"
        load_env_file

        echo "[${worker}] Running ${case_name}..."
        if ! (
            cd "${ROOT_DIR}/marble"
            PYTHONPATH=.. python "${MAIN_SCRIPT}" --config_path "${worker_config_rel}"
        ); then
            echo "[${worker}] Failed: ${case_name}" >&2
            failed=1
        else
            echo "[${worker}] Completed: ${case_name}"
        fi

        return "${failed}"
    }

    for (( index = CASE_START; index <= ${#CONFIGS[@]}; index++ )); do
        if (( CASE_END > 0 && index > CASE_END )); then
            break
        fi

        if (( max_parallelism == 1 )); then
            if ! run_single_case "${CONFIGS[index-1]}"; then
                failed=1
            fi
            continue
        fi

        run_single_case "${CONFIGS[index-1]}" &
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

echo "Starting database simulations"
echo "Config directory: ${CONFIG_DIR}"
echo "Config glob: ${CONFIG_GLOB}"
echo "Generated config directory: ${GENERATED_CONFIG_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Evaluation model: ${EVALUATE_MODEL}"
echo "Agent count: ${AGENT_COUNT}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Env file: ${ENV_FILE}"
echo "Workers: ${WORKERS[*]}"
echo "Case range: start=${CASE_START}, end=${CASE_END:-all}"
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

mapfile -t CONFIGS < <(collect_configs)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
    echo "No configs found in ${CONFIG_DIR} matching ${CONFIG_GLOB}" >&2
    exit 1
fi

validate_case_range "${#CONFIGS[@]}"

pids=()
failed=0
for worker in "${WORKERS[@]}"; do
    run_worker "${worker}" "$(memory_type_for_worker "${worker}")" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "At least one database worker failed." >&2
    exit 1
fi

echo "All database workers completed successfully."
