#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREP_SCRIPT="${ROOT_DIR}/scripts/database/utils/prepare_amem_database_config.py"
MAIN_SCRIPT="${ROOT_DIR}/marble/main.py"
CONFIG_DIR="${CONFIG_DIR:-${ROOT_DIR}/marble/configs/test_config_database}"
CONFIG_GLOB="${CONFIG_GLOB:-gpt-3.5-turbo_*.yaml}"
GENERATED_CONFIG_DIR_BASE="${GENERATED_CONFIG_DIR_BASE:-${ROOT_DIR}/marble/configs/test_config_database_generated_amem}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_NAME="${MODEL_NAME:-bedrock/converse/qwen.qwen3-32b-v1:0}"
EVALUATE_MODEL="${EVALUATE_MODEL:-bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bedrock/amazon.titan-embed-text-v2:0}"
CASE_START="${CASE_START:-1}"
CASE_END="${CASE_END:-100}"
CASE_IDS="${CASE_IDS:-}"
TASK_PARALLELISM="${TASK_PARALLELISM:-4}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
TEAM_SIZE="${TEAM_SIZE:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
RUN_TAG="${RUN_TAG:-amem_database_default}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-3}"
MAX_MEMORY_CONTEXT="${MAX_MEMORY_CONTEXT:-5}"
LINK_THRESHOLD="${LINK_THRESHOLD:-0.72}"
PORT_STRIDE="${PORT_STRIDE:-10}"
RUN_PORT_OFFSET="${RUN_PORT_OFFSET:-}"
RUN_PORT_BLOCK_SIZE="${RUN_PORT_BLOCK_SIZE:-3000}"

if [[ "${TEAM_SIZE}" != "3" ]]; then
    echo "This runner currently enforces TEAM_SIZE=3." >&2
    exit 1
fi

if [[ "${MAX_ITERATIONS}" != "3" ]]; then
    echo "This runner currently enforces MAX_ITERATIONS=3." >&2
    exit 1
fi

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR_BASE}/${safe_model_name}/${RUN_TAG}"
RESULT_DIR_REL="result/database/${safe_model_name}/amem"
RESULT_DIR_ABS="${ROOT_DIR}/marble/${RESULT_DIR_REL}"
CONFIGS=()

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
    find "${CONFIG_DIR}" -maxdepth 1 -type f -name "${CONFIG_GLOB}" \
        ! -name '*_bedrock_*' \
        ! -name '*_nomem.yaml' \
        ! -name '*_sharedmem.yaml' \
        ! -name '*_llmamem.yaml' \
        ! -name '*_amem.yaml' | sort -V
}

validate_parallelism() {
    if ! [[ "${TASK_PARALLELISM}" =~ ^[0-9]+$ ]] || (( TASK_PARALLELISM < 1 )); then
        echo "TASK_PARALLELISM must be an integer >= 1." >&2
        exit 1
    fi

    if ! [[ "${PORT_STRIDE}" =~ ^[0-9]+$ ]] || (( PORT_STRIDE < 4 )); then
        echo "PORT_STRIDE must be an integer >= 4." >&2
        exit 1
    fi

    if ! [[ "${RUN_PORT_BLOCK_SIZE}" =~ ^[0-9]+$ ]] || (( RUN_PORT_BLOCK_SIZE < 1000 )); then
        echo "RUN_PORT_BLOCK_SIZE must be an integer >= 1000." >&2
        exit 1
    fi
}

validate_case_bounds() {
    local total_configs="$1"

    if [[ -n "${CASE_IDS}" ]]; then
        while IFS= read -r case_id; do
            if ! [[ "${case_id}" =~ ^[0-9]+$ ]] || (( case_id < 1 || case_id > total_configs )); then
                echo "Invalid CASE_IDS entry: ${case_id}. Valid range is 1-${total_configs}." >&2
                exit 1
            fi
        done < <(printf '%s\n' "${CASE_IDS}" | tr ',' '\n' | sed '/^\s*$/d')
        return 0
    fi

    if ! [[ "${CASE_START}" =~ ^[0-9]+$ ]] || ! [[ "${CASE_END}" =~ ^[0-9]+$ ]]; then
        echo "CASE_START and CASE_END must be integers." >&2
        exit 1
    fi

    if (( CASE_START < 1 || CASE_END < CASE_START || CASE_END > total_configs )); then
        echo "Invalid case range ${CASE_START}-${CASE_END}. Valid range is 1-${total_configs}." >&2
        exit 1
    fi
}

case_sequence() {
    if [[ -n "${CASE_IDS}" ]]; then
        printf '%s\n' "${CASE_IDS}" | tr ',' '\n' | sed '/^\s*$/d'
    else
        seq "${CASE_START}" "${CASE_END}"
    fi
}

compute_run_port_offset() {
    if [[ -n "${RUN_PORT_OFFSET}" ]]; then
        if ! [[ "${RUN_PORT_OFFSET}" =~ ^[0-9]+$ ]]; then
            echo "RUN_PORT_OFFSET must be a non-negative integer." >&2
            exit 1
        fi
        echo "${RUN_PORT_OFFSET}"
        return
    fi

    local checksum
    checksum="$(printf '%s' "${safe_model_name}:${RUN_TAG}" | cksum | awk '{print $1}')"
    echo $(( (checksum % 10) * RUN_PORT_BLOCK_SIZE ))
}

case_name_for_index() {
    local case_index="$1"
    local base_config="${CONFIGS[case_index-1]}"
    basename "${base_config}" .yaml
}

compose_project_name_for_case() {
    local case_index="$1"
    local name_hash
    name_hash="$(printf '%s' "${safe_model_name}:${RUN_TAG}:${case_index}" | cksum | awk '{print $1}')"
    printf 'amemdb_c%s_%s' "${case_index}" "${name_hash}"
}

output_path_for_case() {
    local case_index="$1"
    local case_name
    case_name="$(case_name_for_index "${case_index}")"
    echo "${RESULT_DIR_ABS}/${case_name}.json"
}

pending_case_sequence() {
    while IFS= read -r case_index; do
        local output_abs
        output_abs="$(output_path_for_case "${case_index}")"
        if [[ "${SKIP_COMPLETED}" == "1" && -s "${output_abs}" ]]; then
            echo "[amem-database] Skipping completed case ${case_index}: $(basename "${output_abs}")" >&2
            continue
        fi
        echo "${case_index}"
    done < <(case_sequence)
}

cleanup_case_stack() {
    local compose_project_name="$1"
    (
        cd "${ROOT_DIR}/marble/environments/db_env_docker"
        COMPOSE_PROJECT_NAME="${compose_project_name}" \
        DB_COMPOSE_PROJECT_NAME="${compose_project_name}" \
        sudo -E docker compose down -v >/dev/null 2>&1 || true
    )
}

run_case() {
    local case_index="$1"
    local base_config="${CONFIGS[case_index-1]}"
    local case_name config_name target_config config_rel
    local output_rel output_abs
    local case_seed postgres_port prometheus_port node_exporter_port pg_exporter_port
    local compose_project_name db_name table_name

    case_name="$(basename "${base_config}" .yaml)"
    config_name="case_${case_index}.yaml"
    target_config="${GENERATED_CONFIG_DIR}/${config_name}"
    config_rel="./configs/test_config_database_generated_amem/${safe_model_name}/${RUN_TAG}/${config_name}"
    output_rel="${RESULT_DIR_REL}/${case_name}.json"
    output_abs="${ROOT_DIR}/marble/${output_rel}"

    case_seed=$(( 20000 + RUN_PORT_OFFSET_VALUE + (case_index * PORT_STRIDE) ))
    postgres_port="${case_seed}"
    prometheus_port="$(( case_seed + 1 ))"
    node_exporter_port="$(( case_seed + 2 ))"
    pg_exporter_port="$(( case_seed + 3 ))"

    compose_project_name="$(compose_project_name_for_case "${case_index}")"
    db_name="$(printf 'sysbench_amem_%s' "${case_index}" | cut -c1-63)"
    table_name="$(printf '%s' "${case_name}_amem_${case_index}" | tr -c '[:alnum:]_' '_' | tr '[:upper:]' '[:lower:]' | cut -c1-55)"

    mkdir -p "${GENERATED_CONFIG_DIR}" "${RESULT_DIR_ABS}"
    rm -f "${output_abs}"

    "${PYTHON_BIN}" "${PREP_SCRIPT}" \
        --source-config "${base_config}" \
        --target-config "${target_config}" \
        --llm-model "${MODEL_NAME}" \
        --evaluate-llm "${EVALUATE_MODEL}" \
        --agent-count "${TEAM_SIZE}" \
        --output-path "${output_rel}" \
        --embedding-model "${EMBEDDING_MODEL}" \
        --retrieval-top-k "${RETRIEVAL_TOP_K}" \
        --max-memory-context "${MAX_MEMORY_CONTEXT}" \
        --link-threshold "${LINK_THRESHOLD}" \
        --max-iterations "${MAX_ITERATIONS}"

    load_env_file

    echo "[amem-database] Running case ${case_index}: ${case_name}"
    if ! (
        cd "${ROOT_DIR}/marble"
        DB_NAME="${db_name}" \
        DB_HOST="localhost" \
        DB_USER="test" \
        DB_PASSWORD="Test123_456" \
        DB_POSTGRES_PORT="${postgres_port}" \
        DB_PROMETHEUS_PORT="${prometheus_port}" \
        DB_NODE_EXPORTER_PORT="${node_exporter_port}" \
        DB_PG_EXPORTER_PORT="${pg_exporter_port}" \
        DB_COMPOSE_PROJECT_NAME="${compose_project_name}" \
        COMPOSE_PROJECT_NAME="${compose_project_name}" \
        DB_TABLE_NAME="${table_name}" \
        PYTHONPATH=.. "${PYTHON_BIN}" "${MAIN_SCRIPT}" --config_path "${config_rel}"
    ); then
        cleanup_case_stack "${compose_project_name}"
        echo "[amem-database] Failed case ${case_index}: ${case_name}" >&2
        return 1
    fi

    cleanup_case_stack "${compose_project_name}"

    "${PYTHON_BIN}" - "${output_abs}" "${case_index}" "${case_name}" "${RUN_TAG}" <<'PY'
import json
import sys

path, case_index, case_name, run_tag = sys.argv[1:5]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
obj["case_id"] = int(case_index)
obj["case_name"] = case_name
obj["run_tag"] = run_tag
with open(path, "w", encoding="utf-8") as f:
    json.dump(obj, f)
PY

    echo "[amem-database] Completed case ${case_index}: ${case_name}"
}

if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "Config directory does not exist: ${CONFIG_DIR}" >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file does not exist: ${ENV_FILE}" >&2
    exit 1
fi

validate_parallelism
mapfile -t CONFIGS < <(collect_configs)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
    echo "No configs found in ${CONFIG_DIR} matching ${CONFIG_GLOB}" >&2
    exit 1
fi

max_port_span=$(( ${#CONFIGS[@]} * PORT_STRIDE + 10 ))
if (( RUN_PORT_BLOCK_SIZE <= max_port_span )); then
    echo "RUN_PORT_BLOCK_SIZE=${RUN_PORT_BLOCK_SIZE} is too small for ${#CONFIGS[@]} configs with PORT_STRIDE=${PORT_STRIDE}." >&2
    exit 1
fi

validate_case_bounds "${#CONFIGS[@]}"
RUN_PORT_OFFSET_VALUE="$(compute_run_port_offset)"

echo "Starting A-Mem database run"
echo "Model: ${MODEL_NAME}"
echo "Run tag: ${RUN_TAG}"
echo "Config directory: ${CONFIG_DIR}"
echo "Generated configs: ${GENERATED_CONFIG_DIR}"
echo "Result directory: ${RESULT_DIR_ABS}"
echo "Evaluation model: ${EVALUATE_MODEL}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Task parallelism: ${TASK_PARALLELISM}"
echo "Skip completed: ${SKIP_COMPLETED}"
echo "Team size: ${TEAM_SIZE}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Run port offset: ${RUN_PORT_OFFSET_VALUE}"
if [[ -n "${CASE_IDS}" ]]; then
    echo "Case IDs: ${CASE_IDS}"
else
    echo "Case range: ${CASE_START}-${CASE_END}"
fi

failures=0
active_jobs=0
while IFS= read -r case_index; do
    run_case "${case_index}" &
    active_jobs=$((active_jobs + 1))

    if (( active_jobs >= TASK_PARALLELISM )); then
        if ! wait -n; then
            failures=$((failures + 1))
        fi
        active_jobs=$((active_jobs - 1))
    fi
done < <(pending_case_sequence)

while (( active_jobs > 0 )); do
    if ! wait -n; then
        failures=$((failures + 1))
    fi
    active_jobs=$((active_jobs - 1))
done

if (( failures > 0 )); then
    echo "A-Mem database run completed with ${failures} case failure(s)." >&2
    exit 1
fi

echo "A-Mem database run completed successfully."
