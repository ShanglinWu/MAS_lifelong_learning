#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PREP_SCRIPT="${ROOT_DIR}/scripts/coding/utils/prepare_amem_coding_config.py"
MAIN_SCRIPT="${ROOT_DIR}/marble/main.py"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_NAME="${MODEL_NAME:-bedrock/converse/qwen.qwen3-32b-v1:0}"
TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-100}"
TASK_IDS="${TASK_IDS:-}"
RUN_TAG="${RUN_TAG:-amem_$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_CONFIG_DIR="${SOURCE_CONFIG_DIR:-${ROOT_DIR}/marble/configs/coding_configs}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bedrock/amazon.titan-embed-text-v2:0}"
EVALUATE_LLM="${EVALUATE_LLM:-bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-3}"
MAX_MEMORY_CONTEXT="${MAX_MEMORY_CONTEXT:-5}"
LINK_THRESHOLD="${LINK_THRESHOLD:-0.72}"
APPEND_RESULTS="${APPEND_RESULTS:-0}"
TASK_PARALLELISM="${TASK_PARALLELISM:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

safe_model_name="$(echo "${MODEL_NAME}" | tr '/:' '__')"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${ROOT_DIR}/marble/configs/coding_config/generated_amem/${safe_model_name}/${RUN_TAG}}"
FINAL_OUTPUT_REL="result/coding/${safe_model_name}/amem/development_output_${RUN_TAG}.jsonl"
FINAL_OUTPUT_ABS="${ROOT_DIR}/marble/${FINAL_OUTPUT_REL}"
TASK_OUTPUT_DIR_REL="result/coding/${safe_model_name}/amem/${RUN_TAG}/tasks"
TASK_OUTPUT_DIR_ABS="${ROOT_DIR}/marble/${TASK_OUTPUT_DIR_REL}"

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

task_to_ids = {}
for name in sorted(os.listdir(source_dir)):
    match = re.fullmatch(r"config_(\d+)\.yaml", name)
    if not match:
        continue
    task_id = int(match.group(1))
    path = os.path.join(source_dir, name)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f)
    task_text = normalize_task((cfg or {}).get("task", {}).get("content", ""))
    if task_text:
        task_to_ids.setdefault(task_text, []).append(task_id)

used_ids = set()
recovered = 0
with open(merged_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        obj = json.loads(line)
        task_text = normalize_task(obj.get("task", ""))
        candidate_ids = task_to_ids.get(task_text, [])
        chosen_id = None
        for task_id in candidate_ids:
            if task_id not in used_ids:
                chosen_id = task_id
                break
        if chosen_id is None:
            continue
        used_ids.add(chosen_id)
        shard_path = os.path.join(tasks_dir, f"task_{chosen_id}.jsonl")
        with open(shard_path, "w", encoding="utf-8") as shard:
            shard.write(line + "\n")
        recovered += 1

print(f"[amem] recovered {recovered} task shard(s) from merged output", file=sys.stderr)
PY
}

pending_task_sequence() {
    while IFS= read -r task_id; do
        if [[ "${SKIP_COMPLETED}" == "1" && -f "${TASK_OUTPUT_DIR_ABS}/task_${task_id}.jsonl" ]]; then
            echo "[amem] Skipping completed task ${task_id}" >&2
            continue
        fi
        echo "${task_id}"
    done < <(task_sequence)
}

run_task() {
    local task_id="$1"
    local source_config="${SOURCE_CONFIG_DIR}/config_${task_id}.yaml"
    local target_config="${GENERATED_CONFIG_DIR}/config_${task_id}.yaml"
    local config_rel="./configs/coding_config/generated_amem/${safe_model_name}/${RUN_TAG}/config_${task_id}.yaml"
    local workspace_rel="workspace_${safe_model_name}/amem/${RUN_TAG}/task_${task_id}"
    local workspace_abs="${ROOT_DIR}/marble/${workspace_rel}"
    local output_rel="${TASK_OUTPUT_DIR_REL}/task_${task_id}.jsonl"
    local output_abs="${ROOT_DIR}/marble/${output_rel}"
    local solution_log_dir="${ROOT_DIR}/marble/logs/${safe_model_name}/amem/${RUN_TAG}"

    if [[ ! -f "${source_config}" ]]; then
        echo "Source config missing for task ${task_id}: ${source_config}" >&2
        return 1
    fi

    mkdir -p "${workspace_abs}" "$(dirname "${output_abs}")" "${solution_log_dir}" "${GENERATED_CONFIG_DIR}"
    rm -f "${output_abs}"

    "${PYTHON_BIN}" "${PREP_SCRIPT}" \
        --source-config "${source_config}" \
        --target-config "${target_config}" \
        --llm-model "${MODEL_NAME}" \
        --evaluate-llm "${EVALUATE_LLM}" \
        --workspace-dir "${workspace_rel}" \
        --output-path "${output_rel}" \
        --embedding-model "${EMBEDDING_MODEL}" \
        --retrieval-top-k "${RETRIEVAL_TOP_K}" \
        --max-memory-context "${MAX_MEMORY_CONTEXT}" \
        --link-threshold "${LINK_THRESHOLD}"

    rm -rf "${workspace_abs:?}"/*
    load_bedrock_token_from_env

    (
        cd "${ROOT_DIR}/marble"
        PYTHONPATH=.. "${PYTHON_BIN}" "${MAIN_SCRIPT}" --config_path "${config_rel}"
    )

    if [[ -f "${workspace_abs}/solution.py" ]]; then
        cp "${workspace_abs}/solution.py" "${solution_log_dir}/solution_${task_id}.py"
    else
        echo "[amem] warning: solution.py missing for task ${task_id}" >&2
    fi

    echo "[amem] Task ${task_id} completed."
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

    echo "[amem] Aggregated task outputs into ${FINAL_OUTPUT_ABS}"
}

clear_selected_task_outputs() {
    mkdir -p "${TASK_OUTPUT_DIR_ABS}"
    while IFS= read -r task_id; do
        rm -f "${TASK_OUTPUT_DIR_ABS}/task_${task_id}.jsonl"
    done < <(task_sequence)
}

if [[ ! -d "${SOURCE_CONFIG_DIR}" ]]; then
    echo "Source config directory does not exist: ${SOURCE_CONFIG_DIR}" >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file does not exist: ${ENV_FILE}" >&2
    exit 1
fi

echo "Starting A-Mem coding run"
echo "Model: ${MODEL_NAME}"
echo "Run tag: ${RUN_TAG}"
echo "Source configs: ${SOURCE_CONFIG_DIR}"
echo "Generated configs: ${GENERATED_CONFIG_DIR}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Evaluation model: ${EVALUATE_LLM}"
echo "Retrieval top-k: ${RETRIEVAL_TOP_K}"
echo "Max memory context: ${MAX_MEMORY_CONTEXT}"
echo "Link threshold: ${LINK_THRESHOLD}"
echo "Task parallelism: ${TASK_PARALLELISM}"
echo "Skip completed: ${SKIP_COMPLETED}"
if [[ -n "${TASK_IDS}" ]]; then
    echo "Task IDs: ${TASK_IDS}"
else
    echo "Task range: ${TASK_START}-${TASK_END}"
fi

mkdir -p "$(dirname "${FINAL_OUTPUT_ABS}")" "${TASK_OUTPUT_DIR_ABS}"
recover_task_outputs_from_merged

if [[ "${APPEND_RESULTS}" != "1" ]]; then
    rm -f "${FINAL_OUTPUT_ABS}"
    if [[ "${SKIP_COMPLETED}" != "1" ]]; then
        clear_selected_task_outputs
    fi
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
    echo "A-Mem coding run completed with ${failures} task failure(s)." >&2
    exit 1
fi

echo "A-Mem coding run completed successfully."
