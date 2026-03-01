SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# set -a && source "${SCRIPT_DIR}/../../.env" && set +a

WORKSPACE_DIR="marble/workspace"
UPDATE_SCRIPT="scripts/coding/utils/update_coding_config.py"
RUN_DEMO_SCRIPT="marble/run_demo.sh"

model_name="bedrock/converse/qwen.qwen3-32b-v1:0"
safe_model_name=$(echo ${model_name} | tr '/' '_')
LOG_DIR="marble/logs/${safe_model_name}"

rm -rf marble/memory_store

mkdir -p ${LOG_DIR}

for id in {1..100}; do
    echo "Processing task with ID=$id..."
    rm -rf ${WORKSPACE_DIR}/*
    python ${UPDATE_SCRIPT} --benchmark_id ${id} --llm_model "${model_name}"
    echo "Running the demo script..."
    (cd marble && PYTHONPATH=.. python main.py --config_path ./configs/coding_config/coding_config.yaml)
    echo "Saving solution file..."
    cp ${WORKSPACE_DIR}/solution.py ${LOG_DIR}/solution_${id}.py
    echo "Task with ID=$id completed."
    echo "==============================="
done

echo "All tasks have been processed!"
