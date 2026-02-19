#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/coding/run_lifelong_experiment.py \
  --model "google.gemma-3-4b-it" \
  --task_start 1 \
  --task_end 10 \
  --modes "baseline,llmamem"

