# Lifelong Runner

The generic lifelong runner supports three modes:

- `baseline`
- `llmamem`
- `amem`

Default remains:

```bash
python scripts/lifelong/run_lifelong_experiment.py --modes "baseline,llmamem"
```

A-MEM only example:

```bash
python scripts/lifelong/run_lifelong_experiment.py \
  --task_type coding \
  --config_dir marble/configs/coding_configs \
  --config_pattern "config_{task_id}.yaml" \
  --task_start 1 --task_end 10 \
  --modes "amem"
```

Three-way comparison example:

```bash
python scripts/lifelong/run_lifelong_experiment.py \
  --task_type coding \
  --config_dir marble/configs/coding_configs \
  --config_pattern "config_{task_id}.yaml" \
  --task_start 1 --task_end 10 \
  --modes "baseline,llmamem,amem"
```

A-MEM options:

- `--amem_topology {local,shared}`
- `--amem_evolution_enabled {true,false}`
- `--amem_evolution_threshold INT`
- `--amem_embedding_model MODEL_NAME`
- `--amem_llm_backend {openai,ollama}`
- `--amem_llm_model MODEL_NAME`
- `--amem_retrieval_k INT`
