#!/bin/bash

# Source the marble conda environment
eval "$(conda shell.bash hook)"
conda activate marble

# Navigate to MARBLE directory and run the simulation
cd "$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
python -m marble.environments.werewolf_env "$@"
