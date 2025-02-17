#!/bin/bash

# Enable debug output and error handling
set -x
set -e

# Required paths and directories setup
MODELS_DIR="models"
PCS_MODEL_DIR="${MODELS_DIR}/agent_pcs"
PCS_FINAL_MODEL="${PCS_MODEL_DIR}/agent_pcs_final.zip"
PCS_NORMALIZER="${PCS_MODEL_DIR}/agent_pcs_normalizer.pkl"

# Create model directory
mkdir -p ${PCS_MODEL_DIR}

# First run training with pcs_game_main.py
echo "Starting PCS training..."
python pcs_game_main.py \
    --algo_type "PPO" \
    --demand_pattern "SINUSOIDAL" \
    --cost_type "CONSTANT" \
    --total_iterations 2 \
    --train_timesteps_per_iteration 1 \
    --eval_episodes 5 \
    --seed 42

TRAIN_EXIT_CODE=$?

# If training succeeded, run evaluation
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "PCS training completed. Starting evaluation..."
    
    # Check if training outputs exist
    if [ ! -f ${PCS_FINAL_MODEL} ] || [ ! -f ${PCS_NORMALIZER} ]; then
        echo "Error: Training outputs not found"
        exit 1
    fi
    
    python eval_agent.py \
        --algo_type "PPO" \
        --trained_model_path "${PCS_FINAL_MODEL}" \
        --normalizer_path "${PCS_NORMALIZER}" \
        --demand_pattern "SINUSOIDAL" \
        --cost_type "CONSTANT" \
        --trained_pcs_model_path None \
        --eval_episodes 5 \
        --env_id "PCSUnitEnv-v0"
    
    EVAL_EXIT_CODE=$?
    if [ ${EVAL_EXIT_CODE} -eq 0 ]; then
        echo "Evaluation completed successfully"
    else
        echo "Evaluation failed with code ${EVAL_EXIT_CODE}"
        exit 1
    fi
else
    echo "Training failed with code ${TRAIN_EXIT_CODE}"
    exit 1
fi
