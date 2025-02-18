#!/bin/bash

# Enable debug output and error handling
set -x
set -e

# Required paths and directories setup
MODELS_DIR="models"
ISO_MODEL_DIR="${MODELS_DIR}/agent_iso"
ISO_FINAL_MODEL="${ISO_MODEL_DIR}/agent_iso_final.zip"
ISO_NORMALIZER="${ISO_MODEL_DIR}/agent_iso_normalizer.pkl"

# Create model directory
mkdir -p ${ISO_MODEL_DIR}

# First run training with iso_game_main.py
echo "Starting ISO training..."
python iso_game_main.py \
    --algo_type "PPO" \
    --trained_pcs_model_path None \
    --pricing_policy "QUADRATIC" \
    --demand_pattern "SINUSOIDAL" \
    --cost_type "CONSTANT" \
    --num_pcs_agents 1 \
    --total_iterations 400 \
    --train_timesteps_per_iteration 4000 \
    --eval_episodes 5 \
    --seed 24

TRAIN_EXIT_CODE=$?

# If training succeeded, run evaluation
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "ISO training completed. Starting evaluation..."
    
    # Check if training outputs exist
    if [ ! -f ${ISO_FINAL_MODEL} ] || [ ! -f ${ISO_NORMALIZER} ]; then
        echo "Error: Training outputs not found"
        exit 1
    fi
    
    python eval_agent.py \
        --algo_type "PPO" \
        --trained_pcs_model_path None \
        --trained_model_path "${ISO_FINAL_MODEL}" \
        --normalizer_path "${ISO_NORMALIZER}" \
        --pricing_policy "QUADRATIC" \
        --demand_pattern "SINUSOIDAL" \
        --cost_type "CONSTANT" \
        --num_pcs_agents 1 \
        --eval_episodes 5
    
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
