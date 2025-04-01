#!/bin/bash
# Simple optimization script for PCS agent only

# Exit on error
set -e

# Create output directories
mkdir -p logs/optimize/pcs

# Ensure environment is on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set optimization parameters
N_TRIALS=5
N_TIMESTEPS=1000
EVAL_FREQ=100
EVAL_EPISODES=48

echo "Starting PCS agent optimization..."

# Optimize PCS agent
python -m rl_zoo3.train \
  --algo ppo \
  --env PCS-RLZoo-v0 \
  --gym-packages energy_net.env.register_envs \
  --n-timesteps $N_TIMESTEPS \
  -optimize \
  --n-trials $N_TRIALS \
  --n-jobs 1 \
  --sampler tpe \
  --pruner median \
  --eval-episodes $EVAL_EPISODES \
  --eval-freq $EVAL_FREQ \
  --log-folder logs/optimize/pcs \
  --tensorboard-log logs/optimize/pcs/tensorboard \
  --study-name pcs_optimization \
  --storage sqlite:///logs/optimize/pcs/study.db \
  --optimization-log-path logs/optimize/pcs/optimization.log \
  --conf configs/ppo_pcs.yml

echo "PCS optimization complete! Best parameters will be in logs/optimize/pcs/study.db"
echo "Run train_rlzoo_direct.sh to start training with the optimized parameters." 