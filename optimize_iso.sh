#!/bin/bash
# Simple optimization script for ISO agent only

# Exit on error
set -e

# Create output directories
mkdir -p logs/optimize/iso

# Ensure environment is on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set optimization parameters
N_TRIALS=5
N_TIMESTEPS=1000
EVAL_FREQ=100
EVAL_EPISODES=48

echo "Starting ISO agent optimization..."

# Optimize ISO agent
python -m rl_zoo3.train \
  --algo ppo \
  --env ISO-RLZoo-v0 \
  --gym-packages energy_net.env.register_envs \
  --n-timesteps $N_TIMESTEPS \
  -optimize \
  --n-trials $N_TRIALS \
  --n-jobs 1 \
  --sampler tpe \
  --pruner median \
  --eval-episodes $EVAL_EPISODES \
  --eval-freq $EVAL_FREQ \
  --log-folder logs/optimize/iso \
  --tensorboard-log logs/optimize/iso/tensorboard \
  --study-name iso_optimization \
  --storage sqlite:///logs/optimize/iso/study.db \
  --optimization-log-path logs/optimize/iso/optimization.log \
  --conf configs/ppo_iso.yml

echo "ISO optimization complete! Best parameters will be in logs/optimize/iso/study.db"
echo "Run train_rlzoo_direct.sh to start training with the optimized parameters." 