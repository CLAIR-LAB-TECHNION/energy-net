#!/bin/bash
# Training script that uses RL-Zoo3 with direct callback

# Create output directories
mkdir -p logs/iso/ppo/run_1
mkdir -p logs/pcs/ppo/run_1
mkdir -p rl-baselines3-zoo/hyperparams/ppo

# Ensure our environment is on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create or update the hyperparameter files
cat > rl-baselines3-zoo/hyperparams/ppo/ISO-RLZoo-v0.yml << EOF
ISO-RLZoo-v0:
  env_wrapper:
    - gymnasium.wrappers.RescaleAction:
        min_action: -1.0
        max_action: 1.0
  
  normalize: true

  n_envs: 4
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  n_steps: 1024
  batch_size: 128
  gae_lambda: 0.95
  gamma: 0.99
  n_epochs: 10
  ent_coef: 0.8
  learning_rate: !!float 1e-3
  clip_range: 0.2
  max_grad_norm: 2.0
  vf_coef: 0.5
  
  policy_kwargs: "dict(log_std_init=-1,
                       ortho_init=True,
                       activation_fn=nn.Tanh,
                        net_arch=dict(shared=[64, 64], pi=[128, 64], vf=[128, 64]))"
    
  # Use our direct callback class
  callback: plot_callback.PlotCallback
EOF

cat > rl-baselines3-zoo/hyperparams/ppo/PCS-RLZoo-v0.yml << EOF
PCS-RLZoo-v0:
  env_wrapper:
    - gymnasium.wrappers.RescaleAction:
        min_action: -1.0
        max_action: 1.0
  
  normalize: true
  n_envs: 2
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  n_steps: 1024
  batch_size: 128
  gae_lambda: 0.95
  gamma: 0.99
  n_epochs: 10
  ent_coef: 0.4
  learning_rate: lin_1e-4
  clip_range: 0.2
  max_grad_norm: 1
  vf_coef: 0.5
  
  policy_kwargs: "dict(log_std_init=-1,
                       ortho_init=True,
                       activation_fn=nn.Tanh,
                        net_arch=dict(shared=[128, 128], pi=[128, 64], vf=[64, 64]))"
    
  # Use our direct callback class
  callback: plot_callback.PlotCallback
EOF

# Define environment kwargs
BASE_ENV_KWARGS=(
  "cost_type:'CONSTANT'"
  "pricing_policy:'QUADRATIC'"
  "demand_pattern:'SINUSOIDAL'"
  "use_dispatch_action:True"
)

# Set number of iterations for alternating training
ITERATIONS=3
# Set training steps per iteration
TIMESTEPS=2
# Set random seed
SEED=42

echo "Starting alternating training of ISO and PCS agents with $ITERATIONS iterations..."

# First iteration: Train ISO agent alone
ITERATION=1
echo "Iteration $ITERATION: Training ISO agent..."

ISO_ENV_KWARGS=("${BASE_ENV_KWARGS[@]}")

python -m rl_zoo3.train \
  --algo ppo \
  --env ISO-RLZoo-v0 \
  --gym-packages energy_net.env.register_envs \
  --eval-freq 500 \
  --eval-episodes 10 \
  --save-freq 500 \
  --log-folder logs/iso/ppo/run_1 \
  --tensorboard-log logs/iso/tensorboard/run_1 \
  --env-kwargs "${ISO_ENV_KWARGS[@]}" \
  --n-timesteps 1 \
  --seed $SEED \
  -conf rl-baselines3-zoo/hyperparams/ppo/ISO-RLZoo-v0.yml

# Define initial model paths
ISO_MODEL_PATH="logs/iso/ppo/run_1/ppo/ISO-RLZoo-v0_1/ISO-RLZoo-v0.zip"

# Check if ISO training succeeded
if [ ! -f "$ISO_MODEL_PATH" ]; then
  ISO_MODEL_PATH="logs/iso/ppo/run_1/ISO-RLZoo-v0.zip"
  if [ ! -f "$ISO_MODEL_PATH" ]; then
    echo "ERROR: ISO training failed. Model not found."
    exit 1
  fi
fi

# Now alternate between training PCS and ISO
for ((ITERATION=1; ITERATION<=$ITERATIONS; ITERATION++)); do
  # Train PCS agent with fixed ISO policy
  echo "Iteration $ITERATION: Training PCS agent with fixed ISO policy..."
  
  PCS_ENV_KWARGS=("${BASE_ENV_KWARGS[@]}")
  PCS_ENV_KWARGS+=("iso_policy_path:'$ISO_MODEL_PATH'")
  
  python -m rl_zoo3.train \
    --algo ppo \
    --env PCS-RLZoo-v0 \
    --gym-packages energy_net.env.register_envs \
    --eval-freq 500 \
    --eval-episodes 10 \
    --save-freq 500 \
    --log-folder logs/pcs/ppo/run_1 \
    --tensorboard-log logs/pcs/tensorboard/run_1 \
    --env-kwargs "${PCS_ENV_KWARGS[@]}" \
    --n-timesteps $TIMESTEPS \
    --seed $SEED \
    -conf rl-baselines3-zoo/hyperparams/ppo/PCS-RLZoo-v0.yml
  
  # Define PCS model path
  PCS_MODEL_PATH="logs/pcs/ppo/run_1/ppo/PCS-RLZoo-v0_1/PCS-RLZoo-v0.zip"
  if [ ! -f "$PCS_MODEL_PATH" ]; then
    PCS_MODEL_PATH="logs/pcs/ppo/run_1/PCS-RLZoo-v0.zip"
    if [ ! -f "$PCS_MODEL_PATH" ]; then
      echo "ERROR: PCS training failed. Model not found."
      exit 1
    fi
  fi
  
  # Skip ISO training in the last iteration
  if [ $ITERATION -eq $ITERATIONS ]; then
    break
  fi
  
  # Train ISO agent with fixed PCS policy
  echo "Iteration $((ITERATION+1)): Training ISO agent with fixed PCS policy..."
  
  ISO_ENV_KWARGS=("${BASE_ENV_KWARGS[@]}")
  ISO_ENV_KWARGS+=("pcs_policy_path:'$PCS_MODEL_PATH'")
  
  python -m rl_zoo3.train \
    --algo ppo \
    --env ISO-RLZoo-v0 \
    --gym-packages energy_net.env.register_envs \
    --eval-freq 500 \
    --eval-episodes 10 \
    --save-freq 500 \
    --log-folder logs/iso/ppo/run_1 \
    --tensorboard-log logs/iso/tensorboard/run_1 \
    --env-kwargs "${ISO_ENV_KWARGS[@]}" \
    --n-timesteps $TIMESTEPS \
    --seed $SEED \
    -conf rl-baselines3-zoo/hyperparams/ppo/ISO-RLZoo-v0.yml
  
  # Update ISO model path for next iteration
  ISO_MODEL_PATH="logs/iso/ppo/run_1/ppo/ISO-RLZoo-v0_1/ISO-RLZoo-v0.zip"
  if [ ! -f "$ISO_MODEL_PATH" ]; then
    ISO_MODEL_PATH="logs/iso/ppo/run_1/ISO-RLZoo-v0.zip"
    if [ ! -f "$ISO_MODEL_PATH" ]; then
      echo "ERROR: ISO training failed. Model not found."
      exit 1
    fi
  fi
done

echo "Alternating training completed!"
echo "Plots should be saved in the respective model directories." 