#!/bin/bash
# Script for visualizing trained models using direct evaluation

# Ensure our environment is on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Default values
ISO_MODEL_PATH="logs/iso/ppo/run_1/ppo/ISO-RLZoo-v0_1/best_model.zip"
PCS_MODEL_PATH="logs/pcs/ppo/run_1/ppo/PCS-RLZoo-v0_1/best_model.zip"
EPISODES=5
RENDER=true
DETERMINISTIC=true

# Helper function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --iso-model PATH      Path to ISO model (default: $ISO_MODEL_PATH)"
  echo "  --pcs-model PATH      Path to PCS model (default: $PCS_MODEL_PATH)"
  echo "  --episodes N          Number of episodes to run (default: $EPISODES)"
  echo "  --no-render           Disable rendering"
  echo "  --stochastic          Use stochastic actions instead of deterministic"
  echo "  --help                Display this help message"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iso-model)
      ISO_MODEL_PATH="$2"
      shift 2
      ;;
    --pcs-model)
      PCS_MODEL_PATH="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --no-render)
      RENDER=false
      shift
      ;;
    --stochastic)
      DETERMINISTIC=false
      shift
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Check if models exist
if [ ! -f "$ISO_MODEL_PATH" ]; then
  echo "ERROR: ISO model not found at $ISO_MODEL_PATH"
  echo "Use --iso-model to specify the correct path"
  exit 1
fi

if [ ! -f "$PCS_MODEL_PATH" ]; then
  echo "ERROR: PCS model not found at $PCS_MODEL_PATH"
  echo "Use --pcs-model to specify the correct path"
  exit 1
fi

# Define environment kwargs
BASE_ENV_KWARGS=(
  "cost_type:'CONSTANT'"
  "pricing_policy:'ONLINE'"
  "demand_pattern:'CONSTANT'"
  "use_dispatch_action:True"
)

echo "=== Evaluating Combined ISO+PCS Agents ==="
echo "Using ISO model: $ISO_MODEL_PATH"
echo "Using PCS model: $PCS_MODEL_PATH"
echo "Episodes: $EPISODES"
echo "Environment settings:"
echo "  - Cost type: CONSTANT"
echo "  - Pricing policy: ONLINE"
echo "  - Demand pattern: CONSTANT"
echo "  - Dispatch action: Enabled"

# Create a temporary Python script to run combined evaluation
cat > tmp_combined_eval.py << EOF
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import energy_net.env.register_envs
from energy_net.env import EnergyNetV0
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enjoy_script")

# Load the models
logger.info(f"Loading ISO model from {repr('$ISO_MODEL_PATH')}")
iso_model = PPO.load("$ISO_MODEL_PATH")
logger.info(f"ISO model loaded, action space shape: {iso_model.action_space.shape}")

logger.info(f"Loading PCS model from {repr('$PCS_MODEL_PATH')}")
pcs_model = PPO.load("$PCS_MODEL_PATH")
logger.info(f"PCS model loaded, action space shape: {pcs_model.action_space.shape}")

# Create the environment
logger.info("Creating environment")
env = EnergyNetV0(
    pricing_policy="ONLINE",
    demand_pattern="CONSTANT",
    cost_type="CONSTANT",
    dispatch_config={
        "use_dispatch_action": True,
        "default_strategy": "PROPORTIONAL"
    }
)
logger.info(f"Environment created, ISO action space: {env.action_space['iso']}, PCS action space: {env.action_space['pcs']}")

# Run evaluation
episode_rewards = []
episode_lengths = []

for episode in range($EPISODES):
    logger.info(f"Starting episode {episode+1}/{$EPISODES}")
    obs_dict, info = env.reset()
    terminated = {"iso": False, "pcs": False}
    truncated = {"iso": False, "pcs": False}
    total_rewards = {"iso": 0, "pcs": 0}
    
    step = 0
    while not (terminated["iso"] or truncated["iso"] or terminated["pcs"] or truncated["pcs"]):
        # Get ISO action
        iso_action, _ = iso_model.predict(obs_dict["iso"], deterministic=${DETERMINISTIC/true/True})
        
        # Get PCS action
        pcs_action, _ = pcs_model.predict(obs_dict["pcs"], deterministic=${DETERMINISTIC/true/True})
        
        # Log the actions
        if step % 10 == 0:
            logger.info(f"Step {step}: ISO action shape {iso_action.shape}, PCS action shape {pcs_action.shape}")
        
        # Step the environment
        action_dict = {
            "iso": iso_action,
            "pcs": pcs_action
        }
        
        try:
            obs_dict, rewards, terminated, truncated, info = env.step(action_dict)
            
            total_rewards["iso"] += rewards["iso"]
            total_rewards["pcs"] += rewards["pcs"]
            
            step += 1
            
            # Render if enabled
            if ${RENDER/true/True}:
                env.render()
        except Exception as e:
            logger.error(f"Error during step: {e}")
            break
    
    episode_rewards.append(sum(total_rewards.values()))
    episode_lengths.append(step)
    
    print(f"Episode {episode+1}/{$EPISODES} completed")
    print(f"  Steps: {step}")
    print(f"  ISO reward: {total_rewards['iso']:.2f}")
    print(f"  PCS reward: {total_rewards['pcs']:.2f}")
    print(f"  Total reward: {sum(total_rewards.values()):.2f}")
    print()

# Print summary statistics
if episode_rewards:
    print("\nSummary Statistics:")
    print(f"  Mean episode reward: {np.mean(episode_rewards):.2f}")
    print(f"  Mean episode length: {np.mean(episode_lengths):.2f}")
    print(f"  Max episode reward: {np.max(episode_rewards):.2f}")
    print(f"  Min episode reward: {np.min(episode_rewards):.2f}")

env.close()
EOF

# Run the combined evaluation
python tmp_combined_eval.py

# Clean up
rm tmp_combined_eval.py

echo "Evaluation complete!" 