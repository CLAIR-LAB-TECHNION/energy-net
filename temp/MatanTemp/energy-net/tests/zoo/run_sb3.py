import importlib
import subprocess
import sys

def install_if_missing(package_name, import_name=None):
    try:
        importlib.import_module(import_name or package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Example usage:
install_if_missing("stable-baselines3[extra]", "stable_baselines3")
install_if_missing("sb3-contrib", "sb3_contrib")
install_if_missing("tensorboard")
install_if_missing("pettingzoo")


import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
from energy_net.envs.wrappers.stable_baselines_wrappers import StableBaselines3Wrapper
import os



def make_env():
    # Create the environment with required parameters
    env = EnergyNetV0(
        controller_name="EnergyNetController",
        controller_module="energy_net.controllers",
        pricing_policy=PricingPolicy.CONSTANT,  # Use constant pricing policy
        cost_type=CostType.CONSTANT,  # Use constant cost function
        demand_pattern=DemandPattern.SINUSOIDAL  # Use sinusoidal demand pattern
    )
    
    # Wrap the environment for Stable Baselines3
    env = StableBaselines3Wrapper(env)
    
    # Create logs directory
    log_dir = "../../logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Wrap with Monitor for episode stats
    env = Monitor(env, log_dir)
    
    return env

def main():
    # Create the environment
    env = make_env()
    
    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_energy_net_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Create evaluation environment
    eval_env = make_env()
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="../../logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(
        total_timesteps=1000000,
        callback=eval_callback
    )
    
    # Save the final model
    model.save("ppo_energy_net")
    
    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":

    main()