#!/usr/bin/env python3
"""
Generate Predefined PCS Action Sequences

This script creates and saves sequences of predefined PCS actions that can be used
with the PreDefinedPCSWrapper during ISO training. It supports multiple methods
of generating actions:
1. Random actions
2. From a trained policy
3. Custom patterns (e.g., charge during low prices, discharge during high prices)

The generated action sequences are saved as .npy files that can be loaded
during training using the --pcs-action-file parameter.
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, TD3
from sb3_contrib import RecurrentPPO
import yaml

# Import our custom env and wrappers
import energy_net.envs.register_envs
from energy_net.controllers.alternating_wrappers import make_pcs_env
from energy_net.envs import EnergyNetV0

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate predefined PCS action sequences")
    
    # Output options
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to save the generated action sequence (.npy file)")
    parser.add_argument("--sequence-length", type=int, default=10000,
                        help="Length of the action sequence to generate")
    
    # Generation method
    parser.add_argument("--method", type=str, choices=["random", "from_policy", "pattern"],
                        default="random", help="Method to generate actions")
    
    # For policy-based generation
    parser.add_argument("--policy-path", type=str, 
                        help="Path to the pre-trained policy model (required for from_policy method)")
    parser.add_argument("--policy-type", type=str, choices=["ppo", "recurrent_ppo", "td3"],
                        default="ppo", help="Type of the pre-trained policy")
    
    # Environment parameters (for policy-based generation)
    parser.add_argument("--demand-pattern", type=str, default="CONSTANT", 
                        choices=["CONSTANT", "SINUSOIDAL", "DOUBLE_PEAK", "TWO_PEAK", "DATA_DRIVEN"],
                        help="Demand pattern for environment creation")
    parser.add_argument("--demand-data", type=str, default=None,
                        help="Path to demand data YAML file (required for DATA_DRIVEN pattern)")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE", 
                        choices=["ONLINE", "CONSTANT", "QUADRATIC", "INTERVALS"],
                        help="Pricing policy for environment creation")
    parser.add_argument("--cost-type", type=str, default="CONSTANT", 
                        choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                        help="Cost type for environment creation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # For pattern-based generation
    parser.add_argument("--pattern-type", type=str, choices=["charge_discharge_cycle", "price_responsive", "yaml_config"],
                        default="charge_discharge_cycle", help="Type of pattern to generate")
    parser.add_argument("--cycle-length", type=int, default=48,
                        help="Length of a single cycle for pattern generation (default 48 = 1 day)")
    parser.add_argument("--pattern-config", type=str, default=None,
                        help="Path to YAML file defining time-of-day action mapping (required for yaml_config pattern)")
    
    return parser.parse_args()

def generate_random_actions(sequence_length, seed=42):
    """Generate random PCS actions within valid bounds"""
    np.random.seed(seed)
    # PCS action space is typically [-1, 1] for charge/discharge
    actions = np.random.uniform(-1, 1, size=sequence_length)
    return actions

def generate_actions_from_policy(policy_path, policy_type, env_config, sequence_length, seed=42):
    """Generate actions by running a pre-trained policy"""
    # Create environment for policy evaluation
    env = make_pcs_env(
        steps_per_iteration=sequence_length,
        cost_type=env_config["cost_type"],
        pricing_policy=env_config["pricing_policy"],
        demand_pattern=env_config["demand_pattern"],
        seed=seed,
        log_dir="./logs/temp",
        demand_data_path=env_config.get("demand_data_path")
    )
    
    # Load the policy
    if policy_type == "recurrent_ppo":
        policy = RecurrentPPO.load(policy_path)
    elif policy_type == "td3":
        policy = TD3.load(policy_path)
    else:  # default to PPO
        policy = PPO.load(policy_path)
    
    # Generate actions
    actions = []
    obs, _ = env.reset(seed=seed)
    done = False
    episode_steps = 0
    
    print(f"Generating {sequence_length} actions using pre-trained policy...")
    
    while len(actions) < sequence_length:
        # For recurrent policies, we need to handle the lstm_states
        if policy_type == "recurrent_ppo":
            # Create zero lstm states for first step
            if episode_steps == 0:
                lstm_states = (np.zeros(policy.policy.lstm_state_shape), 
                              np.zeros(policy.policy.lstm_state_shape))
                dones = np.array([False])
            
            # Get action from policy
            action, lstm_states = policy.predict(
                observation=obs,
                state=lstm_states,
                episode_start=dones,
                deterministic=True
            )
        else:
            # Standard policy prediction
            action, _ = policy.predict(obs, deterministic=True)
        
        # Record the action
        actions.append(action)
        
        # Step the environment
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_steps += 1
        
        # Reset if done
        if done:
            obs, _ = env.reset()
            episode_steps = 0
            dones = np.array([True])
            print(f"Episode completed. Generated {len(actions)} actions so far.")
    
    env.close()
    return np.array(actions[:sequence_length])

def generate_pattern_actions(pattern_type, sequence_length, cycle_length=48, seed=42, pattern_config=None):
    """Generate actions following a predefined pattern"""
    np.random.seed(seed)
    
    if pattern_type == "charge_discharge_cycle":
        # Simple pattern: charge for half cycle, discharge for half cycle
        cycles_needed = (sequence_length + cycle_length - 1) // cycle_length
        pattern = np.concatenate([
            np.ones(cycle_length // 2) * -0.7,  # Charge (negative values in our convention)
            np.ones(cycle_length // 2) * 0.7    # Discharge (positive values)
        ])
        
        # Repeat the pattern and add some noise
        actions = np.tile(pattern, cycles_needed)[:sequence_length]
        noise = np.random.normal(0, 0.1, size=sequence_length)
        actions = np.clip(actions + noise, -1, 1)
        
    elif pattern_type == "price_responsive":
        # More complex pattern: estimate price cycles and respond accordingly
        # Assuming typical day-night price patterns
        cycles_needed = (sequence_length + cycle_length - 1) // cycle_length
        
        # Create price-following behavior (simplified model)
        # This assumes lower prices at night (first third), higher during day (middle third),
        # and moderate in evening (last third)
        third = cycle_length // 3
        
        pattern = np.concatenate([
            np.ones(third) * 50,              # Night: charge heavily (low prices)
            np.ones(third) * -20,               # Day peak: discharge heavily (high prices)
            np.ones(third) * -10                # Evening: light discharge (moderate prices)
        ])
        
        # Repeat the pattern and add some noise
        actions = np.tile(pattern, cycles_needed)[:sequence_length]
        # Add more randomness to make it realistic
        noise = np.random.normal(0, 0.15, size=sequence_length) 
        actions = np.clip(actions + noise, -1, 1)
    elif pattern_type == "yaml_config":
        # Load actions/time mapping from YAML
        if pattern_config is None:
            raise ValueError("pattern_config must be provided for yaml_config pattern")
        with open(pattern_config) as f:
            cfg = yaml.safe_load(f)
        actions = np.zeros(sequence_length)
        # Map each entry's time to a step index and assign
        for entry in cfg.get("actions", []):
            time_str = entry["time"]  # format 'HH:MM'
            hour, minute = map(int, time_str.split(':'))
            # Convert time to cycle index
            step_in_cycle = int(hour * (cycle_length / 24) + minute * (cycle_length / 24) / 60)
            # Apply action at every cycle offset
            for idx in range(step_in_cycle, sequence_length, cycle_length):
                actions[idx] = entry["action"]
        # Forward-fill missing values
        last_val = actions[0]
        for i in range(sequence_length):
            if actions[i] != 0:
                last_val = actions[i]
            else:
                actions[i] = last_val
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    return actions

def main():
    args = parse_args()
    
    # Validate arguments based on method
    if args.method == "from_policy" and not args.policy_path:
        print("ERROR: The from_policy method requires a policy path to be specified.")
        print("Please set --policy-path to the location of your pre-trained agent.")
        return
    
    # Prepare environment configuration for policy-based generation
    if args.method == "from_policy":
        # Convert string pattern to enum
        if args.demand_pattern == "DATA_DRIVEN":
            from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
            if not args.demand_data:
                print("ERROR: DATA_DRIVEN demand pattern requires a demand data file.")
                print("Please specify the file path using --demand-data")
                return
            demand_pattern = DemandPattern.DATA_DRIVEN
        else:
            from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
            demand_pattern = DemandPattern[args.demand_pattern]
        
        env_config = {
            "cost_type": args.cost_type,
            "pricing_policy": args.pricing_policy,
            "demand_pattern": demand_pattern,
            "demand_data_path": args.demand_data
        }
    
    # Generate actions based on the selected method
    if args.method == "random":
        print(f"Generating {args.sequence_length} random actions...")
        actions = generate_random_actions(args.sequence_length, seed=args.seed)
    
    elif args.method == "from_policy":
        print(f"Generating {args.sequence_length} actions from policy {args.policy_path}...")
        actions = generate_actions_from_policy(
            args.policy_path,
            args.policy_type,
            env_config,
            args.sequence_length,
            seed=args.seed
        )
    
    elif args.method == "pattern":
        print(f"Generating {args.sequence_length} actions with pattern {args.pattern_type}...")
        actions = generate_pattern_actions(
            args.pattern_type,
            args.sequence_length,
            cycle_length=args.cycle_length,
            seed=args.seed,
            pattern_config=args.pattern_config
        )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Save the generated actions
    np.save(args.output_file, actions)
    print(f"Saved {len(actions)} actions to {args.output_file}")
    
    # Print some statistics about the generated actions
    print(f"Action statistics:")
    print(f"  Mean: {actions.mean():.3f}")
    print(f"  Std: {actions.std():.3f}")
    print(f"  Min: {actions.min():.3f}")
    print(f"  Max: {actions.max():.3f}")
    print(f"  First 5 actions: {actions[:5]}")

if __name__ == "__main__":
    main() 