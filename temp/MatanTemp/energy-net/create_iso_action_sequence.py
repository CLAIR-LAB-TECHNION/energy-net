#!/usr/bin/env python3
"""
Generate Predefined ISO Action Sequences

This script creates and saves sequences of predefined ISO actions that can be used
with the PreDefinedISOWrapper during PCS training. It supports multiple methods
of generating actions:
1. Random actions
2. From a trained policy
3. Custom patterns (e.g., time-of-day schedules or YAML-configured)

The generated action sequences are saved as .npy files that can be loaded
during training using the --iso-action-file parameter.
"""

import os
import argparse
import numpy as np
import yaml
import gymnasium as gym
from stable_baselines3 import PPO, TD3
from sb3_contrib import RecurrentPPO
from energy_net.envs import EnergyNetV0  # for raw action_space bounds

# Import our custom env builder
import energy_net.envs.register_envs
from energy_net.controllers.alternating_wrappers import make_iso_env


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predefined ISO action sequences")
    parser.add_argument("--use-dispatch", action="store_true",
                        help="Include dispatch in ISO action sequence (adds dispatch dimension)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to save the generated action sequence (.npy file)")
    parser.add_argument("--sequence-length", type=int, default=10000,
                        help="Length of the action sequence to generate")
    parser.add_argument("--method", type=str, choices=["random", "from_policy", "pattern", "yaml_config"],
                        default="random", help="Method to generate actions")
    parser.add_argument("--policy-path", type=str, default=None,
                        help="Path to the pre-trained ISO policy model (required for from_policy)")
    parser.add_argument("--policy-type", type=str, choices=["ppo", "recurrent_ppo", "td3"],
                        default="ppo", help="Type of the pre-trained policy")
    parser.add_argument("--demand-pattern", type=str, default="CONSTANT",
                        choices=["CONSTANT", "SINUSOIDAL", "DOUBLE_PEAK", "TWO_PEAK", "DATA_DRIVEN"],
                        help="Demand pattern for environment creation")
    parser.add_argument("--demand-data", type=str, default=None,
                        help="Path to demand data YAML file (required for DATA_DRIVEN pattern)")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE",
                        choices=["ONLINE", "CONSTANT", "QUADRATIC", "INTERVALS", "QUADRATIC_INTERVALS", "SMP"],
                        help="Pricing policy for environment creation")
    parser.add_argument("--cost-type", type=str, default="CONSTANT",
                        choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                        help="Cost type for environment creation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pattern-type", type=str, choices=["charge_discharge_cycle", "price_responsive", "yaml_config"],
                        default="charge_discharge_cycle", help="Type of pattern to generate")
    parser.add_argument("--cycle-length", type=int, default=48,
                        help="Length of a single cycle for pattern generation (default 48 = 1 day)")
    parser.add_argument("--pattern-config", type=str, default=None,
                        help="Path to YAML file defining time-of-day action mapping (required for yaml_config)")
    return parser.parse_args()


def generate_random_actions(sequence_length, seed=42, env_config=None):
    np.random.seed(seed)
    # Instantiate a dummy env to inspect ISO action space (with or without dispatch)
    env = make_iso_env(
        steps_per_iteration=1,
        cost_type=env_config["cost_type"],
        pricing_policy=env_config["pricing_policy"],
        demand_pattern=env_config["demand_pattern"],
        seed=seed,
        use_dispatch_action=env_config.get("use_dispatch", False)
    )
    action_shape = env.action_space.shape
    env.close()
    # Uniform random in [-1,1] for each ISO action dim
    actions = np.random.uniform(-1, 1, size=(sequence_length, *action_shape))
    return actions


def generate_actions_from_policy(policy_path, policy_type, env_config, sequence_length, seed=42):
    env = make_iso_env(
        steps_per_iteration=sequence_length,
        cost_type=env_config["cost_type"],
        pricing_policy=env_config["pricing_policy"],
        demand_pattern=env_config["demand_pattern"],
        seed=seed,
        use_dispatch_action=env_config.get("use_dispatch", False)
    )
    # Load the policy
    if policy_type == "recurrent_ppo":
        model = RecurrentPPO.load(policy_path)
    elif policy_type == "td3":
        model = TD3.load(policy_path)
    else:
        model = PPO.load(policy_path)

    actions = []
    obs, _ = env.reset(seed=seed)
    done = False
    lstm_states = None
    while len(actions) < sequence_length:
        if policy_type == "recurrent_ppo":
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=[done], deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env.reset()
            lstm_states = None
    env.close()
    return np.array(actions[:sequence_length])


def generate_pattern_actions(pattern_type, sequence_length, cycle_length=48, seed=42, pattern_config=None, env_config=None):
    """Generate actions following a predefined pattern"""
    np.random.seed(seed)
    # Inspect ISO action dims by creating raw env to get true bounds
    if env_config is None:
        raise ValueError("env_config must be provided to generate pattern-based ISO actions")
    raw_env = EnergyNetV0(
        cost_type=env_config["cost_type"],
        pricing_policy=env_config["pricing_policy"],
        demand_pattern=env_config["demand_pattern"],
        num_pcs_agents=1,
        dispatch_config={"use_dispatch_action": env_config.get("use_dispatch", False), "dispatch_strategy": "PROPORTIONAL"},
        demand_data_path=env_config.get("demand_data_path")
    )
    iso_space = raw_env.action_space["iso"]
    low = iso_space.low
    high = iso_space.high
    action_shape = iso_space.shape
    n_dims = action_shape[0]
    raw_env.close()

    if pattern_type == "charge_discharge_cycle":
        # Simple pattern: charge for half cycle, discharge for half cycle
        np_lines = []
        cycles_needed = (sequence_length + cycle_length - 1) // cycle_length
        pattern = np.concatenate([
            np.ones(cycle_length // 2) * -0.7,
            np.ones(cycle_length // 2) * 0.7
        ])
        pattern_vals = np.tile(pattern, cycles_needed)[:sequence_length]
        noise = np.random.normal(0, 0.1, size=sequence_length)
        pattern_vals = np.clip(pattern_vals + noise, -1, 1)
        # Broadcast scalar pattern to full action vector
        actions = np.tile(pattern_vals.reshape(-1,1), (1, n_dims))
    elif pattern_type == "price_responsive":
        # More complex pattern: estimate price cycles and respond accordingly
        np_lines = []
        cycles_needed = (sequence_length + cycle_length - 1) // cycle_length
        third = cycle_length // 3
        pattern = np.concatenate([
            np.ones(third) * -0.8,
            np.ones(third) * 0.9,
            np.ones(third) * 0.3
        ])
        pattern_vals = np.tile(pattern, cycles_needed)[:sequence_length]
        noise = np.random.normal(0, 0.15, size=sequence_length)
        pattern_vals = np.clip(pattern_vals + noise, -1, 1)
        actions = np.tile(pattern_vals.reshape(-1,1), (1, n_dims))
    elif pattern_type == "yaml_config":
        # Load actions/time mapping from YAML
        if pattern_config is None:
            raise ValueError("pattern_config must be provided for yaml_config pattern")
        with open(pattern_config) as f:
            cfg = yaml.safe_load(f)
        # Build scalar schedule then place into action vector
        scalar_actions = np.zeros(sequence_length)
        for entry in cfg.get("actions", []):
            time_str = entry["time"]  # format 'HH:MM'
            hour, minute = map(int, time_str.split(':'))
            step = int(hour * (cycle_length / 24) + minute * (cycle_length / 24) / 60)
            for idx in range(step, sequence_length, cycle_length):
                scalar_actions[idx] = entry["action"]
        # Forward-fill missing values to carry previous setting
        last_val = scalar_actions[0]
        for i in range(sequence_length):
            if scalar_actions[i] != 0:
                last_val = scalar_actions[i]
            else:
                scalar_actions[i] = last_val
        # Place raw MWh dispatch values into the dispatch dimension, or broadcast scalar across dims
        if env_config.get("use_dispatch", False) and n_dims > 1:
            dispatch_idx = n_dims - 1
            actions = np.zeros((sequence_length, n_dims), dtype=np.float32)
            actions[:, dispatch_idx] = scalar_actions
        else:
            actions = np.tile(scalar_actions.reshape(-1,1), (1, n_dims))
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    return actions


def main():
    args = parse_args()
    if args.method == "from_policy" and not args.policy_path:
        print("ERROR: --policy-path must be set for from_policy method.")
        return

    # Build environment config including dispatch flag
    from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
    from energy_net.market.pricing.pricing_policy import PricingPolicy
    from energy_net.market.pricing.cost_types import CostType
    
    if args.demand_pattern == "DATA_DRIVEN" and not args.demand_data:
        print("ERROR: DATA_DRIVEN demand pattern requires --demand-data")
        return
    
    demand_pattern = DemandPattern[args.demand_pattern]
    pricing_policy = PricingPolicy[args.pricing_policy]
    cost_type = CostType[args.cost_type]

    env_config = {
        'cost_type': cost_type,
        'pricing_policy': pricing_policy,
        'demand_pattern': demand_pattern,
        'demand_data_path': args.demand_data,
        'use_dispatch': args.use_dispatch
    }

    if args.method == "random":
        actions = generate_random_actions(args.sequence_length, args.seed, env_config)
    elif args.method == "from_policy":
        actions = generate_actions_from_policy(
            args.policy_path, args.policy_type, env_config, args.sequence_length, args.seed
        )
    elif args.method == "pattern":
        actions = generate_pattern_actions(
            args.pattern_type, args.sequence_length, args.cycle_length, args.seed,
            args.pattern_config, env_config
        )
    elif args.method == "yaml_config":
        actions = generate_pattern_actions(
            "yaml_config", args.sequence_length, args.cycle_length, args.seed,
            args.pattern_config, env_config
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    np.save(args.output_file, actions)
    print(f"Saved {len(actions)} ISO actions to {args.output_file}")
    print(f"Stats: mean={actions.mean():.3f}, std={actions.std():.3f}, min={actions.min()}, max={actions.max()}")


if __name__ == "__main__":
    main() 