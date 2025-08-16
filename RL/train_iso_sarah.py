#!/usr/bin/env python3
"""
Direct Training Script for ISO Agent with Policy Gradient Methods

This script implements direct training of the ISO agent using PPO algorithms.
It supports both standard PPO and RecurrentPPO with LSTM policy.
It allows loading a pre-trained PCS policy to create a more realistic training
environment for the ISO.

Features:
- Focused training of ISO agent only
- Supports both standard PPO and recurrent (LSTM) policies
- Proper state handling for sequential data
- Environment normalization and vectorization
- Final evaluation of trained agent
- Option to use fixed PCS policy during training
- Option to use predefined PCS action sequence instead of a policy
"""

import os
import time
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import argparse
import logging
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, ActionNoise

# Import our custom env and wrappers
import energy_net.envs.register_envs
from RL.train_and_eval.setup import get_model
from energy_net.controllers.alternating_wrappers import make_iso_env
from energy_net.envs import EnergyNetV0
from energy_net.controllers.plot_callback import PlotCallback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("iso_trainer")


# Custom decaying action noise class
class LinearDecayActionNoise(ActionNoise):
    """Action noise with linear decay over time"""

    def __init__(self, mean, sigma, final_sigma=None, decay_steps=1000):
        """
        Constructor.

        Args:
            mean: Mean value for noise
            sigma: Initial standard deviation
            final_sigma: Final standard deviation after decay
            decay_steps: Number of steps for complete decay
        """
        self.mean = mean
        self.sigma = sigma
        self.final_sigma = final_sigma if final_sigma is not None else sigma * 0.2
        self.decay_steps = decay_steps
        self.current_step = 0
        self._dt = 1

    def __call__(self) -> np.ndarray:
        """Sample noise and update sigma value"""
        # Calculate current sigma using linear decay
        progress = min(1.0, self.current_step / self.decay_steps)
        current_sigma = self.sigma - progress * (self.sigma - self.final_sigma)

        # Log current sigma every 100 steps
        if self.current_step % 100 == 0:
            print(f"Current exploration sigma: {current_sigma:.4f} (step {self.current_step})")

        # Generate noise using current sigma
        noise = np.random.normal(self.mean, current_sigma)

        # Increment step counter
        self.current_step += 1

        return noise

    def reset(self) -> None:
        """Reset the action noise"""
        pass


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train ISO agent with PPO or RecurrentPPO")

    # Training parameters
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of training iterations")
    parser.add_argument("--timesteps", type=int, default=480,
                        help="Steps per iteration (480 = 10 days)")
    parser.add_argument("--seed", type=int, default=34,
                        help="Random seed")

    # Environment parameters
    parser.add_argument("--demand-pattern", type=str, default="CONSTANT",
                        choices=["CONSTANT", "SINUSOIDAL", "DOUBLE_PEAK", "TWO_PEAK", "DATA_DRIVEN"],
                        help="Demand pattern type")
    parser.add_argument("--demand-data", type=str, default=None,
                        help="Path to demand data YAML file (required for DATA_DRIVEN pattern)")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE",
                        choices=["ONLINE", "CONSTANT", "QUADRATIC", "INTERVALS", "QUADRATIC_INTERVALS", "DISPATCHONLY"],
                        help="Pricing policy type")
    parser.add_argument("--cost-type", type=str, default="CONSTANT",
                        choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                        help="Cost type")
    parser.add_argument("--use-dispatch", action="store_true",
                        help="Use dispatch action for ISO")

    # PCS behavior control
    parser.add_argument("--pcs-action-file", type=str, default=None,
                        help="Path to a numpy file (.npy) containing a sequence of predefined PCS actions")

    # Algorithm parameters
    parser.add_argument("--algorithm", type=str, choices=["ppo", "recurrent_ppo", "td3"], default="recurrent_ppo",
                        help="Algorithm to use: 'ppo' for standard PPO, 'recurrent_ppo' for RecurrentPPO with LSTM (default), or 'td3' for TD3")
    parser.add_argument("--lstm-size", type=int, default=64,
                        help="LSTM hidden size (only for RecurrentPPO)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (reduced for LSTM)")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy coefficient for exploration (only for PPO)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--net-arch", nargs="+", type=int, default=[64, 64],
                        help="Network architecture for policy network")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size (only for TD3)")
    parser.add_argument("--train-freq", type=int, default=2,
                        help="Update model every x steps (only for TD3)")
    parser.add_argument("--policy-noise", type=float, default=0.2,
                        help="Noise added to target policy (only for TD3)")
    parser.add_argument("--final-noise", type=float, default=0.05,
                        help="Final exploration noise after decay (only for TD3)")
    parser.add_argument("--noise-type", type=str, choices=["normal", "ou"], default="normal",
                        help="Type of action noise for TD3: 'normal' or 'ou' (Ornstein-Uhlenbeck)")

    # Paths
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for logs")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory for saved models")
    parser.add_argument("--plot-dir", type=str, default="plots",
                        help="Directory for plots")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes for evaluation")

    # Continue training from existing models
    parser.add_argument("--continue-from-best", action="store_true",
                        help="Continue training from the best ISO model saved previously")
    parser.add_argument("--initial-iso-model", type=str, default=None,
                        help="Path to initial ISO model for continued training")
    parser.add_argument("--pcs-model", type=str, default=None,
                        help="Path to PCS model to use during ISO training (optional)")
    parser.add_argument("--pcs-norm-path", type=str, default=None,
                        help="Path to VecNormalize stats for PCS model normalization (optional)")
    parser.add_argument("--start-iteration", type=int, default=1,
                        help="Starting iteration number when continuing training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate the best ISO model and exit")
    parser.add_argument("--best-model", type=str, default=None,
                        help="Path to the best ISO model for evaluation")
    parser.add_argument("--norm-path", type=str, default=None,
                        help="Path to VecNormalize stats file for evaluation")

    return parser.parse_args()


def create_recurrent_model(env, lstm_size, seed=None, **kwargs):
    """
    Create a RecurrentPPO model with appropriate hyperparameters.

    Args:
        env: The vectorized environment
        lstm_size: Size of LSTM hidden state
        seed: Random seed

    Returns:
        A RecurrentPPO model with LSTM policy
    """
    # Common kwargs with defaults that can be overridden
    model_kwargs = {
        "policy": "MlpLstmPolicy",
        "env": env,
        "learning_rate": 1e-3,
        "n_steps": 48,  # 5 days (240 steps) of experience before update
        "batch_size": 16,  # Reduced for LSTM
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "clip_range": 0.2,
        "ent_coef": 0.1,
        "vf_coef": 0.2,
        "max_grad_norm": 1.0,
        "verbose": 1,
        "tensorboard_log": "logs/iso/tensorboard",
        "policy_kwargs": dict(
            net_arch=[lstm_size * 2, lstm_size],
            lstm_hidden_size=lstm_size
        )
    }

    # Override with any provided kwargs
    model_kwargs.update(kwargs)

    # Set seed if provided
    if seed is not None:
        model_kwargs["seed"] = seed

    # Create and return the model
    return RecurrentPPO(**model_kwargs)


def create_ppo_model(env, net_arch=[64, 64], seed=None, **kwargs):
    """
    Create a standard PPO model with appropriate hyperparameters.

    Args:
        env: The vectorized environment
        net_arch: Network architecture
        seed: Random seed

    Returns:
        A PPO model with MlpPolicy
    """
    # Common kwargs with defaults that can be overridden
    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 1e-3,
        "n_steps": 64,  # 5 days (240 steps) of experience before update
        "batch_size": 16,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": "logs/iso/tensorboard",
        "policy_kwargs": dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            activation_fn=torch.nn.Tanh
        )
    }

    # Override with any provided kwargs
    model_kwargs.update(kwargs)

    # Set seed if provided
    if seed is not None:
        model_kwargs["seed"] = seed

    # Create and return the model
    return PPO(**model_kwargs)


def create_td3_model(env, net_arch=[64, 64, 32], seed=None, **kwargs):
    """
    Create a TD3 model with appropriate hyperparameters.

    Args:
        env: The vectorized environment
        net_arch: Network architecture
        seed: Random seed

    Returns:
        A TD3 model with MlpPolicy
    """
    # Common kwargs with defaults that can be overridden
    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": 1e-3,
        "buffer_size": 30000,
        "learning_starts": 50,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "verbose": 1,
        "tensorboard_log": "logs/iso/tensorboard",
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=torch.nn.ReLU
        )
    }

    # Override with any provided kwargs
    model_kwargs.update(kwargs)

    # Set seed if provided
    if seed is not None:
        model_kwargs["seed"] = seed

    # Create and return the model
    return TD3(**model_kwargs)


def evaluate_iso(iso_model, pcs_model, env_config, override_env=None, pcs_action_sequence=None, num_episodes=5,
                 seed=None, plot_dir=None, is_recurrent=True):
    """
    Evaluate ISO agent performance by running episodes and plotting results.

    Args:
        iso_model: Trained ISO model
        pcs_model: PCS model to use during evaluation (or None)
        env_config: Environment configuration
        override_env: Optional pre-configured environment to use
        pcs_action_sequence: Optional sequence of predefined PCS actions to use
        num_episodes: Number of episodes to run
        seed: Random seed
        is_recurrent: Whether the model is recurrent (RecurrentPPO) or not (PPO)

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating ISO agent for {num_episodes} episodes...")

    # Use provided wrapped environment if available, else create properly wrapped env
    if override_env is not None:
        eval_env = override_env
        # Ensure evaluation mode is set
        if hasattr(eval_env, 'training'):
            eval_env.training = False
        if hasattr(eval_env, 'norm_reward'):
            eval_env.norm_reward = False
    else:
        # Create wrapped environment using make_iso_env with eval_mode=True
        eval_env = make_iso_env(
            steps_per_iteration=480,  # Default to 10 days
            cost_type=env_config.get('cost_type', 'CONSTANT'),
            pricing_policy=env_config.get('pricing_policy', 'ONLINE'),
            demand_pattern=env_config.get('demand_pattern', 'CONSTANT'),
            log_dir="../logs",
            model_dir="../models",
            plot_dir="../plots",
            pcs_policy=pcs_model,
            pcs_action_sequence=pcs_action_sequence,  # Add predefined sequence
            use_dispatch_action=env_config.get('use_dispatch_action', False),
            norm_path=os.path.join("../logs", "iso", "vec_normalize.pkl"),
            # Load saved normalization from training path
            eval_mode=True,  # Enable evaluation mode
            demand_data_path=env_config.get('demand_data_path', None)  # Add demand data path
        )

    # Initialize metrics
    total_iso_reward = 0
    episode_metrics = []

    # Initialize LSTM states (only for RecurrentPPO)
    iso_lstm_states = None

    # Determine plot directory
    eval_plots_dir = plot_dir or os.path.join("../eval_plots", "iso")
    os.makedirs(eval_plots_dir, exist_ok=True)

    # Run evaluation episodes
    for episode in range(num_episodes):
        print(f"Running evaluation episode {episode + 1}/{num_episodes}")

        # Reset environment and get ISO observation (already wrapped in VecEnv)
        # Use seed via env.seed() since VecNormalize.reset() does not accept seed
        if seed is not None:
            eval_env.seed(seed + episode)
        obs = eval_env.reset()[0]

        # Episode tracking
        done = False
        episode_iso_reward = 0
        episode_steps = 0
        episode_data = []
        episode_start = True

        while not done:
            # Get ISO action (with LSTM state if recurrent)
            if is_recurrent:
                iso_action, iso_lstm_states = iso_model.predict(
                    obs,
                    state=iso_lstm_states,
                    episode_start=np.array([episode_start]),
                    deterministic=True
                )
                episode_start = False
            else:
                # Standard PPO doesn't use recurrent state
                iso_action, _ = iso_model.predict(
                    obs,
                    deterministic=True
                )

            # Batch action for single env: make shape (1, action_dim)
            if isinstance(iso_action, np.ndarray) and iso_action.ndim == 1:
                batch_action = iso_action[np.newaxis, :]
            else:
                batch_action = np.array([iso_action])
            obs, reward, done, info = eval_env.step(batch_action)

            # The done flag from VecEnv is a boolean
            done = done.any()

            # Extract reward (scalar from VecEnv)
            iso_reward = reward[0]

            # Update episode metrics
            episode_iso_reward += iso_reward
            episode_steps += 1

            # Extract info from VecEnv (first env only)
            step_info = info[0] if isinstance(info, list) else info

            # Store step data for plotting
            # Handle numpy arrays returned by VecEnv (shape [n_envs, action_dim])
            if isinstance(iso_action, np.ndarray):
                # If batch, take first env's action vector
                act_vec = iso_action[0] if iso_action.ndim > 1 else iso_action
                action_scalar = float(act_vec[0])
                iso_action_val = act_vec
            else:
                action_scalar = float(iso_action)
                iso_action_val = iso_action
            step_data = {
                'step': episode_steps,
                'action': action_scalar,
                'iso_action': iso_action_val,
                'pcs_action': step_info.get('pcs_action', np.zeros(1)),
                'predicted_demand': step_info.get('predicted_demand', 0),
                'realized_demand': step_info.get('realized_demand', 0),
                'battery_level': step_info.get('battery_level', 0),
                'iso_sell_price': step_info.get('iso_sell_price', 0),
                'iso_buy_price': step_info.get('iso_buy_price', 0),
                'net_exchange': step_info.get('net_exchange', 0),
                'dispatch': step_info.get('dispatch', 0),
                'dispatch_cost': step_info.get('dispatch_cost', 0),
                'reserve_cost': step_info.get('reserve_cost', 0),
                'shortfall': step_info.get('shortfall', 0),
                'pcs_exchange_cost': step_info.get('pcs_exchange_cost', 0),
                'iso_reward': iso_reward,
            }
            # Include background process info from step_info
            for key, value in step_info.items():
                if isinstance(key, str) and key.startswith('background_'):
                    step_data[key] = value
            episode_data.append(step_data)

        # Update total metrics
        total_iso_reward += episode_iso_reward

        # Store episode metrics
        episode_metrics.append({
            'episode': episode,
            'iso_reward': episode_iso_reward,
            'steps': episode_steps,
            'data': episode_data
        })

        # Create an ActionTrackingCallback-compatible structure for plotting
        callback = PlotCallback(verbose=0)
        callback.agent_name = "iso_eval"
        callback.save_path = eval_plots_dir
        callback.all_episodes_actions = [episode_data]  # Wrap in list as it expects list of episodes

        # Use the callback's plotting function
        callback.plot_episode_results(0, eval_plots_dir)

        print(f"Episode {episode + 1} complete: ISO reward={episode_iso_reward:.2f}")

    # Calculate average metrics
    avg_iso_reward = total_iso_reward / num_episodes

    print(f"\nEvaluation complete:")
    print(f"Average ISO reward: {avg_iso_reward:.2f}")
    print(f"Plots saved to: {eval_plots_dir}")

    # List generated plot files
    plot_files = sorted(os.listdir(eval_plots_dir))
    print(f"Generated {len(plot_files)} plot files in {eval_plots_dir}:")
    for fname in plot_files:
        print(f"  - {fname}")

    # Add statistical analysis of ISO evaluation
    # Collect per-step data across all episodes for component analysis
    dispatch_vals = []  # ISO dispatch power
    reserve_vals = []  # Shortfall power
    pcs_energy = []  # PCS energy exchange
    dispatch_costs = []
    reserve_costs = []
    pcs_costs = []
    for ep in episode_metrics:
        for d in ep['data']:
            dispatch_vals.append(d.get('dispatch', 0))
            reserve_vals.append(d.get('shortfall', 0))
            pcs_energy.append(d.get('net_exchange', 0))
            dispatch_costs.append(d.get('dispatch_cost', 0))
            reserve_costs.append(d.get('reserve_cost', 0))
            pcs_costs.append(d.get('pcs_exchange_cost', 0))

    # Compute statistics
    def compute_stats(arr):
        a = np.array(arr)
        return {
            'total': float(a.sum()),
            'mean': float(a.mean()),
            'std': float(a.std(ddof=1)),
            'min': float(a.min()),
            'max': float(a.max()),
            'median': float(np.median(a))
        }

    stats = {
        'dispatch_power': compute_stats(dispatch_vals),
        'reserve_power': compute_stats(reserve_vals),
        'pcs_energy_MWh': compute_stats(pcs_energy),
        'dispatch_cost': compute_stats(dispatch_costs),
        'reserve_cost': compute_stats(reserve_costs),
        'pcs_exchange_cost': compute_stats(pcs_costs)
    }
    # Write stats to file
    stats_file = os.path.join(eval_plots_dir, 'evaluation_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write('Statistical Analysis of ISO Evaluation:\n')
        for comp, comp_stats in stats.items():
            f.write(f'\n{comp}:\n')
            for k, v in comp_stats.items():
                f.write(f'  {k}: {v}\n')
    print(f"Statistical analysis written to {stats_file}")

    # Return evaluation results
    return {
        'avg_iso_reward': avg_iso_reward,
        'episodes': episode_metrics
    }


def evaluate_iso_agent(iso_model, env_config, num_episodes=5, seed=None, plot_dir=None):
    """
    Evaluate ISO agent using stable_baselines3 evaluate_policy on the model's wrapped env.
    """
    # Get the evaluation environment from the model
    eval_env = iso_model.get_env()
    # Run evaluation
    mean_reward, std_reward = evaluate_policy(
        iso_model,
        eval_env,
        n_eval_episodes=num_episodes,
        deterministic=True
    )
    print(f"Evaluated ISO agent over {num_episodes} episodes - avg reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return {
        'avg_iso_reward': mean_reward,
        'std_iso_reward': std_reward
    }


def main():
    # Parse arguments
    args = parse_args()

    parse_arguments(args)

    # Determine which algorithm to use
    is_recurrent = args.algorithm == "recurrent_ppo"
    is_td3 = args.algorithm == "td3"
    algorithm_name = "RecurrentPPO" if is_recurrent else ("TD3" if is_td3 else "PPO")
    print(f"Using {algorithm_name} algorithm")

    # Load predefined PCS actions if specified
    pcs_action_sequence = parse_PCS_actions(args)

    # If eval-only, load best model and normalization then evaluate
    if args.eval_only:
        is_recurrent, is_td3 = evaulate(args, is_recurrent, is_td3, pcs_action_sequence)
        return

    # End eval-only
    # Create log directories
    create_log_dirs(args)

    # Environment configuration
    best_iso_model_path, best_iso_reward, env_config, is_recurrent, is_td3, iso_env, iso_model, pcs_model, start_iteration = create_iso(
        algorithm_name, args, is_recurrent, is_td3, pcs_action_sequence)

    # Training loop
    best_iso_model_path = train(args, best_iso_model_path, best_iso_reward, is_recurrent, is_td3, iso_env, iso_model,
                                pcs_action_sequence, pcs_model, start_iteration)

    eval_results = evaluate(args, best_iso_model_path, env_config, is_recurrent, is_td3, pcs_action_sequence, pcs_model)

    print("\nFinal Evaluation complete!")
    print(f"Average ISO reward: {eval_results['avg_iso_reward']:.2f}")
    print("\nPlots saved to eval_plots/iso/ directory")


def create_iso(algorithm_name, args, is_recurrent, is_td3, pcs_action_sequence):
    env_config = {
        'cost_type': args.cost_type,
        'pricing_policy': args.pricing_policy,
        'demand_pattern': args.demand_pattern,
        'use_dispatch_action': args.use_dispatch,
        'demand_data_path': args.demand_data,  # Add demand data path
    }
    print(f"Starting ISO training with {args.iterations} iterations using {algorithm_name}")
    print(f"Environment config: {env_config}")
    # Initialize tracking for best model
    best_iso_reward = float('-inf')
    best_iso_model_path = None
    # Determine if we're continuing from existing models
    continuing_training = args.continue_from_best or args.initial_iso_model
    start_iteration = args.start_iteration if continuing_training else 1
    # If continuing from best, look for best model files
    if args.continue_from_best:
        # Check for algorithm-specific best models first
        if is_recurrent:
            prefix = "rppo"
        elif is_td3:
            prefix = "td3"
        else:
            prefix = "ppo"
        best_iso_path = os.path.join(args.model_dir, f"{prefix}_iso_best.zip")

        # Fall back to legacy model path if not found
        if not os.path.exists(best_iso_path):
            best_iso_path = os.path.join(args.model_dir, "iso_best.zip")

        if os.path.exists(best_iso_path):
            args.initial_iso_model = best_iso_path
            print(f"Found best ISO model at {best_iso_path}")
        else:
            print(f"Warning: Best ISO model not found at {best_iso_path}")
    # Load PCS model if specified
    pcs_model = None
    if args.pcs_model and os.path.exists(args.pcs_model):
        print(f"Loading PCS model from {args.pcs_model}")
        try:
            # Load the model but don't attach it to an environment yet
            if "recurrent" in args.pcs_model or os.path.basename(args.pcs_model).startswith("rppo_"):
                pcs_model = RecurrentPPO.load(args.pcs_model)
            elif "td3" in args.pcs_model or os.path.basename(args.pcs_model).startswith("td3_"):
                pcs_model = TD3.load(args.pcs_model)
            else:
                pcs_model = PPO.load(args.pcs_model)
            print(f"PCS model loaded successfully!")
        except Exception as e:
            print(f"Error loading PCS model: {e}")
            print("Continuing without PCS model")
    else:
        print("No PCS model specified, ISO will train with default PCS behavior")
    # If a PCS normalization file was provided, wrap PCS model's env
    if pcs_model is not None and args.pcs_norm_path and os.path.exists(args.pcs_norm_path):
        print(f"Loading PCS normalization from: {args.pcs_norm_path}")
        from energy_net.controllers.alternating_wrappers import make_pcs_env
        # Create a dummy PCS env with normalization loaded
        pcs_env_norm = make_pcs_env(
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            iso_policy=None,
            iso_action_sequence=None,
            norm_path=args.pcs_norm_path,
            use_dispatch_action=args.use_dispatch,
            eval_mode=True,
            demand_data_path=args.demand_data
        )
        pcs_model.set_env(pcs_env_norm)
    # Create ISO environment using existing wrapper
    iso_env = make_iso_env(
        steps_per_iteration=args.timesteps,
        cost_type=args.cost_type,
        pricing_policy=args.pricing_policy,
        demand_pattern=args.demand_pattern,
        seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        plot_dir=args.plot_dir,
        pcs_policy=pcs_model,  # Pass the loaded PCS model (or None)
        pcs_action_sequence=pcs_action_sequence,  # Pass predefined sequence (or None)
        use_dispatch_action=args.use_dispatch,
        demand_data_path=args.demand_data
    )
    print(">>> ISO action space:", iso_env.action_space)  # <<< ADD THIS
    # Save environment normalization for evaluation
    try:
        iso_env.save(f"{args.model_dir}/iso_vecnormalize.pkl")
        print(f"Saved VecNormalize stats to {args.model_dir}/iso_vecnormalize.pkl")
    except Exception:
        pass
    # Initialize or load ISO model
    is_recurrent, is_td3, iso_model = init_or_load_iso_model(algorithm_name, args, is_recurrent, is_td3, iso_env)
    # Save normalization stats
    iso_env.save(f"{args.log_dir}/iso/vec_normalize.pkl")
    return best_iso_model_path, best_iso_reward, env_config, is_recurrent, is_td3, iso_env, iso_model, pcs_model, start_iteration


def evaluate(args, best_iso_model_path, env_config, is_recurrent, is_td3, pcs_action_sequence, pcs_model):
    # Load best model for final evaluation
    print("\n" + "=" * 50)
    print("Training complete! Loading best model for evaluation...")
    # Choose which ISO model to use - prioritize best model
    if best_iso_model_path and os.path.exists(best_iso_model_path):
        final_iso_model_path = best_iso_model_path
        print(f"Using best ISO model: {final_iso_model_path}")
    else:
        # Fall back to latest numbered model
        if is_recurrent:
            prefix = "rppo"
        elif is_td3:
            prefix = "td3"
        else:
            prefix = "ppo"
        final_iso_model_path = f"{args.model_dir}/{prefix}_iso_{args.iterations}.zip"
        if not os.path.exists(final_iso_model_path):
            print(f"WARNING: Latest ISO model not found at {final_iso_model_path}")
            print(f"Searching for the most recent ISO model available...")
            # Find the most recent ISO model
            iso_models = [f for f in os.listdir(args.model_dir) if
                          f.startswith(f"{prefix}_iso_") and f.endswith(".zip")]
            if iso_models:
                iso_models.sort(
                    key=lambda x: int(x.split("_")[2].split(".")[0]) if x.split("_")[2].split(".")[0].isdigit() else 0,
                    reverse=True)
                final_iso_model_path = os.path.join(args.model_dir, iso_models[0])
                print(f"Found ISO model: {final_iso_model_path}")
            else:
                print("No ISO model found. Skipping evaluation.")
                raise FileNotFoundError("No ISO model found. Skipping evaluation.")
    # Load the model
    if is_recurrent or "rppo" in final_iso_model_path:
        final_iso_model = RecurrentPPO.load(final_iso_model_path)
        is_recurrent = True
        is_td3 = False
    elif is_td3 or "td3" in final_iso_model_path:
        final_iso_model = TD3.load(final_iso_model_path)
        is_recurrent = False
        is_td3 = True
    else:
        final_iso_model = PPO.load(final_iso_model_path)
        is_recurrent = False
        is_td3 = False
    # Evaluate ISO agent
    eval_results = evaluate_iso(
        final_iso_model,
        pcs_model,  # Use the same PCS model we trained with (or None)
        env_config,
        pcs_action_sequence=pcs_action_sequence,  # Pass predefined sequence
        num_episodes=args.eval_episodes,
        seed=args.seed + 1000,
        is_recurrent=is_recurrent
    )
    return eval_results


def train(args, best_iso_model_path, best_iso_reward, is_recurrent, is_td3, iso_env, iso_model, pcs_action_sequence,
          pcs_model, start_iteration):
    for iteration in range(start_iteration, args.iterations + 1):
        print(f"\n{'=' * 20} Iteration {iteration}/{args.iterations} {'=' * 20}")

        # Save current iteration to file for PlotCallback
        with open("../temp/current_iteration.txt", "w") as f:
            f.write(str(iteration))

        # Create callback for plotting
        iso_callback = PlotCallback(verbose=1)
        iso_callback.agent_name = "iso"
        iso_callback.save_path = f"{args.plot_dir}/iso_{iteration}"
        os.makedirs(iso_callback.save_path, exist_ok=True)

        # Train ISO model
        print(f"\nTraining ISO agent for iteration {iteration}...")
        iso_model.learn(
            total_timesteps=args.timesteps,
            callback=iso_callback,
            tb_log_name=f"iso_iter_{iteration}"
        )

        # Use algorithm-specific prefix for model files
        if is_recurrent:
            prefix = "rppo"
        elif is_td3:
            prefix = "td3"
        else:
            prefix = "ppo"

        # Save ISO model and normalization stats
        iso_model_path = f"{args.model_dir}/{prefix}_iso_{iteration}.zip"
        iso_model.save(iso_model_path)
        iso_env.save(f"{args.log_dir}/iso/vec_normalize.pkl")
        print(f"ISO model saved to {iso_model_path}")

        # Evaluate current ISO model to determine if it's the best so far
        # We'll do a quick evaluation on a few episodes
        print("\nEvaluating current ISO model...")

        # Create test environment
        eval_env = make_iso_env(
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed + 700 + iteration,  # Different seed for evaluation
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            pcs_policy=pcs_model,  # Pass the same PCS model
            pcs_action_sequence=pcs_action_sequence,  # Pass same action sequence
            use_dispatch_action=args.use_dispatch,
            norm_path=f"{args.log_dir}/iso/vec_normalize.pkl",  # Use latest normalization
            eval_mode=True,  # Important: set evaluation mode
            demand_data_path=args.demand_data
        )

        # For RecurrentPPO, we need to unwrap the VecEnv to get the wrapped environment
        if hasattr(eval_env, 'venv'):
            eval_env.training = False
            eval_env.norm_reward = False

        # Evaluate ISO performance over a few episodes
        total_iso_reward = 0
        n_eval_episodes = 3  # Smaller number for quick evaluation during training

        for eval_episode in range(n_eval_episodes):
            obs = eval_env.reset()[0]  # VecEnv returns tuple with obs at index 0
            done = False
            episode_reward = 0
            lstm_states = None if not is_recurrent else None
            episode_start = True

            while not done:
                # Get ISO action from current model (with recurrent states if using RecurrentPPO)
                if is_recurrent:
                    iso_action, lstm_states = iso_model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=np.array([episode_start]),
                        deterministic=True
                    )
                    episode_start = False
                else:
                    # Standard PPO or TD3 doesn't use recurrent state
                    iso_action, _ = iso_model.predict(
                        obs,
                        deterministic=True
                    )

                # Batch action for single env: reshape to (1, action_dim)
                if isinstance(iso_action, np.ndarray) and iso_action.ndim == 1:
                    batch_action = iso_action[np.newaxis, :]
                else:
                    batch_action = np.array([iso_action])
                obs, reward, terminated, info = eval_env.step(batch_action)

                # Extract reward (scalar from VecEnv)
                iso_reward = reward[0]

                # Update episode reward
                episode_reward += iso_reward

                # Check termination
                done = terminated.any()

            # Add episode reward to total
            total_iso_reward += episode_reward

        # Calculate average reward
        avg_iso_reward = total_iso_reward / n_eval_episodes
        print(f"ISO Evaluation - Average reward: {avg_iso_reward:.2f}")

        # Check if this is the best ISO model so far
        if avg_iso_reward > best_iso_reward:
            best_iso_reward = avg_iso_reward
            best_iso_model_path = f"{args.model_dir}/{prefix}_iso_best.zip"
            print(f"New best ISO model found! Saving to {best_iso_model_path}")
            iso_model.save(best_iso_model_path)

            # Also save the normalization stats for this best model
            eval_env.save(f"{args.model_dir}/{prefix}_iso_best_norm.pkl")
            print(f"Saved best model normalization stats to {args.model_dir}/{prefix}_iso_best_norm.pkl")
    return best_iso_model_path


def init_or_load_iso_model(algorithm_name, args, is_recurrent, is_td3, iso_env):
    if args.initial_iso_model and os.path.exists(args.initial_iso_model):
        print(f"\nLoading initial ISO model from {args.initial_iso_model}")
        if is_recurrent or "recurrent" in args.initial_iso_model or os.path.basename(args.initial_iso_model).startswith(
                "rppo_"):
            iso_model = RecurrentPPO.load(
                args.initial_iso_model,
                env=iso_env,
                custom_objects={"learning_rate": args.learning_rate}
            )
            is_recurrent = True  # Force recurrent mode if loading a recurrent model
            is_td3 = False
        elif is_td3 or "td3" in args.initial_iso_model or os.path.basename(args.initial_iso_model).startswith("td3_"):
            iso_model = TD3.load(
                args.initial_iso_model,
                env=iso_env,
                custom_objects={"learning_rate": args.learning_rate}
            )
            is_recurrent = False
            is_td3 = True
        else:
            iso_model = PPO.load(
                args.initial_iso_model,
                env=iso_env,
                custom_objects={"learning_rate": args.learning_rate}
            )
            is_recurrent = False
            is_td3 = False
    else:
        print(f"\nInitializing new {algorithm_name} model...")
        if is_recurrent:
            iso_model = create_recurrent_model(
                iso_env,
                lstm_size=args.lstm_size,
                seed=args.seed,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                ent_coef=args.ent_coef
            )
        elif is_td3:
            # For TD3, create action noise object for exploration
            action_dim = iso_env.action_space.shape
            total_timesteps = args.iterations * args.timesteps

            # Set up once during model creation
            action_noise = LinearDecayActionNoise(
                mean=np.zeros(action_dim),
                sigma=args.policy_noise,
                final_sigma=args.final_noise,
                decay_steps=total_timesteps
            )

            # Create TD3 model with action noise
            iso_model = create_td3_model(
                iso_env,
                net_arch=args.net_arch,
                seed=args.seed,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                train_freq=args.train_freq,
                target_policy_noise=args.policy_noise,
                batch_size=args.batch_size,
                action_noise=action_noise
            )
        else:
            iso_model = create_ppo_model(
                iso_env,
                net_arch=args.net_arch,
                seed=args.seed,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                ent_coef=args.ent_coef
            )

        # Short training step to initialize the model
        iso_model.learn(total_timesteps=1)
        # Use algorithm-specific prefix for model files
        if is_recurrent:
            prefix = "rppo"
        elif is_td3:
            prefix = "td3"
        else:
            prefix = "ppo"
        iso_model_path = f"{args.model_dir}/{prefix}_iso_init.zip"
        iso_model.save(iso_model_path)
    return is_recurrent, is_td3, iso_model


def create_log_dirs(args):
    os.makedirs(f"{args.log_dir}/iso/monitor", exist_ok=True)
    os.makedirs(f"{args.log_dir}/iso/tensorboard", exist_ok=True)
    os.makedirs(f"{args.model_dir}", exist_ok=True)
    os.makedirs(f"{args.plot_dir}/iso", exist_ok=True)
    os.makedirs("../eval_plots/iso", exist_ok=True)
    # Create temp directory for iteration tracking
    os.makedirs("../temp", exist_ok=True)


def evaulate(args, is_recurrent, is_td3, pcs_action_sequence):
    # Default best model and norm paths
    if args.best_model:
        best_model = args.best_model
    else:
        # Check for algorithm-specific best models first
        if is_recurrent:
            prefix = "rppo"
        elif is_td3:
            prefix = "td3"
        else:
            prefix = "ppo"
        best_model = f"{args.model_dir}/{prefix}_iso_best.zip"

        # Fall back to legacy model path if not found
        if not os.path.exists(best_model):
            best_model = f"{args.model_dir}/iso_best.zip"
    norm_path = args.norm_path or os.path.join("../logs", "iso", "vec_normalize.pkl")
    print(f"Evaluation-only mode. Loading model: {best_model}\nNormalization file: {norm_path}")
    # Load fixed PCS policy if provided
    pcs_policy = None
    if args.pcs_model and os.path.exists(args.pcs_model):
        print(f"Loading PCS model from {args.pcs_model}")
        try:
            if "recurrent" in args.pcs_model or os.path.basename(args.pcs_model).startswith("rppo_"):
                pcs_policy = RecurrentPPO.load(args.pcs_model)
            elif "td3" in args.pcs_model or os.path.basename(args.pcs_model).startswith("td3_"):
                pcs_policy = TD3.load(args.pcs_model)
            else:
                pcs_policy = PPO.load(args.pcs_model)
            print("PCS model loaded successfully!")
        except Exception as e:
            print(f"Error loading PCS model: {e}")
            print("Continuing with default PCS actions")
    # Load PCS normalization if provided
    if pcs_policy is not None and args.pcs_norm_path and os.path.exists(args.pcs_norm_path):
        print(f"Loading PCS normalization from: {args.pcs_norm_path}")
        from energy_net.controllers.alternating_wrappers import make_pcs_env
        pcs_env_norm = make_pcs_env(
            steps_per_iteration=args.timesteps,
            cost_type=args.cost_type,
            pricing_policy=args.pricing_policy,
            demand_pattern=args.demand_pattern,
            seed=args.seed,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            iso_policy=None,
            iso_action_sequence=None,
            norm_path=args.pcs_norm_path,
            use_dispatch_action=args.use_dispatch,
            eval_mode=True,
            demand_data_path=args.demand_data
        )
        pcs_policy.set_env(pcs_env_norm)
    # Create evaluation environment with saved normalization
    eval_env = make_iso_env(
        steps_per_iteration=args.timesteps,
        cost_type=args.cost_type,
        pricing_policy=args.pricing_policy,
        demand_pattern=args.demand_pattern,
        seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        plot_dir=args.plot_dir,
        use_dispatch_action=args.use_dispatch,
        pcs_policy=pcs_policy,
        pcs_action_sequence=pcs_action_sequence,  # Add predefined sequence
        norm_path=norm_path,
        eval_mode=True,
        demand_data_path=args.demand_data
    )
    # Load model - determine type from file
    if "recurrent" in best_model or os.path.basename(best_model).startswith("rppo_"):
        print("Loading RecurrentPPO model")
        iso_model = RecurrentPPO.load(best_model, env=eval_env,
                                      custom_objects={"learning_rate": args.learning_rate})
        is_recurrent = True
        is_td3 = False
    elif "td3" in best_model or os.path.basename(best_model).startswith("td3_"):
        print("Loading TD3 model")
        iso_model = TD3.load(best_model, env=eval_env, custom_objects={"learning_rate": args.learning_rate})
        is_recurrent = False
        is_td3 = True
    else:
        print("Loading standard PPO model")
        iso_model = PPO.load(best_model, env=eval_env, custom_objects={"learning_rate": args.learning_rate})
        is_recurrent = False
        is_td3 = False
    # Evaluate best model
    results = evaluate_iso(
        iso_model=iso_model,
        pcs_model=pcs_policy,
        env_config={
            'cost_type': args.cost_type,
            'pricing_policy': args.pricing_policy,
            'demand_pattern': args.demand_pattern,
            'use_dispatch_action': args.use_dispatch,
            'demand_data_path': args.demand_data,  # Add demand data path
        },
        override_env=eval_env,
        pcs_action_sequence=pcs_action_sequence,  # Add predefined sequence
        num_episodes=args.eval_episodes,
        seed=args.seed,
        is_recurrent=is_recurrent
    )
    print(f"Evaluation complete. Avg ISO reward: {results['avg_iso_reward']:.2f}")
    print("Plots saved to: eval_plots/iso/")
    return is_recurrent, is_td3


def parse_PCS_actions(args):
    pcs_action_sequence = None
    if args.pcs_action_file and os.path.exists(args.pcs_action_file):
        try:
            pcs_action_sequence = np.load(args.pcs_action_file)
            print(f"Loaded predefined PCS action sequence from {args.pcs_action_file}")
            print(f"Sequence shape: {pcs_action_sequence.shape}")
            print(f"First few actions: {pcs_action_sequence[:5]}")
        except Exception as e:
            print(f"Error loading PCS action sequence: {e}")
            print("Continuing without predefined PCS actions")
    return pcs_action_sequence


def parse_arguments(args):
    # Check if DATA_DRIVEN pattern requires a data file
    if args.demand_pattern == "DATA_DRIVEN" and not args.demand_data:
        print("ERROR: DATA_DRIVEN demand pattern requires a demand data file.")
        print("Please specify the file path using --demand-data")
        raise ValueError("DATA_DRIVEN demand pattern requires a demand data file. "
                         "Please specify the file path using --demand-data.")
    # Convert string pattern to enum
    if args.demand_pattern == "DATA_DRIVEN":
        from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
        print(f"Using DATA_DRIVEN demand pattern with data from: {args.demand_data}")
        demand_pattern = DemandPattern.DATA_DRIVEN
    else:
        from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
        demand_pattern = DemandPattern[args.demand_pattern]


if __name__ == "__main__":
    main()