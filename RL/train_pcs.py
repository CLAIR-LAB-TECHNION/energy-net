#!/usr/bin/env python3
"""
Direct Training Script for PCS Agent with Predefined ISO Action Sequences

This script trains a PCS agent against a fixed ISO schedule. It supports
PPO, RecurrentPPO, and TD3 algorithms, and can load an ISO action sequence
(.npy file) instead of using an ISO policy.
"""
import os
import argparse
import numpy as np
import logging
from stable_baselines3 import PPO, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

# Import helper functions from ISO trainer
from RL.train_iso import create_ppo_model, create_recurrent_model, create_td3_model, LinearDecayActionNoise

# Import env factory and plotting callback
from energy_net.controllers.alternating_wrappers import make_pcs_env
from energy_net.controllers.plot_callback import PlotCallback

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pcs_trainer")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCS agent with fixed ISO schedule")
    # Training parameters
    parser.add_argument("--iterations", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--timesteps", type=int, default=480, help="Steps per iteration (e.g., 480 = 10 days)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # Environment parameters
    parser.add_argument("--demand-pattern", type=str, default="CONSTANT", 
                        choices=["CONSTANT", "SINUSOIDAL", "DOUBLE_PEAK", "TWO_PEAK", "DATA_DRIVEN"],
                        help="Demand pattern for environment creation")
    parser.add_argument("--demand-data", type=str, default=None,
                        help="Path to demand data YAML file (required for DATA_DRIVEN pattern)")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE", 
                        choices=["ONLINE", "CONSTANT", "QUADRATIC"],
                        help="Pricing policy type")
    parser.add_argument("--cost-type", type=str, default="CONSTANT", 
                        choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                        help="Cost type for environment creation")
    parser.add_argument("--use-dispatch", action="store_true", 
                        help="Include dispatch action in ISO space (affects env)")
    # ISO sequence control
    parser.add_argument("--iso-action-file", type=str, default=None,
                        help="Path to numpy file (.npy) containing ISO action sequence for PCS training")
    # Algorithm parameters
    parser.add_argument("--algorithm", type=str, choices=["ppo", "recurrent_ppo", "td3"],
                        default="ppo", help="RL algorithm for PCS agent")
    parser.add_argument("--lstm-size", type=int, default=64, 
                        help="LSTM hidden size (for RecurrentPPO)")
    parser.add_argument("--net-arch", nargs="+", type=int, default=[64, 64],
                        help="Network architecture for policy/value nets")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Batch size (PPO/TD3)")
    parser.add_argument("--ent-coef", type=float, default=0.05, 
                        help="Entropy coefficient (PPO only)")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Replay buffer size (TD3 only)")
    parser.add_argument("--train-freq", type=int, default=1,
                        help="Train frequency (TD3 only)")
    parser.add_argument("--policy-noise", type=float, default=0.2,
                        help="Initial policy noise sigma (TD3)")
    parser.add_argument("--final-noise", type=float, default=0.05,
                        help="Final policy noise sigma after decay (TD3)")
    parser.add_argument("--noise-type", type=str, choices=["normal", "ou"],
                        default="normal", help="Type of action noise for TD3")
    # IO paths
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory for training logs and Monitor CSVs")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--plot-dir", type=str, default="plots",
                        help="Directory to save training plots")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes for final evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate a saved PCS model and exit")
    parser.add_argument("--best-model", type=str, default=None,
                        help="Path to a saved PCS model for evaluation only")
    parser.add_argument("--norm-path", type=str, default=None,
                        help="Path to VecNormalize stats file for evaluation only")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Convert string enums to actual enums
    from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
    from energy_net.market.pricing.pricing_policy import PricingPolicy
    from energy_net.market.pricing.cost_types import CostType

    if args.demand_pattern == "DATA_DRIVEN" and not args.demand_data:
        print("ERROR: DATA_DRIVEN demand pattern requires --demand-data")
        return
    demand_pattern = DemandPattern[args.demand_pattern]
    pricing_policy = PricingPolicy[args.pricing_policy]
    cost_type = CostType[args.cost_type]

    # Load ISO action sequence if provided
    iso_sequence = None
    if args.iso_action_file and os.path.exists(args.iso_action_file):
        iso_sequence = np.load(args.iso_action_file)
        print(f"Loaded ISO action sequence: {iso_sequence.shape} steps")
    else:
        print("No ISO action sequence provided, PCS will see default ISO policy behavior")

    # Evaluation-only mode
    if args.eval_only:
        # Create evaluation environment
        eval_env = make_pcs_env(
            steps_per_iteration=args.timesteps,
            cost_type=cost_type,
            pricing_policy=pricing_policy,
            demand_pattern=demand_pattern,
            seed=args.seed,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            plot_dir=args.plot_dir,
            iso_policy=None,
            iso_action_sequence=iso_sequence,
            use_dispatch_action=args.use_dispatch,
            norm_path=args.norm_path,
            eval_mode=True,
            demand_data_path=args.demand_data
        )
        # Load model
        model_path = args.best_model or os.path.join(args.model_dir, f"pcs_best.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PCS model not found at {model_path}")
        print(f"Loading PCS model from {model_path}")
        if args.algorithm == "recurrent_ppo":
            model = RecurrentPPO.load(model_path, env=eval_env)
        elif args.algorithm == "td3":
            model = TD3.load(model_path, env=eval_env)
        else:
            model = PPO.load(model_path, env=eval_env)
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
        )
        print(f"Evaluation complete - mean: {mean_reward:.2f} +/- {std_reward:.2f}")
        return

    # Create directories
    os.makedirs(os.path.join(args.log_dir, "pcs_monitor"), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Build training environment
    pcs_env = make_pcs_env(
        steps_per_iteration=args.timesteps,
        cost_type=cost_type,
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        seed=args.seed,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        plot_dir=args.plot_dir,
        iso_policy=None,
        iso_action_sequence=iso_sequence,
        use_dispatch_action=args.use_dispatch,
        demand_data_path=args.demand_data
    )
    # Save normalization stats
    try:
        pcs_env.save(f"{args.model_dir}/pcs_vecnormalize.pkl")
    except Exception:
        pass
    
    # Initialize or load model
    prefix = args.algorithm
    if args.best_model and os.path.exists(args.best_model):
        print(f"Loading initial PCS model from {args.best_model}")
        model = (
            RecurrentPPO.load(args.best_model, env=pcs_env) if args.algorithm == "recurrent_ppo"
            else (TD3.load(args.best_model, env=pcs_env) if args.algorithm == "td3"
                  else PPO.load(args.best_model, env=pcs_env))
        )
    else:
        print(f"Initializing new PCS model ({args.algorithm})")
        if args.algorithm == "recurrent_ppo":
            model = create_recurrent_model(
                pcs_env, lstm_size=args.lstm_size, seed=args.seed,
                learning_rate=args.learning_rate, batch_size=args.batch_size
            )
        elif args.algorithm == "td3":
            # Setup decaying noise for TD3
            action_dim = pcs_env.action_space.shape[0]
            total_steps = args.iterations * args.timesteps
            noise = LinearDecayActionNoise(
                mean=np.zeros(action_dim), sigma=args.policy_noise,
                final_sigma=args.final_noise, decay_steps=total_steps
            )
            model = create_td3_model(
                pcs_env, net_arch=args.net_arch, seed=args.seed,
                learning_rate=args.learning_rate, buffer_size=args.buffer_size,
                train_freq=args.train_freq, action_noise=noise
            )
        else:
            model = create_ppo_model(
                pcs_env, net_arch=args.net_arch, seed=args.seed,
                learning_rate=args.learning_rate, batch_size=args.batch_size,
            ent_coef=args.ent_coef
        )
        # Warm-up step
        model.learn(total_timesteps=1)
        init_path = f"{args.model_dir}/{prefix}_pcs_init.zip"
        model.save(init_path)
        print(f"Saved initial PCS model to {init_path}")

    best_reward = float('-inf')
    # Training loop
    for iteration in range(1, args.iterations + 1):
        print(f"\n=== PCS Training Iteration {iteration}/{args.iterations} ===")
        # Plot callback setup
        pcs_callback = PlotCallback(verbose=1)
        pcs_callback.agent_name = "pcs"
        pcs_callback.save_path = os.path.join(args.plot_dir, f"pcs_{iteration}")
        os.makedirs(pcs_callback.save_path, exist_ok=True)
        
        # Train
        model.learn(total_timesteps=args.timesteps, callback=pcs_callback, tb_log_name=f"pcs_iter_{iteration}")
        # Save checkpoint
        chkpt = f"{args.model_dir}/{prefix}_pcs_{iteration}.zip"
        model.save(chkpt)
        print(f"Saved PCS model checkpoint to {chkpt}")
        
        # Quick evaluation
        mean_r, _ = evaluate_policy(model, pcs_env, n_eval_episodes=3, deterministic=True)
        print(f"Iteration {iteration} eval mean reward: {mean_r:.2f}")
        if mean_r > best_reward:
            best_reward = mean_r
            best_path = f"{args.model_dir}/{prefix}_pcs_best.zip"
            model.save(best_path)
            pcs_env.save(f"{args.model_dir}/{prefix}_pcs_best_norm.pkl")
            print(f"New best model! Saved to {best_path}")

    print("\nTraining complete. Best average reward: {best_reward:.2f}")
    print(f"Best PCS model at: {args.model_dir}/{prefix}_pcs_best.zip")

    # Final evaluation
    mean_f, std_f = evaluate_policy(model, pcs_env, n_eval_episodes=args.eval_episodes, deterministic=True)
    print(f"Final evaluation - avg reward: {mean_f:.2f} +/- {std_f:.2f}")


if __name__ == "__main__":
    main()