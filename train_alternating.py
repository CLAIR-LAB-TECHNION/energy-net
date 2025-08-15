#!/usr/bin/env python3
"""
Alternating Training Script for ISO and PCS Agents

This script trains both ISO and PCS agents in an alternating fashion, allowing
for co-evolution where each agent adapts to the other's changing strategy.
Supports PPO, RecurrentPPO, and TD3 algorithms for both agents.
"""
import os
import argparse
import numpy as np
import logging
import torch
from stable_baselines3 import PPO, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, ActionNoise

# Import helper functions from individual trainers
from train_iso import create_ppo_model, create_recurrent_model, create_td3_model, LinearDecayActionNoise, evaluate_iso
from train_pcs import main as train_pcs_main

# Import env factory and plotting callback
import energy_net.envs.register_envs
from energy_net.controllers.alternating_wrappers import make_iso_env, make_pcs_env
from energy_net.controllers.plot_callback import PlotCallback

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("alternating_trainer")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ISO and PCS agents in alternating fashion")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=20, 
                        help="Number of alternating training iterations")
    parser.add_argument("--timesteps", type=int, default=480, 
                        help="Steps per iteration (480 = 10 days)")
    parser.add_argument("--seed", type=int, default=42222, 
                        help="Random seed")
    
    # Environment parameters
    parser.add_argument("--demand-pattern", type=str, default="CONSTANT", 
                        choices=["CONSTANT", "SINUSOIDAL", "DOUBLE_PEAK", "TWO_PEAK", "DATA_DRIVEN"],
                        help="Demand pattern type")
    parser.add_argument("--demand-data", type=str, default=None,
                        help="Path to demand data YAML file (required for DATA_DRIVEN pattern)")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE", 
                        choices=["ONLINE", "CONSTANT", "QUADRATIC", "INTERVALS", "QUADRATIC_INTERVALS", "SMP"],
                        help="Pricing policy type")
    parser.add_argument("--cost-type", type=str, default="CONSTANT", 
                        choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                        help="Cost type")
    parser.add_argument("--use-dispatch", action="store_true", 
                        help="Use dispatch action for ISO")
    
    # Algorithm parameters (shared for both agents)
    parser.add_argument("--iso-algorithm", type=str, choices=["ppo", "recurrent_ppo", "td3"], 
                        default="td3", help="Algorithm for ISO agent")
    parser.add_argument("--pcs-algorithm", type=str, choices=["ppo", "recurrent_ppo", "td3"], 
                        default="td3", help="Algorithm for PCS agent")
    parser.add_argument("--lstm-size", type=int, default=64, 
                        help="LSTM hidden size (for RecurrentPPO)")
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Batch size")
    parser.add_argument("--ent-coef", type=float, default=0.2, 
                        help="Entropy coefficient (PPO only)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, 
                        help="Learning rate")
    parser.add_argument("--net-arch", nargs="+", type=int, default=[64, 64],
                        help="Network architecture")
    
    # TD3 specific parameters
    parser.add_argument("--iso-buffer-size", type=int, default=1000,
                        help="ISO replay buffer size (TD3 only)")
    parser.add_argument("--pcs-buffer-size", type=int, default=1000,
                        help="PCS replay buffer size (TD3 only)")
    parser.add_argument("--train-freq", type=int, default=2,
                        help="Train frequency (TD3 only)")
    parser.add_argument("--iso-policy-noise", type=float, default=0.3,
                        help="Initial ISO policy noise sigma (TD3)")
    parser.add_argument("--pcs-policy-noise", type=float, default=0.3,
                        help="Initial PCS policy noise sigma (TD3)")
    parser.add_argument("--iso-final-noise", type=float, default=0.05,
                        help="Final ISO policy noise sigma after decay (TD3)")
    parser.add_argument("--pcs-final-noise", type=float, default=0.05,
                        help="Final PCS policy noise sigma after decay (TD3)")
    parser.add_argument("--noise-type", type=str, choices=["normal", "ou"],
                        default="normal", help="Type of action noise for TD3")
    
    # Initial models (optional)
    parser.add_argument("--initial-iso-model", type=str, default=None,
                        help="Path to initial ISO model")
    parser.add_argument("--initial-pcs-model", type=str, default=None,
                        help="Path to initial PCS model")
    parser.add_argument("--iso-norm-path", type=str, default=None,
                        help="Path to ISO VecNormalize stats")
    parser.add_argument("--pcs-norm-path", type=str, default=None,
                        help="Path to PCS VecNormalize stats")
    
    # Paths
    parser.add_argument("--log-dir", type=str, default="logs", 
                        help="Directory for logs")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory for saved models")
    parser.add_argument("--plot-dir", type=str, default="plots",
                        help="Directory for plots")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes for evaluation")
    
    # Training control
    parser.add_argument("--start-with", type=str, choices=["iso", "pcs"], default="iso",
                        help="Which agent to train first")
    parser.add_argument("--block-size", type=int, default=1,
                        help="Number of consecutive iterations to train each agent before switching")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate both agents and exit")
    parser.add_argument("--eval-iso-model", type=str, default=None,
                        help="Path to ISO model for evaluation") 
    parser.add_argument("--eval-pcs-model", type=str, default=None,
                        help="Path to PCS model for evaluation")
    
    return parser.parse_args()


def create_iso_model(env, algorithm, args):
    """Create ISO model based on algorithm choice"""
    if algorithm == "recurrent_ppo":
        return create_recurrent_model(
            env, 
            lstm_size=args.lstm_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef
        )
    elif algorithm == "td3":
        action_dim = env.action_space.shape
        total_timesteps = args.iterations * args.timesteps
        action_noise = LinearDecayActionNoise(
            mean=np.zeros(action_dim),
            sigma=args.iso_policy_noise,
            final_sigma=args.iso_final_noise,
            decay_steps=total_timesteps
        )
        return create_td3_model(
            env,
            net_arch=args.net_arch,
            seed=args.seed,
            learning_rate=args.learning_rate,
            buffer_size=args.iso_buffer_size,
            train_freq=args.train_freq,
            target_policy_noise=args.iso_policy_noise,
            batch_size=args.batch_size,
            action_noise=action_noise
        )
    else:  # ppo
        return create_ppo_model(
            env,
            net_arch=args.net_arch,
            seed=args.seed,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef
        )


def create_pcs_model(env, algorithm, args):
    """Create PCS model based on algorithm choice"""
    if algorithm == "recurrent_ppo":
        return create_recurrent_model(
            env, 
            lstm_size=args.lstm_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef
        )
    elif algorithm == "td3":
        action_dim = env.action_space.shape
        total_timesteps = args.iterations * args.timesteps
        action_noise = LinearDecayActionNoise(
            mean=np.zeros(action_dim),
            sigma=args.pcs_policy_noise,
            final_sigma=args.pcs_final_noise,
            decay_steps=total_timesteps
        )
        return create_td3_model(
            env,
            net_arch=args.net_arch,
            seed=args.seed,
            learning_rate=args.learning_rate,
            buffer_size=args.pcs_buffer_size,
            train_freq=args.train_freq,
            target_policy_noise=args.pcs_policy_noise,
            batch_size=args.batch_size,
            action_noise=action_noise
        )
    else:  # ppo
        return create_ppo_model(
            env,
            net_arch=args.net_arch,
            seed=args.seed,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef
        )


def load_model(model_path, algorithm, env=None):
    """Load a model from file"""
    if not model_path or not os.path.exists(model_path):
        return None
    
    try:
        if algorithm == "recurrent_ppo" or "recurrent" in model_path or "rppo" in model_path:
            return RecurrentPPO.load(model_path, env=env)
        elif algorithm == "td3" or "td3" in model_path:
            return TD3.load(model_path, env=env)
        else:
            return PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


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
    
    print(f"Starting block alternating training with {args.iterations} iterations")
    print(f"ISO Algorithm: {args.iso_algorithm}, PCS Algorithm: {args.pcs_algorithm}")
    print(f"Starting with: {args.start_with}")
    print(f"Block size: {args.block_size} iterations per agent")
    
    # Show the training pattern
    total_blocks = (args.iterations + args.block_size - 1) // args.block_size
    print(f"Training pattern ({total_blocks} blocks total):")
    for block in range(total_blocks):
        start_iter = block * args.block_size + 1
        end_iter = min((block + 1) * args.block_size, args.iterations)
        if args.start_with == "iso":
            agent = "ISO" if block % 2 == 0 else "PCS"
        else:
            agent = "PCS" if block % 2 == 0 else "ISO"
        print(f"  Block {block + 1}: Train {agent} (iterations {start_iter}-{end_iter})")
    print()
    
    # Create directories
    os.makedirs(f"{args.log_dir}/alternating/iso/monitor", exist_ok=True)
    os.makedirs(f"{args.log_dir}/alternating/pcs/monitor", exist_ok=True)
    os.makedirs(f"{args.log_dir}/alternating/iso/tensorboard", exist_ok=True)
    os.makedirs(f"{args.log_dir}/alternating/pcs/tensorboard", exist_ok=True)
    os.makedirs(f"{args.model_dir}/alternating", exist_ok=True)
    os.makedirs(f"{args.plot_dir}/alternating/iso", exist_ok=True)
    os.makedirs(f"{args.plot_dir}/alternating/pcs", exist_ok=True)
    os.makedirs("eval_plots/alternating/iso", exist_ok=True)
    os.makedirs("eval_plots/alternating/pcs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Initialize models
    iso_model = None
    pcs_model = None
    best_iso_reward = float('-inf')
    best_pcs_reward = float('-inf')
    
    # Load initial models if provided
    if args.initial_iso_model:
        print(f"Loading initial ISO model from {args.initial_iso_model}")
        iso_model = load_model(args.initial_iso_model, args.iso_algorithm)
        
    if args.initial_pcs_model:
        print(f"Loading initial PCS model from {args.initial_pcs_model}")
        pcs_model = load_model(args.initial_pcs_model, args.pcs_algorithm)
    
    # Evaluation-only mode
    if args.eval_only:
        print("Evaluation-only mode")
        
        # Load models for evaluation
        iso_eval_model = load_model(args.eval_iso_model or args.initial_iso_model, args.iso_algorithm)
        pcs_eval_model = load_model(args.eval_pcs_model or args.initial_pcs_model, args.pcs_algorithm)
        
        if not iso_eval_model or not pcs_eval_model:
            print("ERROR: Both ISO and PCS models required for evaluation")
            return
            
        # Create evaluation environment
        eval_env = make_iso_env(
            steps_per_iteration=args.timesteps,
            cost_type=cost_type,
            pricing_policy=pricing_policy,
            demand_pattern=demand_pattern,
            seed=args.seed,
            log_dir=f"{args.log_dir}/alternating",
            model_dir=f"{args.model_dir}/alternating",
            plot_dir=f"{args.plot_dir}/alternating",
            pcs_policy=pcs_eval_model,
            use_dispatch_action=args.use_dispatch,
            norm_path=args.iso_norm_path,
            eval_mode=True,
            demand_data_path=args.demand_data
        )
        
        # Set PCS normalization if provided
        if args.pcs_norm_path and os.path.exists(args.pcs_norm_path):
            pcs_env_norm = make_pcs_env(
                steps_per_iteration=args.timesteps,
                cost_type=cost_type,
                pricing_policy=pricing_policy,
                demand_pattern=demand_pattern,
                seed=args.seed,
                log_dir=f"{args.log_dir}/alternating",
                model_dir=f"{args.model_dir}/alternating",
                plot_dir=f"{args.plot_dir}/alternating",
                iso_policy=None,
                norm_path=args.pcs_norm_path,
                use_dispatch_action=args.use_dispatch,
                eval_mode=True,
                demand_data_path=args.demand_data
            )
            pcs_eval_model.set_env(pcs_env_norm)
        
        iso_eval_model.set_env(eval_env)
        
        # Evaluate both agents
        results = evaluate_iso(
            iso_model=iso_eval_model,
            pcs_model=pcs_eval_model,
            env_config={
                'cost_type': cost_type,
                'pricing_policy': pricing_policy,
                'demand_pattern': demand_pattern,
                'use_dispatch_action': args.use_dispatch,
                'demand_data_path': args.demand_data,
            },
            override_env=eval_env,
            num_episodes=args.eval_episodes,
            seed=args.seed,
            plot_dir="eval_plots/alternating/iso",
            is_recurrent=(args.iso_algorithm == "recurrent_ppo")
        )
        
        print(f"Evaluation complete. Avg ISO reward: {results['avg_iso_reward']:.2f}")
        return
    
    # Training loop
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*20} Alternating Iteration {iteration}/{args.iterations} {'='*20}")
        
        # Save current iteration to file for PlotCallback
        with open("temp/current_iteration.txt", "w") as f:
            f.write(str(iteration))
        
        # Determine which agent to train this iteration using block pattern
        # Calculate which block we're in (0-based)
        block_num = (iteration - 1) // args.block_size
        
        if args.start_with == "iso":
            # ISO first: even blocks train ISO, odd blocks train PCS
            train_iso_this_iteration = (block_num % 2 == 0)
        else:
            # PCS first: even blocks train PCS, odd blocks train ISO  
            train_iso_this_iteration = (block_num % 2 == 1)
        
        if train_iso_this_iteration:
            print(f"Training ISO agent (iteration {iteration}, block {block_num + 1}, step {((iteration-1) % args.block_size) + 1}/{args.block_size})")
            
            # Set up PCS model with proper normalization if it exists
            if pcs_model is not None:
                pcs_norm_path = f"{args.model_dir}/alternating/pcs_vecnormalize_{iteration-1}.pkl"
                if iteration == 1:
                    # First iteration, check for initial normalization
                    pcs_norm_path = args.pcs_norm_path
                
                if pcs_norm_path and os.path.exists(pcs_norm_path):
                    print(f"Loading PCS normalization for opponent: {pcs_norm_path}")
                    # Create a temporary PCS environment to set proper normalization
                    pcs_temp_env = make_pcs_env(
                        steps_per_iteration=args.timesteps,
                        cost_type=cost_type,
                        pricing_policy=pricing_policy,
                        demand_pattern=demand_pattern,
                        seed=args.seed,
                        log_dir=f"{args.log_dir}/alternating",
                        model_dir=f"{args.model_dir}/alternating",
                        plot_dir=f"{args.plot_dir}/alternating",
                        iso_policy=None,
                        norm_path=pcs_norm_path,
                        use_dispatch_action=args.use_dispatch,
                        eval_mode=True,  # Use eval mode for opponent
                        demand_data_path=args.demand_data
                    )
                    pcs_model.set_env(pcs_temp_env)
                    print("PCS opponent model normalization updated")

            # Create ISO environment with current PCS model
            iso_env = make_iso_env(
                steps_per_iteration=args.timesteps,
                cost_type=cost_type,
                pricing_policy=pricing_policy,
                demand_pattern=demand_pattern,
                seed=args.seed + iteration,
                log_dir=f"{args.log_dir}/alternating",
                model_dir=f"{args.model_dir}/alternating",
                plot_dir=f"{args.plot_dir}/alternating",
                pcs_policy=pcs_model,
                use_dispatch_action=args.use_dispatch,
                demand_data_path=args.demand_data
            )
            
            # Initialize or load ISO model
            if iso_model is None:
                print("Initializing new ISO model from scratch")
                iso_model = create_iso_model(iso_env, args.iso_algorithm, args)
                iso_model.learn(total_timesteps=1)  # Warm-up
                # Save initial model
                init_path = f"{args.model_dir}/alternating/{args.iso_algorithm}_iso_init.zip"
                iso_model.save(init_path)
                print(f"Saved initial ISO model to {init_path}")
            else:
                iso_model.set_env(iso_env)
            
            # Create callback for plotting
            iso_callback = PlotCallback(verbose=1)
            iso_callback.agent_name = "iso"
            iso_callback.save_path = f"{args.plot_dir}/alternating/iso/iter_{iteration}"
            os.makedirs(iso_callback.save_path, exist_ok=True)
            
            # Train ISO
            iso_model.learn(
                total_timesteps=args.timesteps,
                callback=iso_callback,
                tb_log_name=f"alternating_iso_iter_{iteration}"
            )
            
            # Save ISO model
            iso_model_path = f"{args.model_dir}/alternating/{args.iso_algorithm}_iso_{iteration}.zip"
            iso_model.save(iso_model_path)
            iso_env.save(f"{args.model_dir}/alternating/iso_vecnormalize_{iteration}.pkl")
            print(f"ISO model saved to {iso_model_path}")
            
            # Quick evaluation of ISO
            mean_r, _ = evaluate_policy(iso_model, iso_env, n_eval_episodes=3, deterministic=True)
            print(f"ISO Iteration {iteration} eval mean reward: {mean_r:.2f}")
            if mean_r > best_iso_reward:
                best_iso_reward = mean_r
                best_path = f"{args.model_dir}/alternating/{args.iso_algorithm}_iso_best.zip"
                iso_model.save(best_path)
                iso_env.save(f"{args.model_dir}/alternating/iso_vecnormalize_best.pkl")
                print(f"New best ISO model! Saved to {best_path}")
                
        else:
            print(f"Training PCS agent (iteration {iteration}, block {block_num + 1}, step {((iteration-1) % args.block_size) + 1}/{args.block_size})")
            
            # Set up ISO model with proper normalization if it exists
            if iso_model is not None:
                iso_norm_path = f"{args.model_dir}/alternating/iso_vecnormalize_{iteration-1}.pkl"
                if iteration == 1:
                    # First iteration, check for initial normalization
                    iso_norm_path = args.iso_norm_path
                
                if iso_norm_path and os.path.exists(iso_norm_path):
                    print(f"Loading ISO normalization for opponent: {iso_norm_path}")
                    # Create a temporary ISO environment to set proper normalization
                    iso_temp_env = make_iso_env(
                        steps_per_iteration=args.timesteps,
                        cost_type=cost_type,
                        pricing_policy=pricing_policy,
                        demand_pattern=demand_pattern,
                        seed=args.seed,
                        log_dir=f"{args.log_dir}/alternating",
                        model_dir=f"{args.model_dir}/alternating",
                        plot_dir=f"{args.plot_dir}/alternating",
                        pcs_policy=None,
                        use_dispatch_action=args.use_dispatch,
                        norm_path=iso_norm_path,
                        eval_mode=True,  # Use eval mode for opponent
                        demand_data_path=args.demand_data
                    )
                    iso_model.set_env(iso_temp_env)
                    print("ISO opponent model normalization updated")

            # Create PCS environment with current ISO model
            pcs_env = make_pcs_env(
                steps_per_iteration=args.timesteps,
                cost_type=cost_type,
                pricing_policy=pricing_policy,
                demand_pattern=demand_pattern,
                seed=args.seed + iteration,
                log_dir=f"{args.log_dir}/alternating",
                model_dir=f"{args.model_dir}/alternating",
                plot_dir=f"{args.plot_dir}/alternating",
                iso_policy=iso_model,
                use_dispatch_action=args.use_dispatch,
                iso_deterministic=True,  # Make ISO deterministic when acting as opponent
                demand_data_path=args.demand_data
            )
            
            # Initialize or load PCS model
            if pcs_model is None:
                print("Initializing new PCS model from scratch")
                pcs_model = create_pcs_model(pcs_env, args.pcs_algorithm, args)
                pcs_model.learn(total_timesteps=1)  # Warm-up
                # Save initial model
                init_path = f"{args.model_dir}/alternating/{args.pcs_algorithm}_pcs_init.zip"
                pcs_model.save(init_path)
                print(f"Saved initial PCS model to {init_path}")
            else:
                pcs_model.set_env(pcs_env)
            
            # Create callback for plotting
            pcs_callback = PlotCallback(verbose=1)
            pcs_callback.agent_name = "pcs"
            pcs_callback.save_path = f"{args.plot_dir}/alternating/pcs/iter_{iteration}"
            os.makedirs(pcs_callback.save_path, exist_ok=True)
            
            # Train PCS
            pcs_model.learn(
                total_timesteps=args.timesteps,
                callback=pcs_callback,
                tb_log_name=f"alternating_pcs_iter_{iteration}"
            )
            
            # Save PCS model
            pcs_model_path = f"{args.model_dir}/alternating/{args.pcs_algorithm}_pcs_{iteration}.zip"
            pcs_model.save(pcs_model_path)
            pcs_env.save(f"{args.model_dir}/alternating/pcs_vecnormalize_{iteration}.pkl")
            print(f"PCS model saved to {pcs_model_path}")
            
            # Quick evaluation of PCS
            mean_r, _ = evaluate_policy(pcs_model, pcs_env, n_eval_episodes=3, deterministic=True)
            print(f"PCS Iteration {iteration} eval mean reward: {mean_r:.2f}")
            if mean_r > best_pcs_reward:
                best_pcs_reward = mean_r
                best_path = f"{args.model_dir}/alternating/{args.pcs_algorithm}_pcs_best.zip"
                pcs_model.save(best_path)
                pcs_env.save(f"{args.model_dir}/alternating/pcs_vecnormalize_best.pkl")
                print(f"New best PCS model! Saved to {best_path}")
    
    print("\n" + "="*50)
    print("Alternating training complete!")
    print(f"Best ISO reward: {best_iso_reward:.2f}")
    print(f"Best PCS reward: {best_pcs_reward:.2f}")
    
    # Final evaluation with best models
    print("\nRunning final evaluation with best models...")
    
    # Load best models
    iso_best_path = f"{args.model_dir}/alternating/{args.iso_algorithm}_iso_best.zip"
    pcs_best_path = f"{args.model_dir}/alternating/{args.pcs_algorithm}_pcs_best.zip"
    
    if os.path.exists(iso_best_path) and os.path.exists(pcs_best_path):
        final_iso_model = load_model(iso_best_path, args.iso_algorithm)
        final_pcs_model = load_model(pcs_best_path, args.pcs_algorithm)
        
        # Create final evaluation environment
        final_eval_env = make_iso_env(
            steps_per_iteration=args.timesteps,
            cost_type=cost_type,
            pricing_policy=pricing_policy,
            demand_pattern=demand_pattern,
            seed=args.seed + 1000,
            log_dir=f"{args.log_dir}/alternating",
            model_dir=f"{args.model_dir}/alternating",
            plot_dir=f"{args.plot_dir}/alternating",
            pcs_policy=final_pcs_model,
            use_dispatch_action=args.use_dispatch,
            norm_path=f"{args.model_dir}/alternating/iso_vecnormalize_best.pkl",
            eval_mode=True,
            demand_data_path=args.demand_data
        )
        
        # Set PCS normalization
        pcs_norm_path = f"{args.model_dir}/alternating/pcs_vecnormalize_best.pkl"
        if os.path.exists(pcs_norm_path):
            pcs_env_norm = make_pcs_env(
                steps_per_iteration=args.timesteps,
                cost_type=cost_type,
                pricing_policy=pricing_policy,
                demand_pattern=demand_pattern,
                seed=args.seed,
                log_dir=f"{args.log_dir}/alternating",
                model_dir=f"{args.model_dir}/alternating",
                plot_dir=f"{args.plot_dir}/alternating",
                iso_policy=None,
                norm_path=pcs_norm_path,
                use_dispatch_action=args.use_dispatch,
                eval_mode=True,
                demand_data_path=args.demand_data
            )
            final_pcs_model.set_env(pcs_env_norm)
        
        final_iso_model.set_env(final_eval_env)
        
        # Final evaluation
        final_results = evaluate_iso(
            iso_model=final_iso_model,
            pcs_model=final_pcs_model,
            env_config={
                'cost_type': cost_type,
                'pricing_policy': pricing_policy,
                'demand_pattern': demand_pattern,
                'use_dispatch_action': args.use_dispatch,
                'demand_data_path': args.demand_data,
            },
            override_env=final_eval_env,
            num_episodes=args.eval_episodes,
            seed=args.seed + 1000,
            plot_dir="eval_plots/alternating/iso",
            is_recurrent=(args.iso_algorithm == "recurrent_ppo")
        )
        
        print(f"Final evaluation complete. Avg ISO reward: {final_results['avg_iso_reward']:.2f}")
        print("Plots saved to: eval_plots/alternating/iso/")
    else:
        print("Best models not found, skipping final evaluation")


if __name__ == "__main__":
    main() 