import gymnasium as gym
import os
import sys
import energy_net.env

from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from energy_net.utils.callbacks import ActionTrackingCallback
from gymnasium.wrappers import RescaleAction
from gymnasium import spaces
import numpy as np
from energy_net.env import PricingPolicy
from energy_net.dynamics.iso.demand_patterns import DemandPattern
from energy_net.dynamics.iso.cost_types import CostType



class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_actions=21):
        super().__init__(env)
        self.n_actions = n_actions
        
        # Get pricing policy and config from the environment's controller
        pricing_policy = env.controller.pricing_policy
        action_spaces_config = env.controller.iso_config.get('action_spaces', {})
        
        if pricing_policy == PricingPolicy.ONLINE:
            # For online policy, use price bounds
            price_config = action_spaces_config.get('online', {}).get('buy_price', {})
            self.min_action = price_config.get('min', 1.0)
            self.max_action = price_config.get('max', 10.0)
        else:  # QUADRATIC
            # For quadratic policy, use polynomial coefficient bounds
            poly_config = action_spaces_config.get('quadratic', {}).get('polynomial', {})
            self.min_action = poly_config.get('min', 0.0)
            self.max_action = poly_config.get('max', 100.0)
            
        self.action_space = spaces.Discrete(n_actions)
    
    def action(self, action_idx):
        step_size = (self.max_action - self.min_action) / (self.n_actions - 1)
        return np.array([self.min_action + action_idx * step_size], dtype=np.float32)


def evaluate_trained_model(
    trained_model_path=None,
    normalizer_path=None,
    env_id=None, #ISOEnv-v0 ,PCSUnitEnv-v0
    env_config_path='configs/environment_config.yaml',
    iso_config_path='configs/iso_config.yaml',
    pcs_unit_config_path='configs/pcs_unit_config.yaml',
    log_file='logs/eval_environments.log',
    eval_episodes=None,
    algo_type='PPO',
    pricing_policy=None,
    cost_type=None,
    num_pcs_agents=None,
    trained_pcs_model_path=None,
    demand_pattern=None  
):
    """Evaluate a trained model using our existing callbacks"""
    
    # Create base environment with demand pattern
    env = gym.make(
        env_id,
        num_pcs_agents=num_pcs_agents,  
        disable_env_checker=True,
        env_config_path=env_config_path,
        iso_config_path=iso_config_path,
        pcs_unit_config_path=pcs_unit_config_path,
        log_file=log_file,
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        cost_type=cost_type  
    )
    env = Monitor(env, filename=os.path.join('logs', 'evaluation_monitor.csv'))

    # Apply appropriate wrappers based on algorithm type
    if algo_type == 'DQN':
        env = DiscreteActionWrapper(env)
    else:
        if pricing_policy == PricingPolicy.ONLINE:
            env = RescaleAction(
                env,
                min_action=np.array([1.0, 1.0], dtype=np.float32),
                max_action=np.array([10.0, 10.0], dtype=np.float32)
            )
        else:  # For QUADRATIC or other policies
            env = RescaleAction(env, min_action=0.0, max_action=100.0)

    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(normalizer_path, env)
    env.training = False
    env.norm_reward = False
    
    # Load appropriate model type
    if algo_type == 'PPO':
        model = PPO.load(trained_model_path, env=env)
    elif algo_type == 'A2C':
        model = A2C.load(trained_model_path, env=env)
    elif algo_type == 'DQN':
        model = DQN.load(trained_model_path, env=env)
    elif algo_type == 'DDPG':
        model = DDPG.load(trained_model_path, env=env)
    elif algo_type == 'SAC':
        model = SAC.load(trained_model_path, env=env)
    elif algo_type == 'TD3':
        model = TD3.load(trained_model_path, env=env)
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")
    
    # Create callback for tracking
    action_tracker = ActionTrackingCallback(env_id)
    
    # Run evaluation episodes
    for episode in range(eval_episodes):
        obs = env.reset()[0]
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        step_data_list = []
        
        print(f"\nStarting Episode {episode + 1}/{eval_episodes}")
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            
            # Handle different action shapes based on algorithm type
            if algo_type == 'DQN':
                # For DQN, we need to wrap the scalar action in a list for DummyVecEnv
                action = [int(action)]  # Convert to list for vectorized env
            else:  # PPO or A2C
                # Ensure action is 2D array with shape (1, action_dim)
                if len(action.shape) == 1:
                    action = action.reshape(1, -1)
            
            # Take step in environment - handle VecEnv step return format
            vec_obs, vec_reward, vec_done, vec_info = env.step(action)
            
            # Extract values from vectorized returns
            obs = vec_obs
            reward = vec_reward[0] if isinstance(vec_reward, np.ndarray) else vec_reward
            done = vec_done[0] if isinstance(vec_done, np.ndarray) else vec_done
            truncated = False  # VecEnv doesn't return truncated, assume False
            info = vec_info[0] if isinstance(vec_info, (list, tuple)) else vec_info

            # Record original action value for plotting
            if algo_type == 'DQN':
                # Convert discrete action back to continuous value for plotting
                step_size = 20.0 / (21 - 1)  
                action_value = -10.0 + (action[0] * step_size)
            else:
                action_value = float(action[0,0]) if len(action.shape) == 2 else float(action[0])
            
            # Create step data using values from info
            step_data = {
                'step': step,
                'action': action_value,  # Store the continuous equivalent
                'iso_sell_price': info.get('iso_sell_price', 0),
                'iso_buy_price': info.get('iso_buy_price', 0),
                'battery_level': info.get('battery_level', 0),
                'net_exchange': info.get('net_exchange', 0),
                'production': info.get('production', 0),
                'consumption': info.get('consumption', 0),
                'predicted_demand': info.get('predicted_demand', 0),
                'realized_demand': info.get('realized_demand', 0),
                'dispatch_cost': info.get('dispatch_cost', 0),
                'reserve_cost': info.get('reserve_cost', 0),
                'dispatch': info.get('dispatch', 0),
                'reward': float(reward)
            }
            
            step_data_list.append(step_data)
            episode_reward += float(reward)
            step += 1
            
        print(f"Episode {episode + 1} completed - Reward: {episode_reward:.2f}")
        
        # Add complete episode data to tracker
        action_tracker.all_episodes_actions.append(step_data_list)
        
        # Create evaluation directory if it doesn't exist
        os.makedirs('evaluation_results', exist_ok=True)
        # Plot results for this episode
        action_tracker.plot_episode_results(episode, 'evaluation_results')

    print("Evaluation completed - Check evaluation_results directory for plots")
    
   
if __name__ == "__main__":
    import argparse
    from eval_agent import evaluate_trained_model, PricingPolicy
    from energy_net.dynamics.iso.demand_patterns import DemandPattern

    parser = argparse.ArgumentParser(description="Evaluate Agent")
    parser.add_argument("--algo_type", default="PPO", help="Algorithm type, e.g. PPO")
    parser.add_argument("--trained_pcs_model_path", required=True, help="Path to the trained PCSs model")
    parser.add_argument("--trained_model_path", required=True, help="Path to the trained model")
    parser.add_argument("--normalizer_path", required=True, help="Path to the normalizer")
    parser.add_argument("--pricing_policy", required=True, help="Pricing policy: QUADRATIC, ONLINE, or CONSTANT")
    parser.add_argument("--num_pcs_agents", type=int, default=1, help="Number of agents")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument(
        "--demand_pattern",
        default="SINUSOIDAL",
        choices=["SINUSOIDAL", "CONSTANT", "DOUBLE_PEAK"],
        help="Demand pattern type"
    )

    parser.add_argument(
        "--cost_type",
        default="CONSTANT",
        choices=["CONSTANT"], 
        help="Cost structure type"
    )

    
    args = parser.parse_args()

    # Convert demand pattern string to enum
    demand_pattern = DemandPattern[args.demand_pattern.upper()]
    cost_type = CostType[args.cost_type.upper()]

    # Convert pricing_policy argument (a string) to the corresponding enum value:
    policy = args.pricing_policy.upper()
    if policy == "QUADRATIC":
        pricing_policy_enum = PricingPolicy.QUADRATIC
    elif policy == "ONLINE":
        pricing_policy_enum = PricingPolicy.ONLINE
    elif policy == "CONSTANT":
        pricing_policy_enum = PricingPolicy.CONSTANT
    else:
        raise ValueError("Invalid pricing_policy value provided. Use QUADRATIC, ONLINE, or CONSTANT.")

    evaluate_trained_model(
        algo_type=args.algo_type,
        cost_type=cost_type,
        trained_pcs_model_path=args.trained_pcs_model_path,
        trained_model_path=args.trained_model_path,
        normalizer_path=args.normalizer_path,
        pricing_policy=pricing_policy_enum,
        demand_pattern=demand_pattern,  
        env_id='ISOEnv-v0', #ISOEnv-v0 ,PCSUnitEnv-v0
        num_pcs_agents=args.num_pcs_agents,
        eval_episodes=args.eval_episodes
    )

