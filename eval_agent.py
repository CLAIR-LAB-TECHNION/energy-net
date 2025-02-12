import gymnasium as gym
import os
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from energy_net.utils.callbacks import ActionTrackingCallback
from gymnasium.wrappers import RescaleAction
from gymnasium import spaces
import numpy as np

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_actions=21, min_action=-10.0, max_action=10.0):
        super().__init__(env)
        self.n_actions = n_actions
        self.min_action = min_action
        self.max_action = max_action
        self.action_space = spaces.Discrete(n_actions)
    
    def action(self, action_idx):
        step_size = (self.max_action - self.min_action) / (self.n_actions - 1)
        return np.array([self.min_action + action_idx * step_size], dtype=np.float32)

def evaluate_trained_model(
    model_path='/Users/matanlevi/energy-net/models/agent_iso/agent_iso_final.zip',
    normalizer_path='/Users/matanlevi/energy-net/models/agent_iso/agent_iso_normalizer.pkl',
    env_id='ISOEnv-v0',
    env_config_path='configs/environment_config.yaml',
    iso_config_path='configs/iso_config.yaml',
    pcs_unit_config_path='configs/pcs_unit_config.yaml',
    log_file='logs/eval_environments.log',
    num_episodes=5,
    algo_type='PPO' 
):
    """Evaluate a trained model using our existing callbacks"""
    
    # Create base environment
    env = gym.make(
        env_id,
        disable_env_checker=True,
        env_config_path=env_config_path,
        iso_config_path=iso_config_path,
        pcs_unit_config_path=pcs_unit_config_path,
        log_file=log_file
    )
    env = Monitor(env, filename=os.path.join('logs', 'evaluation_monitor.csv'))

    # Apply appropriate wrappers based on algorithm type
    if algo_type == 'DQN':
        env = DiscretizeActionWrapper(env)
    else:  # PPO or A2C
        env = RescaleAction(env, min_action=-10, max_action=10)

    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(normalizer_path, env)
    env.training = False
    env.norm_reward = False
    
    # Load appropriate model type
    if algo_type == 'PPO':
        model = PPO.load(model_path, env=env)
    elif algo_type == 'A2C':
        model = A2C.load(model_path, env=env)
    elif algo_type == 'DQN':
        model = DQN.load(model_path, env=env)
    elif algo_type == 'DDPG':
        model = DDPG.load(model_path, env=env)
    elif algo_type == 'SAC':
        model = SAC.load(model_path, env=env)
    elif algo_type == 'TD3':
        model = TD3.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")
    
    # Create callback for tracking
    action_tracker = ActionTrackingCallback(env_id)
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        obs = env.reset()[0]
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        step_data_list = []
        
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        
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
                step_size = 20.0 / (21 - 1)  # -10 to +10 range with 21 steps
                action_value = -10.0 + (action[0] * step_size)
            else:
                action_value = float(action[0,0]) if len(action.shape) == 2 else float(action[0])
            
            # Create step data using values from info
            step_data = {
                'step': step,
                'action': action_value,  # Store the continuous equivalent
                'buy_price': info.get('buy_price', 0),
                'sell_price': info.get('sell_price', 0),
                'battery_level': info.get('battery_level', 0),
                'net_exchange': info.get('net_exchange', 0),
                'production': info.get('production', 0),
                'consumption': info.get('consumption', 0),
                'predicted_demand': info.get('predicted_demand', 0),
                'realized_demand': info.get('realized_demand', 0),
                'dispatch_cost': info.get('dispatch_cost', 0),
                'reserve_cost': info.get('reserve_cost', 0),
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
    # Example usage with different algorithms
   # evaluate_trained_model(
   #     algo_type='PPO',
   #     model_path='models/agent_pcs/ppo_model.zip',
   #     normalizer_path='models/agent_pcs/ppo_normalizer.pkl'
  #  )
    
   # evaluate_trained_model(
   #     algo_type='A2C',
   #     model_path='/Users/matanlevi/energy-net/models/agent_pcs/agent_pcs_iter_0.zip',
   #     normalizer_path='/Users/matanlevi/energy-net/models/agent_pcs/agent_pcs_normalizer.pkl'
   # )
    
    evaluate_trained_model(
        algo_type='PPO',
        model_path='/Users/matanlevi/energy-net/models/agent_iso/agent_iso_final.zip',
        normalizer_path='/Users/matanlevi/energy-net/models/agent_iso/agent_iso_normalizer.pkl'
    )
    
   

