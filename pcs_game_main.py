import gymnasium as gym
import energy_net.env
import os
import pandas as pd
import numpy as np

from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from energy_net.utils.callbacks import ActionTrackingCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RescaleAction, ClipAction
from gymnasium import spaces
from stable_baselines3.common.noise import NormalActionNoise
from energy_net.dynamics.iso.demand_patterns import DemandPattern
import argparse

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_actions=21, min_action=-10.0, max_action=10.0):
        super().__init__(env)
        self.n_actions = n_actions
        self.min_action = min_action
        self.max_action = max_action
        self.action_space = spaces.Discrete(n_actions)
    
    def action(self, action_idx):
        step_size = (self.max_action - self.min_action) / (self.n_actions - 1)
        return np.array([self.min_action + action_idx * step_size], dtype=np.float32)

def main():
    """
    Main function that demonstrates basic environment interaction with both PCSUnitEnv and ISOEnv.
    This function:
    1. Creates and configures both environments
    2. Runs a basic simulation with random actions
    3. Renders the environment state (if implemented)
    4. Prints observations, rewards, and other information
    
    The simulation runs until a terminal state is reached or the environment
    signals truncation.
    """
    # Define configuration paths (update paths as necessary)
    env_config_path = 'configs/environment_config.yaml'
    iso_config_path = 'configs/iso_config.yaml'
    pcs_unit_config_path = 'configs/pcs_unit_config.yaml'
    log_file = 'logs/environments.log'
    pcs_id = 'PCSUnitEnv-v0'
    iso_id = 'ISOEnv-v0'
    # Attempt to create the environment using gym.make
    try:
        env = gym.make(
            pcs_id,
            disable_env_checker = True,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file
        )
    except gym.error.UnregisteredEnv:
        print("Error: The environment '{env_id}' is not registered. Please check your registration.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while creating the environment: {e}")
        return

    # Reset the environment to obtain the initial observation and info
    observation, info = env.reset()

    done = False
    truncated = False

    print("Starting PCSUnitEnv Simulation...")

    while not done and not truncated:
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take a step in the environment using the sampled action
        observation, reward, done, truncated, info = env.step(action)
        print(f"PCS Step | Obs: {observation}, Reward: {reward}, Done: {done}, Trunc: {truncated}, Info: {info}")
        
        # Render the current state (if implemented)
        try:
            env.render()
        except NotImplementedError:
            pass  # Render not implemented; skip
        
        # Print observation, reward, and additional info
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print("-" * 50)

    print("Simulation completed.")

    # Close the environment to perform any necessary cleanup
    env.close()

        # Attempt to create the environment using gym.make
    try:
        env = gym.make(
            iso_id,
            disable_env_checker = True,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file
        )
    except gym.error.UnregisteredEnv:
        print("Error: The environment '{env_id}' is not registered. Please check your registration.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while creating the environment: {e}")
        return

    # Reset the environment to obtain the initial observation and info
    observation, info = env.reset()

    done = False
    truncated = False

    print("Starting ISOEnv Simulation...")

    while not done and not truncated:
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take a step in the environment using the sampled action
        observation, reward, done, truncated, info = env.step(action)
        print(f"ISO Step | Obs: {observation}, Reward: {reward}, Done: {done}, Trunc: {truncated}, Info: {info}")
        
        # Render the current state (if implemented)
        try:
            env.render()
        except NotImplementedError:
            pass  # Render not implemented; skip
        
        # Print observation, reward, and additional info
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print("-" * 50)

    print("Simulation completed.")

    # Close the environment to perform any necessary cleanup
    env.close()
    
    
def train_and_evaluate_agent(
    algo_type='PPO',
    env_id_pcs='PCSUnitEnv-v0',
    demand_pattern=DemandPattern.SINUSOIDAL,  
    total_iterations=100,             
    train_timesteps_per_iteration=48*400,  
    eval_episodes=5,                 
    log_dir_pcs='logs/agent_pcs',
    model_save_path_pcs='models/agent_pcs/agent_pcs',
    seed=42
):
    """
    Implements an iterative training process for two agents (ISO and PCS) using different RL algorithms.
    
    Training Process:
    1. Create and configure both environments
    2. Initialize models for both agents
    3. For each iteration:
       - Train PCS agent while using current ISO model
       - Evaluate PCS agent performance
       - Train ISO agent while using current PCS model
       - Evaluate ISO agent performance
    4. Save final models and generate performance plots
    
    Args:
        algo_type (str): Algorithm to use ('PPO', 'A2C')
        env_id_pcs (str): Gymnasium environment ID for PCS agent
        env_id_iso (str): Gymnasium environment ID for ISO agent
        total_iterations (int): Number of training iterations
        train_timesteps_per_iteration (int): Steps per training iteration
        eval_episodes (int): Number of evaluation episodes
        log_dir_pcs (str): Directory for PCS training logs
        log_dir_iso (str): Directory for ISO training logs
        model_save_path_pcs (str): Save path for PCS model
        model_save_path_iso (str): Save path for ISO model
        seed (int): Random seed for reproducibility
    
    Results:
    - Saves trained models at specified intervals
    - Generates training and evaluation plots
    - Creates CSV files with evaluation metrics
    """
    # --- Prepare environments for pcs
    os.makedirs(log_dir_pcs, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path_pcs), exist_ok=True)

    # Create base environments with pattern
    train_env_pcs = gym.make(
        env_id_pcs,
        demand_pattern=demand_pattern  
    )
    eval_env_pcs = gym.make(
        env_id_pcs,
        demand_pattern=demand_pattern  
    )

    if algo_type == 'DQN':
        # Wrap with discrete-action wrapper
        train_env_pcs = DiscreteActionWrapper(train_env_pcs)
        eval_env_pcs = DiscreteActionWrapper(eval_env_pcs)
    else:
        train_env_pcs = RescaleAction(train_env_pcs, min_action=-10.0, max_action=10.0)
        eval_env_pcs = RescaleAction(eval_env_pcs, min_action=-10.0, max_action=10.0)
    
    # Add monitoring
    train_env_pcs = Monitor(train_env_pcs, filename=os.path.join(log_dir_pcs, 'train_monitor_pcs.csv'))
    eval_env_pcs = Monitor(eval_env_pcs, filename=os.path.join(log_dir_pcs, 'eval_monitor.csv'))

    train_env_pcs.reset(seed=seed)
    train_env_pcs.action_space.seed(seed)
    train_env_pcs.observation_space.seed(seed)

    eval_env_pcs.reset(seed=seed+1)
    eval_env_pcs.action_space.seed(seed+1)
    eval_env_pcs.observation_space.seed(seed+1)

    # Create vectorized environments with normalization for PCS
    train_env_pcs = DummyVecEnv([lambda: train_env_pcs])
    train_env_pcs = VecNormalize(
        train_env_pcs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=1.,
        clip_reward=1.,
        gamma=0.99,
        epsilon=1e-8,
    )

    eval_env_pcs = DummyVecEnv([lambda: eval_env_pcs])
    eval_env_pcs = VecNormalize(
        eval_env_pcs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=1.,
        clip_reward=1.,
        gamma=0.99,
        epsilon=1e-8,
    )


    # Copy statistics from training to eval environment
    eval_env_pcs.obs_rms = train_env_pcs.obs_rms
    eval_env_pcs.ret_rms = train_env_pcs.ret_rms

    # debug prints for initial observation spaces
  #  print("\nInitial Observation/Action Spaces:")
   # print(f"PCS Raw Observation Space: {train_env_pcs.observation_space}")
    #print(f"PCS Raw Action Space: {train_env_pcs.action_space}")

    # Create algorithm instances based on type
    def create_model(env, log_dir, seed):
        if algo_type == 'DQN':
            n_actions = 21 
            action_space = spaces.Discrete(n_actions)
            
            def action_wrapper(discrete_action):
                return -10.0 + (discrete_action * 1.0) 
                
            return DQN('MlpPolicy', 
                      env, 
                      learning_rate=0.001,
                      buffer_size=100000,
                      learning_starts=1000,
                      batch_size=64,
                      tau=0.001,
                      gamma=0.99,
                      train_freq=1,  # Train every step
                      gradient_steps=1,
                      target_update_interval=48,  # Update target network every episode
                      exploration_fraction=0.2,
                      exploration_initial_eps=2.0,
                      exploration_final_eps=0.2,
                      seed=seed,
                      tensorboard_log=log_dir)
        if algo_type == 'PPO':
            return PPO('MlpPolicy', env, verbose=1, seed=seed, tensorboard_log=log_dir)
        elif algo_type == 'A2C':
            return A2C('MlpPolicy', 
                      env, 
                      verbose=1, 
                      seed=seed, 
                      tensorboard_log=log_dir,
                      n_steps=48)  # Set steps per update to match episode length
        elif algo_type == 'DDPG':
            return DDPG('MlpPolicy', 
                       env,
                       learning_rate=0.001,
                       buffer_size=100000,
                       learning_starts=1000,
                       batch_size=64,
                       tau=0.005,
                       gamma=0.99,
                       train_freq=1,
                       gradient_steps=1,
                       seed=seed,
                       tensorboard_log=log_dir)
        elif algo_type == 'SAC':
            return SAC('MlpPolicy',
                      env,
                      learning_rate=0.001,
                      buffer_size=100000,
                      learning_starts=1000,
                      batch_size=64,
                      tau=0.005,
                      gamma=0.99,
                      train_freq=1,
                      gradient_steps=1,
                      ent_coef='auto',  # Automatically adjust entropy coefficient
                      seed=seed,
                      tensorboard_log=log_dir)
        elif algo_type == 'TD3':
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
            return TD3('MlpPolicy',
                      env,
                      learning_rate=0.001,
                      buffer_size=100000,
                      learning_starts=1000,
                      batch_size=64,
                      tau=0.005,
                      gamma=0.99,
                      train_freq=1,
                      gradient_steps=1,
                      action_noise=action_noise,
                      seed=seed,
                      tensorboard_log=log_dir)
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type}")

    # Initialize models
    pcs_model = create_model(train_env_pcs, log_dir_pcs, seed)

    # Initialize separate reward callbacks for each agent
    class RewardCallback(BaseCallback):
        def __init__(self, agent_name: str, verbose=0):
            super(RewardCallback, self).__init__(verbose)
            self.rewards = []
            self.agent_name = agent_name

        def _on_step(self) -> bool:
            for info in self.locals.get('infos', []):
                if 'episode' in info.keys():
                    self.rewards.append(info['episode']['r'])
            return True

    pcs_reward_callback = RewardCallback("PCS")

    # Save evaluation results directly during training
    def evaluate_and_save(model, eval_env, log_dir, agent_name, iteration):
        """Updated to handle normalized environments"""
        # Don't update normalization statistics during evaluation
        eval_env.training = False
        eval_env.norm_reward = False
        
        mean_reward, std_reward = evaluate_policy(
            model, 
            eval_env, 
            n_eval_episodes=eval_episodes, 
            deterministic=True
        )
        
        # Re-enable updates for training
        eval_env.training = True
        eval_env.norm_reward = True
        
        # Save evaluation results to CSV
        with open(os.path.join(log_dir, 'eval_results.csv'), 'a') as f:
            if f.tell() == 0:  # If file is empty, write header
                f.write('iteration,mean_reward,std_reward\n')
            f.write(f'{iteration},{mean_reward},{std_reward}\n')
            
        print(f"[{agent_name}] Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    # Initialize callbacks
    pcs_reward_callback = RewardCallback("PCS")
    pcs_action_tracker = ActionTrackingCallback("PCS")

    # Create dictionary to map algo_type string to actual class
    algo_classes = {
        'PPO': PPO,
        'A2C': A2C,
        'DQN': DQN,
        'DDPG': DDPG,
        'SAC': SAC,
        'TD3': TD3
    }
    
    # Get the correct algorithm class
    AlgorithmClass = algo_classes[algo_type]


    # Keep track of the latest vectorized environment
    vec_env_pcs = train_env_pcs

    # Training loop with model exchange
    print(f"Starting iterative training for {total_iterations} iterations.")
    for iteration in range(total_iterations):
        print("Training PCS, using current ISO model")
        if iteration > 0:
            # Reload normalization statistics and update model environment
            new_normalizer = VecNormalize.load(
                f"{model_save_path_pcs}_normalizer.pkl",
                DummyVecEnv([lambda: gym.make(env_id_pcs)])
            )
            new_normalizer.training = True
            new_normalizer.norm_reward = True
            pcs_model.set_env(new_normalizer)
            vec_env_pcs = new_normalizer

        # Train PCS
        pcs_model.learn(
            total_timesteps=train_timesteps_per_iteration, 
            callback=[pcs_reward_callback, pcs_action_tracker],
            progress_bar=True
        )
        
        # Save current PCS model and normalizer state
        pcs_model.save(f"{model_save_path_pcs}_iter_{iteration}")
        vec_env_pcs.save(f"{model_save_path_pcs}_normalizer.pkl")
        
        # Reload the newly saved model
        updated_pcs_model = AlgorithmClass.load(f"{model_save_path_pcs}_iter_{iteration}")
        
        # Evaluate PCS
        mean_reward_pcs, std_reward_pcs = evaluate_and_save(
            updated_pcs_model, eval_env_pcs, log_dir_pcs, "PCS", iteration
        )

        pcs_action_tracker.plot_episode_results( 
            episode_num=iteration,
            save_path=log_dir_pcs
        )

    print("Iterative training completed.")

    # Save final models
    pcs_model.save(f"{model_save_path_pcs}_final")
    vec_env_pcs.save(f"{model_save_path_pcs}_normalizer.pkl")  # This should now work
    print(f"Final PCS model saved to {model_save_path_pcs}_final.zip")

    # Save normalizer states after training
    vec_env_pcs.save(f"{model_save_path_pcs}_normalizer.pkl")

  


    def load_env_and_normalizer(env_id, normalizer_path, log_dir, min_action=-10, max_action=10):
        """
        Loads a gym environment along with its VecNormalize normalizer.
        """
        env = gym.make(env_id)
        env = RescaleAction(env, min_action=min_action, max_action=max_action)
        env = Monitor(env, filename=os.path.join(log_dir, 'eval_monitor.csv'))
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(normalizer_path, vec_env)
        vec_env.training = False      
        vec_env.norm_reward = False    
        return vec_env



    # Plot Training Rewards - separate plots for each agent
    def plot_rewards(rewards, agent_name, log_dir):
        if rewards:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards, label=f'{agent_name} Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'{agent_name} Training Rewards over Episodes')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, f'{agent_name.lower()}_training_rewards.png'))
            plt.close()
        else:
            print(f"No training rewards recorded for {agent_name}")

    # Plot rewards for both agents
    plot_rewards(pcs_reward_callback.rewards, "PCS", log_dir_pcs)

    # Plot Evaluation Rewards - separate for each agent
    def plot_eval_rewards(log_dir, agent_name):
        eval_file = os.path.join(log_dir, 'eval_results.csv')
        if os.path.exists(eval_file):
            eval_data = pd.read_csv(eval_file)
            plt.figure(figsize=(12, 6))
            plt.errorbar(
                eval_data['iteration'], 
                eval_data['mean_reward'],
                yerr=eval_data['std_reward'],
                marker='o',
                linestyle='-',
                label=f'{agent_name} Evaluation Reward'
            )
            plt.xlabel('Training Iteration')
            plt.ylabel('Reward')
            plt.title(f'{agent_name} Evaluation Rewards over Training')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, f'{agent_name.lower()}_evaluation_rewards.png'))
            plt.close()
        else:
            print(f"No evaluation results found for {agent_name}")

    # Plot evaluation rewards for both agents
    plot_eval_rewards(log_dir_pcs, "PCS")

    print("Training and evaluation process completed.")

    pcs_eval_env = load_env_and_normalizer(env_id_pcs, f"{model_save_path_pcs}_normalizer.pkl", log_dir_pcs, min_action=-10, max_action=10)
    
    if algo_type == 'PPO':
        pcs_model_final = PPO.load(f"{model_save_path_pcs}_final.zip", env=pcs_eval_env)
    elif algo_type == 'A2C':
        pcs_model_final = A2C.load(f"{model_save_path_pcs}_final.zip", env=pcs_eval_env)
    elif algo_type == 'DQN':
        pcs_model_final = DQN.load(f"{model_save_path_pcs}_final.zip", env=pcs_eval_env)
    elif algo_type == 'DDPG':
        pcs_model_final = DDPG.load(f"{model_save_path_pcs}_final.zip", env=pcs_eval_env)
    elif algo_type == 'SAC':
        pcs_model_final = SAC.load(f"{model_save_path_pcs}_final.zip", env=pcs_eval_env)
    elif algo_type == 'TD3':
        pcs_model_final = TD3.load(f"{model_save_path_pcs}_final.zip", env=pcs_eval_env)
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")

    mean_reward_pcs_final, std_reward_pcs_final = evaluate_policy(
        pcs_model_final, 
        pcs_eval_env, 
        n_eval_episodes=10, 
        deterministic=True
    )
    print(f"Final PCS Model - Mean Reward: {mean_reward_pcs_final} +/- {std_reward_pcs_final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate PCS Agent")
    parser.add_argument("--algo_type", default="PPO", 
                      choices=["PPO", "A2C", "DQN", "DDPG", "SAC", "TD3"],
                      help="Algorithm type to use")
    parser.add_argument("--total_iterations", type=int, default=100,
                      help="Number of training iterations")
    parser.add_argument("--train_timesteps_per_iteration", type=int, default=48*400,
                      help="Number of timesteps per training iteration")
    parser.add_argument("--eval_episodes", type=int, default=5,
                      help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--demand_pattern",
                      default="SINUSOIDAL",
                      choices=["SINUSOIDAL", "CONSTANT", "DOUBLE_PEAK"],
                      help="Demand pattern type")
    
    args = parser.parse_args()
    
    # Convert demand pattern string to enum
    demand_pattern = DemandPattern[args.demand_pattern.upper()]
    
    train_and_evaluate_agent(
        algo_type=args.algo_type,
        total_iterations=args.total_iterations,
        train_timesteps_per_iteration=args.train_timesteps_per_iteration,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        demand_pattern=demand_pattern
    )