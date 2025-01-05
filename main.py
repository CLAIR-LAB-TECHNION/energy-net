import gymnasium as gym
import energy_net.env
import os
import pandas as pd

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

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
    env_id_iso='ISOEnv-v0',
    total_iterations=10,             
    train_timesteps_per_iteration=1000, 
    eval_episodes=10,                 
    log_dir_pcs='logs/agent_pcs',
    log_dir_iso='logs/agent_iso',
    model_save_path_pcs='models/agent_pcs/agent_pcs',
    model_save_path_iso='models/agent_iso/agent_iso',
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

    train_env_pcs = gym.make(env_id_pcs)
    train_env_pcs = Monitor(train_env_pcs,
                               filename=os.path.join(log_dir_pcs, 'train_monitor_pcs.csv'),
                               allow_early_resets=True)

    eval_env_pcs = gym.make(env_id_pcs)
    eval_env_pcs = Monitor(eval_env_pcs,
                              filename=os.path.join(log_dir_pcs, 'eval_monitor.csv'), 
                              allow_early_resets=True)

    train_env_pcs.reset(seed=seed)
    train_env_pcs.action_space.seed(seed)
    train_env_pcs.observation_space.seed(seed)

    eval_env_pcs.reset(seed=seed+1)
    eval_env_pcs.action_space.seed(seed+1)
    eval_env_pcs.observation_space.seed(seed+1)


    # --- Prepare environments for iso
    os.makedirs(log_dir_iso, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path_iso), exist_ok=True)

    train_env_iso = gym.make(
        env_id_iso,
        reward_type='iso',  # Explicitly set reward type for ISO env
        disable_env_checker=True
    )
    train_env_iso = Monitor(train_env_iso,
                               filename=os.path.join(log_dir_iso, 'train_monitor_iso.csv'),
                               allow_early_resets=True)

    eval_env_iso = gym.make(
        env_id_iso,
        reward_type='iso',  # Also set for eval env
        disable_env_checker=True
    )
    eval_env_iso = Monitor(eval_env_iso,
                              filename=os.path.join(log_dir_iso, 'eval_monitor.csv'),  
                              allow_early_resets=True)

    train_env_iso.reset(seed=seed+2)
    train_env_iso.action_space.seed(seed+2)
    train_env_iso.observation_space.seed(seed+2)

    eval_env_iso.reset(seed=seed+3)
    eval_env_iso.action_space.seed(seed+3)
    eval_env_iso.observation_space.seed(seed+3)

    # Create algorithm instances based on type
    def create_model(env, log_dir, seed):
        if algo_type == 'PPO':
            return PPO('MlpPolicy', env, verbose=1, seed=seed, tensorboard_log=log_dir,learning_rate=0.01, ent_coef=0.01, clip_range=0.3, batch_size=128)
        elif algo_type == 'A2C':
            return A2C('MlpPolicy', env, verbose=1, seed=seed, tensorboard_log=log_dir)
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type}")

    # Initialize models
    pcs_model = create_model(train_env_pcs, log_dir_pcs, seed)
    iso_model = create_model(train_env_iso, log_dir_iso, seed+10)

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
    iso_reward_callback = RewardCallback("ISO")

    # Save evaluation results directly during training
    def evaluate_and_save(model, eval_env, log_dir, agent_name):
        mean_reward, std_reward = evaluate_policy(
            model, 
            eval_env, 
            n_eval_episodes=eval_episodes, 
            deterministic=True
        )
        
        # Save evaluation results to CSV
        with open(os.path.join(log_dir, 'eval_results.csv'), 'a') as f:
            if f.tell() == 0:  # If file is empty, write header
                f.write('iteration,mean_reward,std_reward\n')
            f.write(f'{iteration},{mean_reward},{std_reward}\n')
            
        print(f"[{agent_name}] Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    # Training loop with model exchange
    print(f"Starting iterative training for {total_iterations} iterations.")
    for iteration in range(total_iterations):
        print("=" * 60)
        print(f"Iteration {iteration + 1} of {total_iterations}")

        # 1. Train PCS while using current ISO model
        print("Training PCS, using current ISO model")
        if iteration > 0:  # Not first iteration
            # Load current ISO model into PCS environment
            train_env_pcs = gym.make(
                env_id_pcs,
                trained_iso_model_path=f"{model_save_path_iso}_iter_{iteration-1}.zip"
            )
            train_env_pcs = Monitor(train_env_pcs, filename=os.path.join(log_dir_pcs, f'train_monitor_pcs_{iteration}.csv'))
            
            # Update the environment in the existing model
            pcs_model.set_env(train_env_pcs)

        # Train PCS
        pcs_model.learn(
            total_timesteps=train_timesteps_per_iteration, 
            callback=pcs_reward_callback,
            progress_bar=True
        )
        
        # Save current PCS model
        pcs_model.save(f"{model_save_path_pcs}_iter_{iteration}")
        
        # Reload the newly saved model
        updated_pcs_model = PPO.load(f"{model_save_path_pcs}_iter_{iteration}")
        
        # Evaluate PCS
        mean_reward_pcs, std_reward_pcs = evaluate_and_save(
            updated_pcs_model, eval_env_pcs, log_dir_pcs, "PCS"
        )

        # 2. Train ISO while using current PCS model
        print("Training ISO, using current PCS model")
        # Load current PCS model into ISO environment
        train_env_iso = gym.make(
            env_id_iso,
            trained_pcs_model_path=f"{model_save_path_pcs}_iter_{iteration}.zip"
        )
        train_env_iso = Monitor(train_env_iso, filename=os.path.join(log_dir_iso, f'train_monitor_iso_{iteration}.csv'))
        
        # Update the environment in the existing model
        iso_model.set_env(train_env_iso)

        # Train ISO
        iso_model.learn(
            total_timesteps=train_timesteps_per_iteration, 
            callback=iso_reward_callback,
            progress_bar=True
        )
        
        # Save current ISO model
        iso_model.save(f"{model_save_path_iso}_iter_{iteration}")

        # Reload the newly saved model
        updated_iso_model = PPO.load(f"{model_save_path_iso}_iter_{iteration}")

        # Evaluate ISO
        mean_reward_iso, std_reward_iso = evaluate_and_save(
            updated_iso_model, eval_env_iso, log_dir_iso, "ISO"
        )

    print("Iterative training completed.")

    # Close environments
    train_env_pcs.close()
    eval_env_pcs.close()
    train_env_iso.close()
    eval_env_iso.close()

    # Save final models
    pcs_model.save(f"{model_save_path_pcs}_final")
    iso_model.save(f"{model_save_path_iso}_final")
    print(f"Final PCS model saved to {model_save_path_pcs}_final.zip")
    print(f"Final ISO model saved to {model_save_path_iso}_final.zip")

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
    plot_rewards(iso_reward_callback.rewards, "ISO", log_dir_iso)

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
    plot_eval_rewards(log_dir_iso, "ISO")

    print("Training and evaluation process completed.")

if __name__ == "__main__":
    # Example usage with different algorithms
    train_and_evaluate_agent(algo_type='PPO') 