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
            # For quadratic policy, use polynomial coeffiRescaleActioncient bounds
            poly_config = action_spaces_config.get('quadratic', {}).get('polynomial', {})
            self.min_action = poly_config.get('min', 0.0)
            self.max_action = poly_config.get('max', 100.0)
            
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
    pricing_policy = PricingPolicy.ONLINE  # Set the pricing policy here
    # Attempt to create the environment using gym.make
    try:
        env = gym.make(
            pcs_id,
            disable_env_checker = True,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file,
            pricing_policy=pricing_policy
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
            log_file=log_file,
            pricing_policy=pricing_policy
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
    env_id_iso='ISOEnv-v0',
    total_iterations=None,             
    train_timesteps_per_iteration=None,  
    eval_episodes=5,                 
    log_dir_iso='logs/agent_iso',
    model_save_path_iso='models/agent_iso/agent_iso',
    seed=None,
    trained_pcs_model_path=None,
    pricing_policy=None,
    demand_pattern=None,
    num_pcs_agents=None,  
    cost_type=None
):
    """
    Implements an iterative training process for two agents (ISO and ISO) using different RL algorithms.
    
    Training Process:
    1. Create and configure both environments
    2. Initialize models for both agents
    3. For each iteration:
       - Train ISO agent while using current ISO model
       - Evaluate ISO agent performance
       - Train ISO agent while using current ISO model
       - Evaluate ISO agent performance
    4. Save final models and generate performance plots
    
    Args:
        algo_type (str): Algorithm to use ('PPO', 'A2C')
        env_id_iso (str): Gymnasium environment ID for ISO agent
        env_id_iso (str): Gymnasium environment ID for ISO agent
        total_iterations (int): Number of training iterations
        train_timesteps_per_iteration (int): Steps per training iteration
        eval_episodes (int): Number of evaluation episodes
        log_dir_iso (str): Directory for ISO training logs
        log_dir_iso (str): Directory for ISO training logs
        model_save_path_iso (str): Save path for ISO model
        model_save_path_iso (str): Save path for ISO model
        seed (int): Random seed for reproducibility
        pricing_policy (PricingPolicy): The pricing policy to use (QUADRATIC/ONLINE/CONSTANT)
    
    Results:
    - Saves trained models at specified intervals
    - Generates training and evaluation plots
    - Creates CSV files with evaluation metrics
    """
    # --- Prepare environments for iso
    os.makedirs(log_dir_iso, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path_iso), exist_ok=True)

    # Create base environments with trained model path and pricing policy
    train_env_iso = gym.make(
        env_id_iso,
        trained_pcs_model_path=trained_pcs_model_path,
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,  
        cost_type=cost_type,
        num_pcs_agents=num_pcs_agents
    )
    eval_env_iso = gym.make(
        env_id_iso,
        trained_pcs_model_path=trained_pcs_model_path,
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern, 
        cost_type=cost_type,
        num_pcs_agents=num_pcs_agents
    )

    if algo_type == 'DQN':
        # Wrap with discrete-action wrapper
        train_env_iso = DiscreteActionWrapper(train_env_iso)
        eval_env_iso = DiscreteActionWrapper(eval_env_iso)
    else:
        if pricing_policy == PricingPolicy.ONLINE:
            train_env_iso = RescaleAction(
                train_env_iso, 
                min_action=np.array([1.0, 1.0], dtype=np.float32),
                max_action=np.array([10.0, 10.0], dtype=np.float32)
            )
            eval_env_iso = RescaleAction(
                eval_env_iso,
                min_action=np.array([1.0, 1.0], dtype=np.float32),
                max_action=np.array([10.0, 10.0], dtype=np.float32)
            )
        else:  # QUADRATIC
            train_env_iso = RescaleAction(train_env_iso, min_action=0.0, max_action=100.0)
            eval_env_iso = RescaleAction(eval_env_iso, min_action=0.0, max_action=100.0)

    # Add monitoring
    train_env_iso = Monitor(train_env_iso, filename=os.path.join(log_dir_iso, 'train_monitor_iso.csv'))
    eval_env_iso = Monitor(eval_env_iso, filename=os.path.join(log_dir_iso, 'eval_monitor.csv'))

    train_env_iso.reset(seed=seed)
    train_env_iso.action_space.seed(seed)
    train_env_iso.observation_space.seed(seed)

    eval_env_iso.reset(seed=seed+1)
    eval_env_iso.action_space.seed(seed+1)
    eval_env_iso.observation_space.seed(seed+1)

    # Create vectorized environments with normalization for ISO
    train_env_iso = DummyVecEnv([lambda: train_env_iso])
    train_env_iso = VecNormalize(
        train_env_iso,
        norm_obs=True,
        norm_reward=True,
        clip_obs=1.,
        clip_reward=1.,
        gamma=0.99,
        epsilon=1e-8,
    )

    eval_env_iso = DummyVecEnv([lambda: eval_env_iso])
    eval_env_iso = VecNormalize(
        eval_env_iso,
        norm_obs=True,
        norm_reward=True,
        clip_obs=1.,
        clip_reward=1.,
        gamma=0.99,
        epsilon=1e-8,
    )


    # Copy statistics from training to eval environment
    eval_env_iso.obs_rms = train_env_iso.obs_rms
    eval_env_iso.ret_rms = train_env_iso.ret_rms

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
            return PPO('MlpPolicy', env, verbose=0, seed=seed, tensorboard_log=log_dir, gamma=1)
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
    iso_model = create_model(train_env_iso, log_dir_iso, seed)

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

    iso_reward_callback = RewardCallback("ISO")

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
    iso_reward_callback = RewardCallback("ISO")
    iso_action_tracker = ActionTrackingCallback("ISO")

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
    vec_env_iso = train_env_iso

    # Training loop with model exchange
    print(f"Starting iterative training for {total_iterations} iterations.")
    for iteration in range(total_iterations):
        print("Training ISO, using current ISO model")
        if iteration > 0:
            # Reload normalization statistics and update model environment
            new_normalizer = VecNormalize.load(
                f"{model_save_path_iso}_normalizer.pkl",
                DummyVecEnv([lambda: gym.make(env_id_iso, pricing_policy=pricing_policy, demand_pattern=demand_pattern, cost_type=cost_type, num_pcs_agents=num_pcs_agents)])
            )
            new_normalizer.training = True
            new_normalizer.norm_reward = True
            iso_model.set_env(new_normalizer)
            vec_env_iso = new_normalizer

        # Train ISO
        iso_model.learn(
            total_timesteps=train_timesteps_per_iteration, 
            callback=[iso_reward_callback, iso_action_tracker],
            progress_bar=True
        )
        
        # Save current ISO model and normalizer state
        iso_model.save(f"{model_save_path_iso}_iter_{iteration}")
        vec_env_iso.save(f"{model_save_path_iso}_normalizer.pkl")
        
        # Reload the newly saved model
        updated_iso_model = AlgorithmClass.load(f"{model_save_path_iso}_iter_{iteration}")
        
        # Evaluate ISO
        mean_reward_iso, std_reward_iso = evaluate_and_save(
            updated_iso_model, eval_env_iso, log_dir_iso, "ISO", iteration
        )

        iso_action_tracker.plot_episode_results( 
            episode_num=iteration,
            save_path=log_dir_iso
        )

    print("Iterative training completed.")

    # Save final models
    iso_model.save(f"{model_save_path_iso}_final")
    vec_env_iso.save(f"{model_save_path_iso}_normalizer.pkl")  # This should now work
    print(f"Final ISO model saved to {model_save_path_iso}_final.zip")

    # Save normalizer states after training
    vec_env_iso.save(f"{model_save_path_iso}_normalizer.pkl")

  


    def load_env_and_normalizer(env_id, normalizer_path, log_dir, pricing_policy,demand_pattern, cost_type, num_pcs_agents):
        """
        Loads a gym environment along with its VecNormalize normalizer.
        """
        env = gym.make(env_id, pricing_policy=pricing_policy, demand_pattern=demand_pattern, cost_type=cost_type, num_pcs_agents=num_pcs_agents)

        if pricing_policy == PricingPolicy.ONLINE:
            env = RescaleAction(
                env,
                min_action=np.array([1.0, 1.0], dtype=np.float32),
                max_action=np.array([10.0, 10.0], dtype=np.float32)
            )
        else:
            env = RescaleAction(env, min_action=0.0, max_action=100.0)
        
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
    plot_eval_rewards(log_dir_iso, "ISO")

    print("Training and evaluation process completed.")

    iso_eval_env = load_env_and_normalizer(env_id_iso, f"{model_save_path_iso}_normalizer.pkl", log_dir_iso, pricing_policy, demand_pattern, cost_type, num_pcs_agents)
    
    if algo_type == 'PPO':
        iso_model_final = PPO.load(f"{model_save_path_iso}_final.zip", env=iso_eval_env)
    elif algo_type == 'A2C':
        iso_model_final = A2C.load(f"{model_save_path_iso}_final.zip", env=iso_eval_env)
    elif algo_type == 'DQN':
        iso_model_final = DQN.load(f"{model_save_path_iso}_final.zip", env=iso_eval_env)
    elif algo_type == 'DDPG':
        iso_model_final = DDPG.load(f"{model_save_path_iso}_final.zip", env=iso_eval_env)
    elif algo_type == 'SAC':
        iso_model_final = SAC.load(f"{model_save_path_iso}_final.zip", env=iso_eval_env)
    elif algo_type == 'TD3':
        iso_model_final = TD3.load(f"{model_save_path_iso}_final.zip", env=iso_eval_env)
    else:
        raise ValueError(f"Unsupported algorithm type: {algo_type}")

    mean_reward_iso_final, std_reward_iso_final = evaluate_policy(
        iso_model_final, 
        iso_eval_env, 
        n_eval_episodes=10, 
        deterministic=True
    )
    print(f"Final ISO Model - Mean Reward: {mean_reward_iso_final} +/- {std_reward_iso_final}")


if __name__ == "__main__":
    import argparse
    from iso_game_main import train_and_evaluate_agent, PricingPolicy
    from energy_net.dynamics.iso.demand_patterns import DemandPattern

    parser = argparse.ArgumentParser(description="Train and Evaluate Agent")
    parser.add_argument("--algo_type", default="PPO", help="Algorithm type, e.g. PPO")
    parser.add_argument("--trained_pcs_model_path", required=True, help="Path to the trained PCSs model")
    parser.add_argument("--pricing_policy", required=True, help="Pricing policy: QUADRATIC, ONLINE, or CONSTANT")
    parser.add_argument("--total_iterations", type=int, default=10, help="Total iterations for training")
    parser.add_argument("--train_timesteps_per_iteration", type=int, default=10000, help="Timesteps per iteration")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num_pcs_agents", type=int, default=1, help="Number of agents")
    
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

    # Convert demand pattern string to enum
    cost_type = CostType[args.cost_type.upper()]

    demand_pattern = DemandPattern[args.demand_pattern.upper()]

    # Directly call train_and_evaluate_agent with the parsed arguments:
    train_and_evaluate_agent(
        cost_type=cost_type,
        algo_type=args.algo_type,
        trained_pcs_model_path=args.trained_pcs_model_path,
        pricing_policy=pricing_policy_enum,
        demand_pattern=demand_pattern,  
        total_iterations=args.total_iterations,
        train_timesteps_per_iteration=args.train_timesteps_per_iteration,
        eval_episodes=args.eval_episodes,
        num_pcs_agents=args.num_pcs_agents,
        seed=args.seed
    )
