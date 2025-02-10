#!/usr/bin/env python3
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


# --- Discrete action wrapper (for algorithms like DQN) ---
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


# --- Simulation Section: Run a short simulation for both environments ---
def main():
    """
    Demonstrates basic simulation of both the PCSUnitEnv and ISOEnv with random actions.
    """
    # Define configuration paths
    env_config_path = 'configs/environment_config.yaml'
    iso_config_path = 'configs/iso_config.yaml'
    pcs_unit_config_path = 'configs/pcs_unit_config.yaml'
    log_file = 'logs/environments.log'
    pcs_id = 'PCSUnitEnv-v0'
    iso_id = 'ISOEnv-v0'

    # --- PCS simulation ---
    try:
        pcs_env = gym.make(
            pcs_id,
            disable_env_checker=True,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file
        )
    except Exception as e:
        print(f"Error creating PCSUnitEnv: {e}")
        return

    obs, info = pcs_env.reset()
    done, truncated = False, False
    print("Starting PCSUnitEnv Simulation...")
    while not done and not truncated:
        action = pcs_env.action_space.sample()
        obs, reward, done, truncated, info = pcs_env.step(action)
        print(f"PCS Step | Obs: {obs}, Reward: {reward}, Info: {info}")
    pcs_env.close()

    # --- ISO simulation ---
    try:
        iso_env = gym.make(
            iso_id,
            disable_env_checker=True,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file
        )
    except Exception as e:
        print(f"Error creating ISOEnv: {e}")
        return

    obs, info = iso_env.reset()
    done, truncated = False, False
    print("Starting ISOEnv Simulation...")
    while not done and not truncated:
        action = iso_env.action_space.sample()
        obs, reward, done, truncated, info = iso_env.step(action)
        print(f"ISO Step | Obs: {obs}, Reward: {reward}, Info: {info}")
    iso_env.close()


# --- Training Section: Ping-Pong Training of PCS and ISO agents ---
def train_and_evaluate_agent(
    algo_type='PPO',
    env_id_pcs='PCSUnitEnv-v0',
    env_id_iso='ISOEnv-v0',
    total_iterations=2,
    train_timesteps_per_iteration=1,
    eval_episodes=5,
    log_dir_pcs='logs/agent_pcs',
    log_dir_iso='logs/agent_iso',
    model_save_path_pcs='models/agent_pcs/agent_pcs',
    model_save_path_iso='models/agent_iso/agent_iso',
    seed=421
):
    """
    Alternating (ping-pong) training for both the PCS and ISO agents.
    In each iteration:
      1. Train the PCS agent (with the current fixed ISO model passed to the PCS environment).
      2. Evaluate the PCS agent.
      3. Then train the ISO agent (with the current fixed PCS model passed to the ISO environment).
      4. Evaluate the ISO agent.
    Normalization, monitoring, callbacks, and evaluation plots are generated.
    """
    # --- Prepare PCS environments ---
    os.makedirs(log_dir_pcs, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path_pcs), exist_ok=True)
    train_env_pcs = gym.make(env_id_pcs)
    eval_env_pcs = gym.make(env_id_pcs)
    # Rescale PCS actions to [-10, 10]
    train_env_pcs = RescaleAction(train_env_pcs, min_action=-10.0, max_action=10.0)
    eval_env_pcs = RescaleAction(eval_env_pcs, min_action=-10.0, max_action=10.0)
    train_env_pcs = Monitor(train_env_pcs, filename=os.path.join(log_dir_pcs, 'train_monitor_pcs.csv'))
    eval_env_pcs = Monitor(eval_env_pcs, filename=os.path.join(log_dir_pcs, 'eval_monitor.csv'))
    train_env_pcs.reset(seed=seed)
    train_env_pcs.action_space.seed(seed)
    train_env_pcs.observation_space.seed(seed)
    eval_env_pcs.reset(seed=seed+1)
    eval_env_pcs.action_space.seed(seed+1)
    eval_env_pcs.observation_space.seed(seed+1)
    train_env_pcs = DummyVecEnv([lambda: train_env_pcs])
    train_env_pcs = VecNormalize(
        train_env_pcs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
        epsilon=1e-8,
    )
    eval_env_pcs = DummyVecEnv([lambda: eval_env_pcs])
    eval_env_pcs = VecNormalize(
        eval_env_pcs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
        epsilon=1e-8,
    )
    eval_env_pcs.obs_rms = train_env_pcs.obs_rms
    eval_env_pcs.ret_rms = train_env_pcs.ret_rms

    # --- Prepare ISO environments ---
    os.makedirs(log_dir_iso, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path_iso), exist_ok=True)
    train_env_iso = gym.make(env_id_iso, reward_type='iso')
    eval_env_iso = gym.make(env_id_iso, reward_type='iso')
    # Rescale ISO actions to [1, 10]
    train_env_iso = RescaleAction(train_env_iso, min_action=1.0, max_action=10.0)
    eval_env_iso = RescaleAction(eval_env_iso, min_action=1.0, max_action=10.0)
    train_env_iso = Monitor(train_env_iso, filename=os.path.join(log_dir_iso, 'train_monitor_iso.csv'))
    eval_env_iso = Monitor(eval_env_iso, filename=os.path.join(log_dir_iso, 'eval_monitor.csv'))
    train_env_iso.reset(seed=seed+2)
    train_env_iso.action_space.seed(seed+2)
    train_env_iso.observation_space.seed(seed+2)
    eval_env_iso.reset(seed=seed+3)
    eval_env_iso.action_space.seed(seed+3)
    eval_env_iso.observation_space.seed(seed+3)
    train_env_iso = DummyVecEnv([lambda: train_env_iso])
    train_env_iso = VecNormalize(
        train_env_iso,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
        epsilon=1e-8,
    )
    eval_env_iso = DummyVecEnv([lambda: eval_env_iso])
    eval_env_iso = VecNormalize(
        eval_env_iso,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
        epsilon=1e-8,
    )
    eval_env_iso.obs_rms = train_env_iso.obs_rms
    eval_env_iso.ret_rms = train_env_iso.ret_rms

    # --- Create models ---
    def create_model(env, log_dir, seed):
        if algo_type == 'PPO':
            return PPO('MlpPolicy', env, verbose=1, seed=seed, tensorboard_log=log_dir, n_steps=48)
        elif algo_type == 'A2C':
            return A2C('MlpPolicy', env, verbose=1, seed=seed, tensorboard_log=log_dir)
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type}")

    pcs_model = create_model(train_env_pcs, log_dir_pcs, seed)
    iso_model = create_model(train_env_iso, log_dir_iso, seed+10)

    # --- Reward Callback ---
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

    # --- Action Tracking Callbacks ---
    pcs_action_tracker = ActionTrackingCallback("PCS")
    iso_action_tracker = ActionTrackingCallback("ISO")

    # --- Evaluation helper ---
    def evaluate_and_save(model, eval_env, log_dir, agent_name, iteration):
        eval_env.training = False
        eval_env.norm_reward = False
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
        eval_env.training = True
        eval_env.norm_reward = True
        with open(os.path.join(log_dir, 'eval_results.csv'), 'a') as f:
            if f.tell() == 0:
                f.write('iteration,mean_reward,std_reward\n')
            f.write(f'{iteration},{mean_reward},{std_reward}\n')
        print(f"[{agent_name}] Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    # --- Normalization verification helper (optional) ---
    def verify_normalization(env, name, obs, action):
        if hasattr(env, 'normalize_obs'):
            norm_obs = env.normalize_obs(obs)
        if hasattr(env, 'normalize_reward'):
            raw_reward = 1.0
            norm_reward = env.normalize_reward(raw_reward)
        try:
            assert np.all(obs >= env.observation_space.low), f"{name} observation below bounds: {obs} < {env.observation_space.low}"
            assert np.all(obs <= env.observation_space.high), f"{name} observation above bounds: {obs} > {env.observation_space.high}"
            assert np.all(np.abs(norm_obs) <= 1.0), f"{name} normalized observation outside [-1,1]: {norm_obs}"
            assert np.all(np.abs(action) <= 1.0), f"{name} action outside [-1,1]: {action}"
        except AssertionError as e:
            print(f"Warning: {e}")

    # --- Training Loop: Alternate training between PCS and ISO ---
    total_iters = total_iterations
    # Initialize vectorized environment pointers
    vec_env_pcs = train_env_pcs
    vec_env_iso = train_env_iso

    for iteration in range(total_iters):
        print("=" * 60)
        print(f"Iteration {iteration+1}/{total_iters}")

        # ---- Train PCS agent (with fixed ISO model) ----
        print("Training PCS (with fixed ISO model)")
        if iteration > 0:
            # Load the latest ISO model and inject it into the PCS environment
            iso_model_path = f"{model_save_path_iso}_iter_{iteration-1}.zip"
            print(f"Loading ISO model from: {iso_model_path} into PCS environment")
            new_env_pcs = gym.make(env_id_pcs, trained_iso_model_path=iso_model_path)
            new_env_pcs = Monitor(new_env_pcs, filename=os.path.join(log_dir_pcs, f'train_monitor_pcs_{iteration}.csv'))
            new_env_pcs = RescaleAction(new_env_pcs, min_action=-10.0, max_action=10.0)
            new_vec_env_pcs = DummyVecEnv([lambda: new_env_pcs])
            new_vec_env_pcs = VecNormalize(
                new_vec_env_pcs,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.,
                clip_reward=10.,
                gamma=0.99,
                epsilon=1e-8,
            )
            new_vec_env_pcs.obs_rms = train_env_pcs.obs_rms
            new_vec_env_pcs.ret_rms = train_env_pcs.ret_rms
            pcs_model.set_env(new_vec_env_pcs)
            vec_env_pcs = new_vec_env_pcs
        else:
            vec_env_pcs = train_env_pcs

        pcs_model.learn(
            total_timesteps=train_timesteps_per_iteration,
            callback=[pcs_reward_callback, pcs_action_tracker],
            progress_bar=True
        )
        pcs_model.save(f"{model_save_path_pcs}_iter_{iteration}")
        vec_env_pcs.save(f"{model_save_path_pcs}_normalizer.pkl")
        updated_pcs_model = type(pcs_model).load(f"{model_save_path_pcs}_iter_{iteration}")
        mean_reward_pcs, std_reward_pcs = evaluate_and_save(updated_pcs_model, eval_env_pcs, log_dir_pcs, "PCS", iteration)
        pcs_action_tracker.plot_episode_results(episode_num=iteration, save_path=log_dir_pcs)

        # ---- Train ISO agent (with fixed PCS model) ----
        print("Training ISO (with fixed PCS model)")
        pcs_model_path = f"{model_save_path_pcs}_iter_{iteration}.zip"
        print(f"Loading PCS model from: {pcs_model_path} into ISO environment")
        new_env_iso = gym.make(env_id_iso, trained_pcs_model_path=pcs_model_path)
        new_env_iso = Monitor(new_env_iso, filename=os.path.join(log_dir_iso, f'train_monitor_iso_{iteration}.csv'))
        new_env_iso = RescaleAction(new_env_iso, min_action=1.0, max_action=10.0)
        new_vec_env_iso = DummyVecEnv([lambda: new_env_iso])
        new_vec_env_iso = VecNormalize(
            new_vec_env_iso,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.,
            gamma=0.99,
            epsilon=1e-8,
        )
        new_vec_env_iso.obs_rms = train_env_iso.obs_rms
        new_vec_env_iso.ret_rms = train_env_iso.ret_rms
        iso_model.set_env(new_vec_env_iso)
        vec_env_iso = new_vec_env_iso

        iso_model.learn(
            total_timesteps=train_timesteps_per_iteration,
            callback=[iso_reward_callback, iso_action_tracker],
            progress_bar=True
        )
        iso_model.save(f"{model_save_path_iso}_iter_{iteration}")
        vec_env_iso.save(f"{model_save_path_iso}_normalizer.pkl")
        updated_iso_model = type(iso_model).load(f"{model_save_path_iso}_iter_{iteration}")
        mean_reward_iso, std_reward_iso = evaluate_and_save(updated_iso_model, eval_env_iso, log_dir_iso, "ISO", iteration)
        iso_action_tracker.plot_episode_results(episode_num=iteration, save_path=log_dir_iso)

        # Optionally, verify normalization for both agents
        sample_obs_pcs = train_env_pcs.reset()[0]
        sample_action_pcs = train_env_pcs.action_space.sample()
        verify_normalization(train_env_pcs, "PCS", sample_obs_pcs, sample_action_pcs)
        sample_obs_iso = train_env_iso.reset()[0]
        sample_action_iso = train_env_iso.action_space.sample()
        verify_normalization(train_env_iso, "ISO", sample_obs_iso, sample_action_iso)

    print("Iterative training completed.")

    # --- Save final models and normalizers ---
    pcs_model.save(f"{model_save_path_pcs}_final")
    iso_model.save(f"{model_save_path_iso}_final")
    vec_env_pcs.save(f"{model_save_path_pcs}_normalizer.pkl")
    vec_env_iso.save(f"{model_save_path_iso}_normalizer.pkl")
    print(f"Final PCS model saved to {model_save_path_pcs}_final.zip")
    print(f"Final ISO model saved to {model_save_path_iso}_final.zip")

    # --- Helper to load environment with normalization for evaluation ---
    def load_env_and_normalizer(env_id, normalizer_path, log_dir, min_action, max_action, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}
        env = gym.make(env_id, **extra_kwargs)
        env = RescaleAction(env, min_action=min_action, max_action=max_action)
        env = Monitor(env, filename=os.path.join(log_dir, 'eval_monitor.csv'))
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(normalizer_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        return vec_env

    # --- Plot evaluation rewards ---
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

    plot_eval_rewards(log_dir_pcs, "PCS")
    plot_eval_rewards(log_dir_iso, "ISO")
    print("Training and evaluation process completed.")

    # --- Final Evaluation ---
    # For PCS evaluation, pass the frozen ISO agent path into the environment
    pcs_eval_env = load_env_and_normalizer(
        env_id_pcs,
        f"{model_save_path_pcs}_normalizer.pkl",
        log_dir_pcs,
        min_action=-10,
        max_action=10,
        extra_kwargs={'trained_iso_model_path': f"{model_save_path_iso}_final.zip"}
    )
    pcs_model_final = PPO.load(f"{model_save_path_pcs}_final.zip", env=pcs_eval_env)
    mean_reward_pcs_final, std_reward_pcs_final = evaluate_policy(pcs_model_final, pcs_eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Final PCS Model - Mean Reward: {mean_reward_pcs_final} +/- {std_reward_pcs_final}")

    # For ISO evaluation, pass the frozen PCS agent path into the environment
    iso_eval_env = load_env_and_normalizer(
        env_id_iso,
        f"{model_save_path_iso}_normalizer.pkl",
        log_dir_iso,
        min_action=1,
        max_action=10,
        extra_kwargs={'trained_pcs_model_path': f"{model_save_path_pcs}_final.zip"}
    )
    iso_model_final = PPO.load(f"{model_save_path_iso}_final.zip", env=iso_eval_env)
    mean_reward_iso_final, std_reward_iso_final = evaluate_policy(iso_model_final, iso_eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Final ISO Model - Mean Reward: {mean_reward_iso_final} +/- {std_reward_iso_final}")


if __name__ == "__main__":
    train_and_evaluate_agent(algo_type='PPO')
