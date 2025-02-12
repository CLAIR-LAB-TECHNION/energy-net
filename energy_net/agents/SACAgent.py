import os
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from energy_net.agents.agent import Agent


class SACAgent(Agent):
    """
    Soft Actor-Critic (SAC) agents using Stable Baselines.
    """

    def __init__(self, env, policy, log_dir='./logs/', verbose=1):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(env, log_dir)
        self.unwrapped = env
        self.policy = policy
        self.verbose = verbose
        self.model = None
        self.eval_callback = None
        self.eval_rewards = []
        self.train_rewards = []
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def train(self, total_timesteps=10000, log_interval=10, eval_freq=1000, progress_bar=True, **kwargs):
        self.eval_callback = EvalCallback(self.env, log_path=self.log_dir, eval_freq=eval_freq,
                                          best_model_save_path=self.log_dir)

        self.model = SAC(self.policy, self.env, verbose=self.verbose, **kwargs)
        self.env = self.model.env
        self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar, log_interval=log_interval,
                         callback=self.eval_callback)

    def eval(self, n_episodes=5):
        rewards, _ = evaluate_policy(self.model, self.env, n_eval_episodes=n_episodes, deterministic=True, render=False)
        return np.mean(rewards)

    def get_action(self, observation, deterministic=False):
        """
        Choose an action based on the given observation.

        Args:
            observation (np.ndarray): The observation from the environment.
            deterministic (bool): Whether to choose the action deterministically or stochastically.

        Returns:
            np.ndarray: The chosen action.
        """

        self.model.env.action_space = self.unwrapped.action_space
        self.model.policy.action_space = self.unwrapped.action_space
        return self.model.predict(observation, deterministic=deterministic)[0]

    def _log_rewards(self, locals_, globals_):
        self.train_rewards.append(locals_['episode_rewards'][-1])
        self.eval_rewards.append(locals_['eval_rewards'][-1])


