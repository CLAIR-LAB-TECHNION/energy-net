"""
Independent System Operator (ISO) Environment

This environment simulates an ISO managing electricity prices in the power grid.

Environment States:
    - Current time (fraction of day)
    - Nominal grid demand (MWh)
    - PCS unit net exchange (MWh)

Actions:
    - Buy price ($/MWh)
    - Sell price ($/MWh)

Key Features:
    - Integrates with trained PCS models for demand response simulation
    - Implements grid stability metrics
    - Supports various pricing strategies
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from energy_net.iso_controller import ISOController 

class ISOEnv(gym.Env):
    """
    Gymnasium environment for ISO training.
    
    The ISO environment simulates a grid operator that:
    1. Observes current grid conditions
    2. Sets buy/sell prices for energy
    3. Monitors grid stability and demand response
    4. Optimizes for both efficiency and stability
    
    The agent learns to set optimal prices based on:
    - Current grid demand
    - PCS units' behavior
    - Time of day
    - Grid stability metrics
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',
        reward_type: str = 'iso',
        trained_pcs_model_path: Optional[str] = None,  
        model_iteration: Optional[int] = None  
    ):
        super().__init__()
        
        # Initialize controller with base configurations
        self.controller = ISOController(
            render_mode=render_mode,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file,
            reward_type=reward_type
        )

        # Use controller's logger
        self.logger = self.controller.logger

        # Load trained PCS model if provided
        if trained_pcs_model_path:
            try:
                trained_pcs_agent = PPO.load(trained_pcs_model_path)
                self.controller.set_trained_pcs_agent(trained_pcs_agent)
                self.logger.info(f"Loaded PCS model: {trained_pcs_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load PCS model: {e}")

        self.model_iteration = model_iteration
        self.observation_space = self.controller.observation_space
        self.action_space = self.controller.action_space

    def update_trained_pcs_model(self, model_path: str) -> bool:
        """Update the trained PCS model during training iterations"""
        try:
            trained_pcs_agent = PPO.load(model_path)
            self.controller.set_trained_pcs_agent(trained_pcs_agent)
            self.logger.info(f"Updated PCS model: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update PCS model: {e}")
            return False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state.

        Args:
            seed: Optional seed for random number generator.
            options: Optional settings like reward type.

        Returns:
            Tuple containing the initial observation and info dictionary.
        """
        super().reset(seed=seed)  # Reset the parent class's state
        return self.controller.reset(seed=seed, options=options)


    def step(self, action: Union[np.ndarray, float]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step for the ISO environment.

        Action is [buy_price, sell_price].
        The environment updates internal states, calculates reward, sets done flags, etc.

        Args:
            action: A 2D vector [buy_price, sell_price].

        Returns:
            observation (np.ndarray): Updated observation.
            reward (float): Reward from this step.
            done (bool): If the episode is terminated.
            truncated (bool): If the episode is truncated (time limit).
            info (dict): Additional info.
        """
        return self.controller.step(action)

    def get_info(self) -> Dict[str, Any]:
        """
        Provides additional information about the environment's state.

        Returns:
            Dict[str, float]: Dictionary containing the running average price.
        """
        return self.controller.get_info()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        return self.controller.load_config(config_path)

    def render(self, mode: Optional[str] = None):
        """
        Rendering method. Not implemented.

        Args:
            mode: Optional rendering mode.
        """
        self.controller.logger.warning("Render method is not implemented.")
        raise NotImplementedError("Rendering is not implemented.")

    def close(self):
        """
        Cleanup method. Closes loggers and releases resources.
        """
        self.controller.close()
