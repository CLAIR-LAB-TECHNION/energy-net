"""
PCS Unit Environment

This environment simulates a Power Consumption & Storage (PCS) unit interacting with the power grid.

Environment States:
    - Battery energy level (MWh)
    - Current time (fraction of day)

Actions:
    - Battery charging/discharging rate
    - Optional: Production control
    - Optional: Consumption control

Key Features:
    - Integrates with trained ISO models for price determination
    - Supports both single and multi-action control schemes
    - Implements configurable reward functions
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from energy_net.pcsunit_controller import PCSUnitController
from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType

class PCSUnitEnv(gym.Env):
    """
    Gymnasium environment for PCS unit training.
    
    The environment simulates a PCS unit that can:
    1. Store energy in a battery
    2. Generate energy through self-production
    3. Consume energy based on demand
    4. Buy/sell energy from/to the grid
    
    The agent learns to optimize these operations based on:
    - Current energy prices (determined by ISO)
    - Internal state (battery level, production, consumption)
    - Time of day
    """
    def __init__(
        self,
        cost_type=None,               
        demand_pattern=None,        
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',
        reward_type: str = 'cost',
        trained_iso_model_path: Optional[str] = None, 
        model_iteration: Optional[int] = None  
    ):
        """
        Initializes the PCSUnitEnv environment.

        Args:
            render_mode (Optional[str], optional): Rendering mode. Defaults to None.
            env_config_path (Optional[str], optional): Path to environment config. Defaults to 'configs/environment_config.yaml'.
            iso_config_path (Optional[str], optional): Path to ISO config. Defaults to 'configs/iso_config.yaml'.
            pcs_unit_config_path (Optional[str], optional): Path to PCS unit config. Defaults to 'configs/pcs_unit_config.yaml'.
            log_file (Optional[str], optional): Path to log file. Defaults to 'logs/environments.log'.
            reward_type (str, optional): Type of reward function. Defaults to 'cost'.
            trained_iso_model_path (Optional[str], optional): Path to trained ISO model. Defaults to None.
            model_iteration (Optional[int], optional): Model iteration number. Defaults to None.
        """
        super().__init__()
        
        # Convert demand_pattern to enum if it's a string
        if isinstance(demand_pattern, str):
            demand_pattern = DemandPattern[demand_pattern.upper()]
        elif demand_pattern is None:
            demand_pattern = DemandPattern.SINUSOIDAL  # Default pattern
            
        # Convert cost_type to enum if it's a string
        if isinstance(cost_type, str):
            cost_type = CostType[cost_type.upper()]
        elif cost_type is None:
            cost_type = CostType.CONSTANT  # Default cost type
        
        # Initialize controller with base configurations and new parameters
        self.controller = PCSUnitController(
            cost_type=cost_type,              
            demand_pattern=demand_pattern,      
            render_mode=render_mode,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file,
            reward_type=reward_type,
            trained_iso_model_path=trained_iso_model_path 
        )

        # Store new parameters
        self.cost_type = cost_type
        self.demand_pattern = demand_pattern

        # Use controller's logger
        self.logger = self.controller.logger

        # Load trained ISO model if provided
        if trained_iso_model_path:
            try:
                trained_iso_agent = PPO.load(trained_iso_model_path)
                self.controller.set_trained_iso_agent(trained_iso_agent)
                self.logger.info(f"Loaded ISO model: {trained_iso_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load ISO model: {e}")

        self.model_iteration = model_iteration
        self.observation_space = self.controller.observation_space
        self.action_space = self.controller.action_space

    def update_trained_iso_model(self, model_path: str) -> bool:
        """Update the trained ISO model during training iterations"""
        try:
            trained_iso_agent = PPO.load(model_path)
            self.controller.set_trained_iso_agent(trained_iso_agent)
            self.logger.info(f"Updated ISO model: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update ISO model: {e}")
            return False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
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

    def step(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes a single time step within the environment.

        Args:
        action (float or np.ndarray): Charging (+) or discharging (-) power.
            - If float: Represents the charging (+) or discharging (-) power directly.
            - If np.ndarray with shape (1,): The scalar value is extracted for processing.

        Returns:
            Tuple containing:
                - Next observation
                - Reward
                - Terminated flag
                - Truncated flag
                - Info dictionary
        """
        return self.controller.step(action)

    def get_info(self) -> Dict[str, float]:
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







