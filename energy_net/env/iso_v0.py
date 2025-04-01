"""
Independent System Operator (ISO) Environment

This environment implements a Gymnasium-compliant interface for training reinforcement
learning agents to act as grid operators (ISOs). The ISO is responsible for setting
electricity prices and managing dispatch to balance supply and demand in the grid.

Key features:
1. Configurable demand patterns, cost structures, and pricing policies
2. Integration with PCS (Power Consumption & Storage) units for demand response
3. Support for both price-only and combined price-dispatch control
4. Realistic grid modeling with demand uncertainty and reserve costs
5. Comprehensive info dictionaries for monitoring and visualization

This environment serves as the main interface between RL algorithms and the
underlying grid simulation, enabling the training of agents that can efficiently
manage electricity markets.

Observation Space:
    [time, predicted_demand, pcs_demand]

Action Space:
    Depends on pricing policy:
    - ONLINE: [buy_price, sell_price, (optional) dispatch]
    - QUADRATIC: [b0, b1, b2, s0, s1, s2, (optional) dispatch_profile]
    - CONSTANT: [buy_price, sell_price, (optional) dispatch_profile]
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Union
import os  
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from energy_net.iso_controller import ISOController
from energy_net.env import PricingPolicy 
from energy_net.market.iso.demand_patterns import DemandPattern

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
    
    The environment follows the standard Gymnasium interface, making it compatible
    with common reinforcement learning libraries.
    """
    
    def __init__(
        self,
        cost_type=None,
        pricing_policy=None,
        num_pcs_agents=None,
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',
        reward_type: str = 'iso',
        trained_pcs_model_path: Optional[str] = None,  
        model_iteration: Optional[int] = None,
        demand_pattern=None,
        dispatch_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the ISOEnv environment.
        
        This environment provides a Gymnasium-compliant interface wrapping around the
        more complex ISOController which handles the actual grid simulation logic.
        
        Args:
            pricing_policy: Which pricing mechanism to use (ONLINE, QUADRATIC, CONSTANT)
            cost_type: How grid operation costs are calculated
            num_pcs_agents: Number of PCS units to simulate
            render_mode: Visual rendering mode (currently not implemented)
            env_config_path: Path to environment configuration file
            iso_config_path: Path to ISO-specific configuration file
            pcs_unit_config_path: Path to PCS unit configuration file
            log_file: Path to log file for storing environment logs
            reward_type: Which reward function to use (default: 'iso')
            trained_pcs_model_path: Optional path to a pre-trained PCS agent model
            model_iteration: Optional model iteration number for tracking
            demand_pattern: Pattern of demand variation over time
            dispatch_config: Optional configuration for dispatch control
        """
        super().__init__()
        self.pricing_policy = pricing_policy
        self.cost_type = cost_type
        self.num_pcs_agents = num_pcs_agents
        self.demand_pattern = demand_pattern
        
        self.controller = ISOController(
            cost_type=cost_type,
            num_pcs_agents=num_pcs_agents,
            pricing_policy=pricing_policy,
            demand_pattern=demand_pattern,  
            render_mode=render_mode,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file,
            reward_type=reward_type,
            dispatch_config=dispatch_config  # Pass dispatch_config to ISOController
        )

        # Use controller's logger
        self.logger = self.controller.logger

        # Load trained PCS model if provided
        if trained_pcs_model_path:
            try:
                print(f"Attempting to load PCS model from: {trained_pcs_model_path}")
                print(f"Number of PCS agents: {num_pcs_agents}")
                
                if not os.path.exists(trained_pcs_model_path):
                    raise FileNotFoundError(f"Model file not found: {trained_pcs_model_path}")
                    
                # Try loading the model first to verify it's valid
                test_model = PPO.load(trained_pcs_model_path)
                print("Successfully loaded model, now setting for each agent")
                
                for i in range(num_pcs_agents):
                    success = self.controller.set_trained_pcs_agent(i, trained_pcs_model_path)
                    print(f"Agent {i} loading status: {'Success' if success else 'Failed'}")
                    
                self.logger.info(f"Loaded PCS model: {trained_pcs_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load PCS model: {e}")
                print(f"Error loading model: {str(e)}")
                raise  # Re-raise the exception to see the full traceback

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
        
        Resets all internal state variables to their initial values and returns
        the initial observation. This is called at the beginning of each episode
        during training or evaluation.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for environment reset (e.g., reward type)
            
        Returns:
            Tuple containing:
            - Initial observation: [time, predicted_demand, pcs_demand]
            - Info dictionary with initial state information
        """
        super().reset(seed=seed)  # Reset the parent class's state
        return self.controller.reset(seed=seed, options=options)


    def step(self, action: Union[np.ndarray, float]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step in the ISO environment.
        
        The ISO agent provides an action (pricing decisions and optionally dispatch),
        and the environment:
        1. Updates the grid state
        2. Simulates PCS unit responses
        3. Calculates costs and rewards
        4. Returns the next observation
        
        Args:
            action: The action from the agent, format depends on pricing policy
            
        Returns:
            Tuple containing:
            - observation: Next state [time, predicted_demand, pcs_demand]
            - reward: Reward for this step
            - terminated: Whether episode is done
            - truncated: Whether episode was truncated (not used)
            - info: Additional information dictionary
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
