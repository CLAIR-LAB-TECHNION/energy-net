"""
Energy Net V0 Environment

A unified multi-agent environment that integrates both the ISO and PCS agents
into a single simulation. This environment follows the multi-agent extension
of the Gym interface, where step() takes multiple actions and returns multiple
observations, rewards, and done flags.

Key features:
1. Integrated controller for both ISO and PCS agents
2. Sequential processing of agent actions
3. Single timeline and shared state management
4. Direct access to comprehensive metrics

This environment serves as the main interface between RL algorithms and the
underlying energy net simulation, enabling the training of agents that can
efficiently manage electricity markets and battery storage.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Union, List, Optional

from energy_net.controllers.energy_net_controller import EnergyNetController
from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType
from energy_net.market.pricing_policy import PricingPolicy


class EnergyNetV0(gym.Env):
    """
    Multi-agent environment for simulating energy grid dynamics.
    
    This environment integrates both ISO and PCS agents into a single simulation,
    following a multi-agent extension of the Gym interface where step() takes multiple
    actions and returns observations, rewards, and done flags for all agents.
    
    The environment uses a unified EnergyNetController to manage the sequential
    simulation, where:
    1. ISO agent sets energy prices
    2. PCS agent responds with battery control actions
    3. Energy exchanges occur
    4. State updates and rewards are calculated
    
    This approach eliminates the need for manual transfers between separate
    environments and provides a more realistic simulation with direct access
    to comprehensive metrics.
    """
    
    def __init__(
        self,
        cost_type: Union[str, CostType] = None,
        pricing_policy: Union[str, PricingPolicy] = None,
        demand_pattern: Union[str, DemandPattern] = None,
        num_pcs_agents: int = 1,
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',
        iso_reward_type: str = 'iso',
        pcs_reward_type: str = 'cost',
        dispatch_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the unified Energy Net environment.
        
        Args:
            cost_type: How grid operation costs are calculated (CONSTANT, VARIABLE, TIME_OF_USE)
            pricing_policy: Policy for determining energy prices (ONLINE, QUADRATIC, CONSTANT)
            demand_pattern: Pattern of demand variation over time (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
            num_pcs_agents: Number of PCS units (currently only supports 1)
            render_mode: Visual rendering mode (not currently implemented)
            env_config_path: Path to environment configuration file
            iso_config_path: Path to ISO-specific configuration file
            pcs_unit_config_path: Path to PCS unit configuration file
            log_file: Path for logging controller events
            iso_reward_type: Type of reward function for ISO agent
            pcs_reward_type: Type of reward function for PCS agent
            dispatch_config: Configuration for dispatch control
        """
        super().__init__()
        
        # Convert enum strings to actual enums if needed
        if isinstance(cost_type, str):
            cost_type = CostType[cost_type.upper()]
        elif cost_type is None:
            cost_type = CostType.CONSTANT  # Default cost type
            
        if isinstance(pricing_policy, str):
            pricing_policy = PricingPolicy[pricing_policy.upper()]
        elif pricing_policy is None:
            pricing_policy = PricingPolicy.ONLINE  # Default pricing policy
            
        if isinstance(demand_pattern, str):
            demand_pattern = DemandPattern[demand_pattern.upper()]
        elif demand_pattern is None:
            demand_pattern = DemandPattern.SINUSOIDAL  # Default demand pattern
        
        # Initialize the unified controller
        self.controller = EnergyNetController(
            cost_type=cost_type,
            pricing_policy=pricing_policy,
            demand_pattern=demand_pattern,
            num_pcs_agents=num_pcs_agents,
            render_mode=render_mode,
            env_config_path=env_config_path,
            iso_config_path=iso_config_path,
            pcs_unit_config_path=pcs_unit_config_path,
            log_file=log_file,
            iso_reward_type=iso_reward_type,
            pcs_reward_type=pcs_reward_type,
            dispatch_config=dispatch_config,
        )
        
        # Define agent spaces
        # Note: In this implementation, we're using dict spaces for compatibility
        # with multi-agent RL frameworks, but the controller internally uses
        # separate spaces for each agent
        self.agents = ["iso", "pcs"]
        
        self.observation_space = {
            "iso": self.controller.get_iso_observation_space(),
            "pcs": self.controller.get_pcs_observation_space()
        }
        
        self.action_space = {
            "iso": self.controller.get_iso_action_space(),
            "pcs": self.controller.get_pcs_action_space()
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple containing:
            - Initial observations for both agents in a dict
            - Info dictionary with initial state information
        """
        observations, info = self.controller.reset(seed=seed, options=options)
        
        # Format observations as a dict for multi-agent compatibility
        obs_dict = {
            "iso": observations[0],
            "pcs": observations[1]
        }
        
        return obs_dict, info

    def step(self, action_dict):
        """
        Execute one time step of the environment.
        
        Args:
            action_dict: Dict containing actions for each agent
                {"iso": iso_action, "pcs": pcs_action}
            
        Returns:
            Tuple containing:
            - Dict of observations for each agent
            - Dict of rewards for each agent
            - Dict of terminated flags for each agent
            - Dict of truncated flags for each agent
            - Dict of info for each agent
        """
        # Extract actions from dict
        iso_action = action_dict["iso"]
        pcs_action = action_dict["pcs"]
        
        # Execute step on the controller
        # New return format: observations, rewards, terminated, truncated, info
        observations, rewards, terminated, truncated, info = self.controller.step(iso_action, pcs_action)
        
        # Format returns as dicts for multi-agent compatibility
        obs_dict = {
            "iso": observations[0],
            "pcs": observations[1]
        }
        
        reward_dict = {
            "iso": rewards[0],
            "pcs": rewards[1]
        }
        
        terminated_dict = {
            "iso": terminated[0] if isinstance(terminated, (list, tuple)) else terminated,
            "pcs": terminated[1] if isinstance(terminated, (list, tuple)) else terminated
        }
        
        # Handle truncated flag
        truncated_dict = {
            "iso": truncated[0] if isinstance(truncated, (list, tuple)) else truncated,
            "pcs": truncated[1] if isinstance(truncated, (list, tuple)) else truncated
        }
        
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info

    def get_metrics(self):
        """
        Get comprehensive metrics for both agents.
        
        Returns:
            Dict containing metrics for both agents and shared metrics
        """
        return self.controller.get_metrics()
    
    def render(self):
        """
        Render the environment (not implemented).
        
        Raises:
            NotImplementedError: Always, as rendering is not implemented.
        """
        raise NotImplementedError("Rendering is not yet implemented for the EnergyNetV0 environment")

    def close(self):
        """
        Clean up any resources used by the environment.
        """
        # Currently, no cleanup is needed
        pass


def make_env(config=None):
    """
    Factory function to create an instance of EnergyNetV0.
    
    Args:
        config: Configuration dictionary for the environment
        
    Returns:
        EnergyNetV0: An instance of the environment
    """
    return EnergyNetV0(**(config or {}))
