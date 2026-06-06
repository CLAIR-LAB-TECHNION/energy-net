# agent_config.py
"""
Agent configuration dataclass for multi-agent training system.

This module defines the AgentConfig dataclass used to configure individual agents
in the multi-agent reinforcement learning training framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Type
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass
class AgentConfig:
    """
    Configuration for a single agent in the multi-agent training system.
    
    Attributes:
        agent_id: Unique identifier for the agent (e.g., "main_iso", "household_1")
        agent_type: Type of agent - "ISO" or "PCS"
        env: Gymnasium environment instance for this agent
        algo_class: RL algorithm class (e.g., PPO, A2C, SAC)
        policy: Policy type (e.g., "MlpPolicy", "CnnPolicy")
        timesteps_per_turn: Number of timesteps this agent trains per cycle
        algo_kwargs: Additional keyword arguments for the algorithm
        model: The trained RL model (initialized during setup)
    """
    agent_id: str
    agent_type: str  # "ISO" or "PCS"
    env: gym.Env
    algo_class: Type[BaseAlgorithm]
    policy: str = "MlpPolicy"
    timesteps_per_turn: int = 48
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)
    model: Optional[BaseAlgorithm] = None
    
    def __post_init__(self):
        """Validate agent configuration after initialization."""
        if self.agent_type not in ["ISO", "PCS"]:
            raise ValueError(f"agent_type must be 'ISO' or 'PCS', got '{self.agent_type}'")
        
        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")
        
        if self.timesteps_per_turn < 1:
            raise ValueError(f"timesteps_per_turn must be >= 1, got {self.timesteps_per_turn}")
        
        if not isinstance(self.algo_kwargs, dict):
            raise ValueError("algo_kwargs must be a dictionary")
    
    def initialize_model(self, verbose: int = 0) -> BaseAlgorithm:
        """
        Initialize the RL model for this agent.
        
        Args:
            verbose: Verbosity level for the algorithm
            
        Returns:
            Initialized RL model
        """
        if self.model is not None:
            return self.model
        
        self.model = self.algo_class(
            self.policy,
            self.env,
            verbose=verbose,
            **self.algo_kwargs
        )
        return self.model
    
    def train(self, reset_num_timesteps: bool = False) -> Dict[str, Any]:
        """
        Train this agent for the configured number of timesteps.
        
        Args:
            reset_num_timesteps: Whether to reset the timestep counter
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            raise RuntimeError(f"Agent {self.agent_id} model not initialized. Call initialize_model() first.")
        
        self.model.learn(
            total_timesteps=self.timesteps_per_turn,
            reset_num_timesteps=reset_num_timesteps
        )
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "timesteps_trained": self.timesteps_per_turn
        }
    
    def predict(self, observation, deterministic: bool = True):
        """
        Get action prediction from this agent's model.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, state)
        """
        if self.model is None:
            raise RuntimeError(f"Agent {self.agent_id} model not initialized.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def __repr__(self):
        """String representation of the agent configuration."""
        model_status = "initialized" if self.model is not None else "not initialized"
        return (f"AgentConfig(id='{self.agent_id}', type={self.agent_type}, "
                f"algo={self.algo_class.__name__}, timesteps={self.timesteps_per_turn}, "
                f"model={model_status})")