"""
PCS Metrics Handler Module

This module handles all calculations related to rewards, costs, and metrics
for the PCS controller. It encapsulates the complex logic of determining costs,
revenues, and battery utilization metrics for the reinforcement learning agent.

Key functions:
1. Cost and revenue calculations based on market prices and grid exchange
2. Reward computation based on configurable parameters
3. Metrics tracking across episodes and steps
4. Building info dictionaries for environment monitoring

By isolating these calculations in a separate module, we improve modularity and testability
of the PCS controller while keeping the core logic clean.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
from energy_net.rewards.base_reward import BaseReward

class PCSMetricsHandler:
    """
    Handles metrics calculations for the PCS unit controller.
    
    This class is responsible for:
    1. Tracking performance metrics over time
    2. Calculating rewards based on configured reward function
    3. Building info dictionaries for environment step returns
    4. Providing metrics visualization data
    5. Maintaining episode statistics and summaries
    
    By extracting this logic from the PCS controller, we make the controller cleaner and more focused
    on its core responsibilities, while making the metrics calculations more maintainable and testable.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        reward_function: BaseReward,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the metrics handler.
        
        Args:
            config: Configuration parameters for metrics and rewards
                May include settings for reward scaling, cost factors, etc.
            reward_function: The reward calculation function/object that implements
                             the BaseReward interface
            logger: Optional logger for tracking metrics and debugging information
        """
        self.logger = logger
        self.config = config
        self.reward_function = reward_function
        
        # Dictionary to track metrics over time
        self.metrics_history = {
            'rewards': [],
            'battery_levels': [],
            'energy_changes': [],
            'market_exchanges': [],
            'revenues': [],
            'productions': [],
            'consumptions': [],
            'buy_prices': [],
            'sell_prices': [],
            'times': [],
            'predicted_demands': [],
            'realized_demands': [],
            'dispatch_costs': [],
            'reserve_costs': [],
            'shortfalls': []
        }
        
        # Running statistics
        self.total_reward = 0.0
        self.episode_count = 0
        self.step_count = 0
        
        if self.logger:
            self.logger.info("PCS Metrics Handler initialized")
    
    def calculate_reward(self, info: Dict[str, Any]) -> float:
        """
        Calculate reward based on current state.
        
        Computes the reward for the current step using the configured reward function
        and the provided state information. This is typically called during the
        environment's step method.
        
        Args:
            info: State/info dictionary with all required fields for reward calculation
                 such as battery level, market prices, production, consumption, etc.
            
        Returns:
            float: Calculated reward value
        """
        # Use the reward function to calculate reward
        reward = self.reward_function.compute_reward(info)
        
        # We don't update total_reward here as it's now handled in build_info_dict
        # after potential corrections to ensure consistency
        self.step_count += 1
        
        if self.logger:
            self.logger.debug(f"Step {self.step_count}, raw reward: {reward:.4f}")
            
        return reward
    
    def build_info_dict(self, state: Dict[str, Any], reward: float) -> Dict[str, Any]:
        """
        Build info dictionary for step returns, enhancing with metrics tracking.
        
        This method takes the current state and reward, tracks metrics over time,
        and returns an enhanced info dictionary for the environment's step method.
        It also updates internal metrics history for later analysis.
        
        Args:
            state: Current state dictionary containing environment variables
            reward: Calculated reward for the current step
            
        Returns:
            Dict[str, Any]: Enhanced info dictionary with metrics, reward tracking,
                           and cumulative statistics
        """
        # Start with the existing state info
        info = state.copy()
        
        # Extract key metrics for logging
        production = state.get('production', 0.0)
        energy_change = state.get('energy_change', 0.0)
        battery_action = state.get('battery_action', 0.0)
        net_exchange = state.get('net_exchange', 0.0)
        battery_level = state.get('battery_level', 0.0)
        
        # Basic logging for debugging the relationship between actions and market
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            action_type = "charging" if battery_action > 0 else "discharging" if battery_action < 0 else "no action"
            exchange_type = "buying" if net_exchange > 0 else "selling" if net_exchange < 0 else "balanced"
            self.logger.debug(
                f"Step {self.step_count}: {action_type} ({battery_action:.4f}), "
                f"{exchange_type} ({net_exchange:.4f}), "
                f"battery: {battery_level:.2f}, reward: {reward:.4f}"
            )
        
        # Add reward tracking info
        info.update({
            'reward': reward,
            'step': self.step_count,
            'total_reward': self.total_reward + reward
        })
        
        # Track metrics
        self.metrics_history['rewards'].append(reward)
        self.metrics_history['battery_levels'].append(battery_level)
        self.metrics_history['energy_changes'].append(energy_change)
        self.metrics_history['market_exchanges'].append(net_exchange)
        self.metrics_history['revenues'].append(state.get('revenue', 0.0))
        self.metrics_history['productions'].append(production)
        self.metrics_history['consumptions'].append(state.get('consumption', 0.0))
        self.metrics_history['buy_prices'].append(state.get('iso_buy_price', 0.0))
        self.metrics_history['sell_prices'].append(state.get('iso_sell_price', 0.0))
        self.metrics_history['times'].append(state.get('current_time', state.get('time', 0.0)))
        self.metrics_history['predicted_demands'].append(state.get('predicted_demand', 0.0))
        self.metrics_history['realized_demands'].append(state.get('realized_demand', 0.0))
        self.metrics_history['dispatch_costs'].append(state.get('dispatch_cost', 0.0))
        self.metrics_history['reserve_costs'].append(state.get('reserve_cost', 0.0))
        self.metrics_history['shortfalls'].append(state.get('shortfall', 0.0))
        
        # Update total reward
        self.total_reward += reward
        
        return info
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of tracked metrics.
        
        Calculates and returns summary statistics for the current episode,
        including mean, standard deviation, min, and max values for reward
        and other key metrics.
        
        Returns:
            Dictionary containing:
            - mean_reward: Average reward per step
            - std_reward: Standard deviation of rewards
            - min_reward: Minimum reward received
            - max_reward: Maximum reward received
            - total_reward: Cumulative reward for the episode
            - steps: Total steps taken
        """
        if not self.metrics_history['rewards']:
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'total_reward': 0.0,
                'steps': 0
            }
            
        rewards = np.array(self.metrics_history['rewards'])
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'total_reward': float(np.sum(rewards)),
            'steps': self.step_count
        }
    
    def get_full_metrics(self) -> Dict[str, List[float]]:
        """
        Get complete metrics history.
        
        Returns the full history of all tracked metrics, which can be used
        for detailed analysis and visualization of agent performance.
        
        Returns:
            Dictionary with all metrics history as lists of values
        """
        return self.metrics_history
    
    def end_episode(self) -> Dict[str, Any]:
        """
        Mark the end of an episode and return summary statistics.
        
        This method should be called at the end of each episode to:
        1. Update episode count
        2. Log episode summary statistics
        3. Prepare summary metrics for external use
        
        Returns:
            Dictionary with episode summary statistics
        """
        self.episode_count += 1
        
        summary = self.get_metrics_summary()
        
        
        if self.logger:
            self.logger.info(f"Episode {self.episode_count} completed:")
            self.logger.info(f"  Total reward: {summary['total_reward']:.4f}")
            self.logger.info(f"  Mean reward: {summary['mean_reward']:.4f}")
            self.logger.info(f"  Steps: {summary['steps']}")
            
        return summary
    
    def reset(self) -> None:
        """
        Reset metrics tracking for a new episode.
        
        Clears all tracked metrics and resets counters for a new episode,
        while preserving the episode count for continuity.
        """
        # Preserve episode count but reset everything else
        self.total_reward = 0.0
        self.step_count = 0
        
        for key in self.metrics_history:
            self.metrics_history[key] = []
            
        if self.logger:
            self.logger.info(f"Metrics handler reset for episode {self.episode_count + 1}")