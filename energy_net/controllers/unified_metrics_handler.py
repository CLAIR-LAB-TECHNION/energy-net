"""
Unified Metrics Handler Module

This module provides a combined metrics handler for both ISO and PCS components
in the unified EnergyNet environment. It handles all calculations related to:
- Grid state metrics (demand, shortfall, dispatch)
- Battery state metrics (level, energy exchange)
- Cost and revenue calculations
- Reward computations for both agents
- Comprehensive metrics tracking and reporting

By unifying the metrics handling in a single module, we ensure consistency
between the metrics reported for both agents and simplify the data flow in
the unified simulation.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt

# Import reward classes
from energy_net.rewards.base_reward import BaseReward


class UnifiedMetricsHandler:
    """
    Unified handler for metrics calculation and tracking in the EnergyNet environment.
    
    This class combines functionality from both the ISO and PCS metrics handlers to
    provide a single point of truth for all metrics in the unified simulation. It:
    
    1. Tracks all relevant state variables for both ISO and PCS
    2. Calculates costs and revenues for both agents
    3. Provides reward calculations based on configurable parameters
    4. Maintains historical metrics for analysis and visualization
    5. Generates comprehensive info dictionaries for monitoring
    
    The unified approach ensures consistency in metric calculations and simplifies
    the data flow in the simulation, while still providing separate metrics and
    rewards for each agent.
    """
    
    def __init__(
        self,
        env_config: Dict[str, Any],
        iso_config: Dict[str, Any],
        pcs_config: Dict[str, Any],
        cost_type: Any,
        reward_function: Optional[BaseReward] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the unified metrics handler.
        
        Args:
            env_config: Environment configuration parameters
            iso_config: ISO-specific configuration parameters
            pcs_config: PCS-specific configuration parameters
            cost_type: The type of cost model being used
            logger: Optional logger for tracking metrics and debugging
        """
        self.logger = logger
        self.env_config = env_config
        self.iso_config = iso_config
        self.pcs_config = pcs_config
        self.cost_type = cost_type
        self.reward_function = reward_function  # Store reference to external reward function
        
        # Extract cost parameters
        dispatch_price = env_config.get('dispatch_price', 0.0)
        reserve_price = env_config.get('reserve_price', 0.0)
        
        if isinstance(dispatch_price, dict):
            self.dispatch_price = dispatch_price.get(cost_type.value, 5.0)
        else:
            self.dispatch_price = dispatch_price
            
        if isinstance(reserve_price, dict):
            self.reserve_price = reserve_price.get(cost_type.value, 10.0)
        else:
            self.reserve_price = reserve_price
            
        # Demand uncertainty parameter
        self.sigma = env_config.get('demand_uncertainty', {}).get('sigma', 0.0)
        
        # Initialize metrics tracking
        self.reset()
        
        if self.logger:
            self.logger.info(f"Initialized UnifiedMetricsHandler with dispatch_price: ${self.dispatch_price:.2f}, reserve_price: ${self.reserve_price:.2f}")
    
    def reset(self) -> None:
        """
        Reset metrics tracking for a new episode.
        
        Clears all tracked metrics and resets counters for a new episode.
        """
        # ISO metrics
        self.iso_metrics = {
            'predicted_demands': [],
            'realized_demands': [],
            'pcs_demands': [],
            'net_demands': [],
            'shortfalls': [],
            'dispatch_costs': [],
            'reserve_costs': [],
            'total_costs': [],
            'buy_prices': [],
            'sell_prices': [],
            'energy_bought': [],
            'energy_sold': [],
            'revenues': [],
            'rewards': [],
            'grid_stability': [],
            'dispatch_levels': [],  # Track dispatch levels
            'reserve_levels': [],   # Track reserve levels
            'price_spreads': [],     # Track price spreads
            'actions': []           # Track ISO actions
        }
        
        # PCS metrics
        self.pcs_metrics = {
            'battery_levels': [],
            'energy_exchanges': [],
            'costs': [],
            'revenues': [],
            'actions': [],
            'rewards': [],
            'battery_utilization': [],
            'charge_rates': [],     # Track charging rates
            'discharge_rates': [],  # Track discharging rates
            'efficiency_losses': [] # Track efficiency losses
        }
        
        # Shared metrics
        self.shared_metrics = {
            'times': [],
            'steps': [],
            'episode_rewards': [],  # Track episode rewards
            'episode_costs': [],    # Track episode costs
            'episode_utilization': [] # Track episode utilization
        }
        
        # Episode stats
        self.total_iso_reward = 0.0
        self.total_pcs_reward = 0.0
        self.episode_count = 0
        self.step_count = 0
        
        # State variables
        self.energy_bought = 0.0
        self.energy_sold = 0.0
        self.iso_buy_price = 0.0
        self.iso_sell_price = 0.0
        self.predicted_demand = 0.0
        self.realized_demand = 0.0
        self.pcs_demand = 0.0
        self.battery_level = 0.0
        self.last_iso_action = None
        self.last_pcs_action = None
        
        if self.logger:
            self.logger.info("Metrics handler reset for new episode")
    
    def update_prices(self, buy_price: float, sell_price: float) -> None:
        """
        Update ISO prices in the metrics tracker.
        
        Args:
            buy_price: The price at which ISO buys energy from PCS
            sell_price: The price at which ISO sells energy to PCS
        """
        self.iso_buy_price = buy_price
        self.iso_sell_price = sell_price
        self.iso_metrics['buy_prices'].append(buy_price)
        self.iso_metrics['sell_prices'].append(sell_price)
        
        # Track price spread for evaluation
        price_spread = sell_price - buy_price
        self.iso_metrics['price_spreads'].append(price_spread)
        
        if self.logger:
            self.logger.debug(f"Updated prices: buy={buy_price:.2f}, sell={sell_price:.2f}, spread={price_spread:.2f}")
    
    def update_demand(self, predicted_demand: float, actual_demand: float = None) -> None:
        """
        Update demand information in the metrics tracker.
        
        If actual_demand is not provided, it will be calculated as predicted_demand + noise
        to simulate demand uncertainty.
        
        Args:
            predicted_demand: The predicted demand for the current step
            actual_demand: The actual demand after adding noise (if not provided)
        """
        self.predicted_demand = predicted_demand
        
        # If actual demand is not provided, simulate it with noise
        if actual_demand is None:
            noise = np.random.normal(0, self.sigma)
            self.realized_demand = predicted_demand + noise
        else:
            self.realized_demand = actual_demand
        
        self.iso_metrics['predicted_demands'].append(self.predicted_demand)
        self.iso_metrics['realized_demands'].append(self.realized_demand)
        
        # Calculate net demand (realized + PCS contribution)
        net_demand = self.realized_demand + self.pcs_demand
        self.iso_metrics['net_demands'].append(net_demand)
        
        if self.logger:
            self.logger.debug(f"Updated demand: predicted={predicted_demand:.2f}, realized={self.realized_demand:.2f}, net={net_demand:.2f}")
    
    def update_energy_exchange(self, energy_needed: float, cost: float) -> None:
        """
        Update energy exchange information from PCS to grid.
        
        Args:
            energy_needed: Amount of energy exchanged (positive = buying from grid, negative = selling to grid)
            cost: Associated cost or revenue of the exchange
        """
        # Add to energy tracking
        if energy_needed > 0:  # PCS buying from grid
            self.energy_bought += energy_needed
        else:  # PCS selling to grid
            self.energy_sold += abs(energy_needed)
        
        # Update PCS demand
        self.pcs_demand = energy_needed / self.env_config.get('time_step', 0.5/24)  # Convert energy to power
        self.iso_metrics['pcs_demands'].append(self.pcs_demand)
        
        # Update PCS metrics
        self.pcs_metrics['energy_exchanges'].append(energy_needed)
        self.pcs_metrics['costs'].append(cost)
        
        # Update ISO energy tracking
        self.iso_metrics['energy_bought'].append(self.energy_bought)
        self.iso_metrics['energy_sold'].append(self.energy_sold)
        
        if self.logger:
            self.logger.debug(f"Energy exchange: {energy_needed:.4f}, cost: {cost:.2f}")
    
    def update_battery_level(self, battery_level: float) -> None:
        """
        Update battery level in the metrics tracker.
        
        Args:
            battery_level: Current battery level
        """
        self.battery_level = battery_level
        self.pcs_metrics['battery_levels'].append(battery_level)
        
        if self.logger:
            self.logger.debug(f"Battery level updated: {battery_level:.4f}")
    
    def update_step_time(self, time: float) -> None:
        """
        Update time in the metrics tracker.
        
        Args:
            time: Current simulation time
        """
        self.shared_metrics['times'].append(time)
        self.shared_metrics['steps'].append(self.step_count)
        self.step_count += 1
        
        if self.logger:
            self.logger.debug(f"Step {self.step_count}, time: {time:.4f}")
    
    def update_episode_metrics(self, iso_reward: float, pcs_reward: float) -> None:
        """
        Update episode-level metrics at each step.
        
        Args:
            iso_reward: Current step ISO reward
            pcs_reward: Current step PCS reward
        """
        # Track cumulative rewards
        self.total_iso_reward += iso_reward
        self.total_pcs_reward += pcs_reward
        
        # Track episode rewards
        self.shared_metrics['episode_rewards'].append(self.total_iso_reward + self.total_pcs_reward)
        
        # Track episode costs from current step
        if self.iso_metrics['total_costs']:
            latest_cost = self.iso_metrics['total_costs'][-1]
            self.shared_metrics['episode_costs'].append(latest_cost)
        
        # Track episode battery utilization from current step
        if self.pcs_metrics['battery_utilization']:
            latest_utilization = self.pcs_metrics['battery_utilization'][-1]
            self.shared_metrics['episode_utilization'].append(latest_utilization)
        
        if self.logger:
            self.logger.debug(f"Episode metrics updated: ISO reward={iso_reward:.4f}, PCS reward={pcs_reward:.4f}")
    
    def calculate_grid_stability(self) -> float:
        """
        Calculate grid stability metric for ISO reward.
        
        Returns:
            A value representing grid stability (higher is better)
        """
        # Get dispatch level - if not available, default to predicted demand
        dispatch_level = self.iso_metrics['dispatch_levels'][-1] if self.iso_metrics['dispatch_levels'] else self.predicted_demand
        
        # Calculate shortfall (if net demand exceeds what was prepared for)
        shortfall = max(0.0, self.realized_demand + self.pcs_demand - dispatch_level)
        self.iso_metrics['shortfalls'].append(shortfall)
        
        # Calculate reserve cost
        reserve_cost = shortfall * self.reserve_price
        self.iso_metrics['reserve_costs'].append(reserve_cost)
        
        # Calculate dispatch cost (based on dispatch level)
        dispatch_cost = dispatch_level * self.dispatch_price
        self.iso_metrics['dispatch_costs'].append(dispatch_cost)
        
        # Track dispatch and reserve levels (amounts)
        self.iso_metrics['dispatch_levels'].append(dispatch_level)
        self.iso_metrics['reserve_levels'].append(shortfall)
        
        # Calculate total cost
        total_cost = dispatch_cost + reserve_cost
        self.iso_metrics['total_costs'].append(total_cost)
        
        # Calculate grid stability (negative of shortfall cost)
        grid_stability = -reserve_cost
        
        if self.logger:
            self.logger.debug(f"Grid stability: {grid_stability:.2f} (shortfall: {shortfall:.2f}, reserve cost: {reserve_cost:.2f})")
        
        return grid_stability
    
    def calculate_iso_revenue(self) -> float:
        """
        Calculate revenue for ISO from energy exchange with PCS.
        
        Returns:
            Revenue value for ISO
        """
        # Calculate revenue from energy sold to PCS
        revenue_from_sales = self.iso_sell_price * self.energy_bought
        
        # Calculate cost from energy bought from PCS
        cost_from_purchases = self.iso_buy_price * self.energy_sold
        
        # Net revenue
        net_revenue = revenue_from_sales - cost_from_purchases
        self.iso_metrics['revenues'].append(net_revenue)
        
        if self.logger:
            self.logger.debug(f"ISO revenue: {net_revenue:.2f} (sales: {revenue_from_sales:.2f}, purchases: {cost_from_purchases:.2f})")
        
        return net_revenue
    
    def calculate_total_pcs_cost(self) -> float:
        """
        Calculate total cost incurred by PCS.
        
        Returns:
            Total cost value (positive means PCS paid money)
        """
        # Use external reward function if available
        if self.reward_function and hasattr(self.reward_function, 'calculate_total_cost'):
            # Pass the costs list to the reward function
            costs_data = {
                'energy_costs': self.pcs_metrics['costs'],
                'energy_exchanges': self.pcs_metrics['energy_exchanges'],
                'battery_levels': self.pcs_metrics['battery_levels']
            }
            return self.reward_function.calculate_total_cost(costs_data)
            
        # Otherwise calculate internally
        if not self.pcs_metrics['costs']:
            return 0.0
        
        total_cost = sum(self.pcs_metrics['costs'])
        
        if self.logger:
            self.logger.debug(f"PCS total cost: {total_cost:.2f}")
        
        return total_cost
    
    def calculate_battery_utilization(self) -> float:
        """
        Calculate battery utilization metric for PCS reward.
        
        Returns:
            A value representing battery utilization (higher is better)
        """
        # If we have an external reward function, use it for consistency
        if self.reward_function and hasattr(self.reward_function, 'calculate_battery_utilization'):
            distance = self.reward_function.calculate_battery_utilization(self.battery_level)
            self.pcs_metrics['battery_utilization'].append(distance)
            return distance
            
        # Otherwise, calculate internally
        # Get battery parameters
        battery_config = self.pcs_config.get('battery', {}).get('model_parameters', {})
        min_level = battery_config.get('min', 0.0)
        max_level = battery_config.get('max', 1.0)
        optimal_level = (min_level + max_level) / 2
        
        # Calculate distance from optimal level (normalized to -1 to 0 scale)
        distance = -abs(self.battery_level - optimal_level) / (max_level - min_level)
        self.pcs_metrics['battery_utilization'].append(distance)
        
        if self.logger:
            self.logger.debug(f"Battery utilization: {distance:.4f} (level: {self.battery_level:.2f}, optimal: {optimal_level:.2f})")
        
        return distance
    
    def calculate_iso_reward(self) -> float:
        """
        Calculate overall reward for ISO agent.
        
        Returns:
            Combined reward value for ISO agent
        """
        # Calculate component rewards
        grid_stability = self.calculate_grid_stability()
        revenue = self.calculate_iso_revenue()
        
        # Get reward weights from config
        reward_config = self.iso_config.get('reward', {})
        stability_weight = reward_config.get('stability_weight', 1.0)
        revenue_weight = reward_config.get('revenue_weight', 0.5)
        
        # Calculate combined reward
        reward = (stability_weight * grid_stability) + (revenue_weight * revenue)
        
        # Track reward
        self.iso_metrics['rewards'].append(reward)
        self.total_iso_reward += reward
        
        if self.logger:
            self.logger.debug(f"ISO reward: {reward:.4f} (stability: {grid_stability:.2f}, revenue: {revenue:.2f})")
        
        return reward
    
    def calculate_pcs_reward(self) -> float:
        """
        Calculate overall reward for PCS agent.
        
        Returns:
            Combined reward value for PCS agent
        """
        # If we have an external reward function with a compute method, use it
        if self.reward_function and hasattr(self.reward_function, 'compute'):
            # Create state dict for reward function
            state = {
                'battery_level': self.battery_level,
                'costs': self.pcs_metrics['costs'],
                'energy_exchanges': self.pcs_metrics['energy_exchanges'],
                'battery_levels': self.pcs_metrics['battery_levels'],
                'iso_buy_price': self.iso_buy_price,
                'iso_sell_price': self.iso_sell_price
            }
            
            # Let the reward function calculate the reward using its own logic
            reward = self.reward_function.compute(state)
            
            # Track reward
            self.pcs_metrics['rewards'].append(reward)
            self.total_pcs_reward += reward
            
            if self.logger:
                self.logger.debug(f"PCS reward (from external function): {reward:.4f}")
                
            return reward
        
        # Otherwise, calculate internally
        # Calculate component rewards
        cost_reward = -self.calculate_total_pcs_cost()
        utilization_reward = self.calculate_battery_utilization()
        
        # Get reward weights from config
        reward_config = self.pcs_config.get('reward', {})
        cost_weight = reward_config.get('cost_weight', 1.0)
        utilization_weight = reward_config.get('utilization_weight', 0.5)
        
        # Calculate combined reward
        reward = (cost_weight * cost_reward) + (utilization_weight * utilization_reward)
        
        # Track reward
        self.pcs_metrics['rewards'].append(reward)
        self.total_pcs_reward += reward
        
        if self.logger:
            self.logger.debug(f"PCS reward: {reward:.4f} (cost: {cost_reward:.2f}, utilization: {utilization_reward:.2f})")
        
        return reward
    
    def update_iso_action(self, action: Any) -> None:
        """
        Update ISO actions in the metrics tracker.
        
        Args:
            action: The ISO action to track
        """
        self.last_iso_action = action
        self.iso_metrics['actions'].append(action)
        
        if self.logger:
            self.logger.debug(f"ISO action tracked: {action}")
            
    def update_pcs_action(self, action: Any) -> None:
        """
        Update PCS actions in the metrics tracker.
        
        Args:
            action: The PCS action to track
        """
        self.last_pcs_action = action
        self.pcs_metrics['actions'].append(action)
        
        if self.logger:
            self.logger.debug(f"PCS action tracked: {action}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for both agents.
        
        Returns:
            Dictionary containing metrics for ISO, PCS, and shared metrics
        """
        return {
            'iso': {
                'predicted_demands': self.iso_metrics['predicted_demands'],
                'realized_demands': self.iso_metrics['realized_demands'],
                'pcs_demands': self.iso_metrics['pcs_demands'],
                'net_demands': self.iso_metrics['net_demands'],
                'shortfalls': self.iso_metrics['shortfalls'],
                'dispatch_costs': self.iso_metrics['dispatch_costs'],
                'reserve_costs': self.iso_metrics['reserve_costs'],
                'total_costs': self.iso_metrics['total_costs'],
                'buy_prices': self.iso_metrics['buy_prices'],
                'sell_prices': self.iso_metrics['sell_prices'],
                'energy_bought': self.iso_metrics['energy_bought'],
                'energy_sold': self.iso_metrics['energy_sold'],
                'revenues': self.iso_metrics['revenues'],
                'rewards': self.iso_metrics['rewards'],
                'actions': self.iso_metrics['actions'],
                'total_reward': self.total_iso_reward
            },
            'pcs': {
                'battery_levels': self.pcs_metrics['battery_levels'],
                'energy_exchanges': self.pcs_metrics['energy_exchanges'],
                'costs': self.pcs_metrics['costs'],
                'rewards': self.pcs_metrics['rewards'],
                'battery_utilization': self.pcs_metrics['battery_utilization'],
                'total_reward': self.total_pcs_reward
            },
            'shared': {
                'times': self.shared_metrics['times'],
                'steps': self.shared_metrics['steps'],
                'episode': self.episode_count,
                'step_count': self.step_count
            }
        }
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the current episode.
        
        Returns:
            Dictionary containing summary metrics for both agents
        """
        # Mark episode as complete
        self.episode_count += 1
        
        # Calculate ISO summary stats
        iso_rewards = np.array(self.iso_metrics['rewards'])
        iso_summary = {
            'mean_reward': float(np.mean(iso_rewards)) if len(iso_rewards) > 0 else 0.0,
            'total_reward': self.total_iso_reward,
            'final_price_spread': self.iso_buy_price - self.iso_sell_price,
            'total_energy_bought': self.energy_bought,
            'total_energy_sold': self.energy_sold
        }
        
        # Calculate PCS summary stats
        pcs_rewards = np.array(self.pcs_metrics['rewards'])
        pcs_summary = {
            'mean_reward': float(np.mean(pcs_rewards)) if len(pcs_rewards) > 0 else 0.0,
            'total_reward': self.total_pcs_reward,
            'final_battery_level': self.battery_level,
            'total_cost': sum(self.pcs_metrics['costs'])
        }
        
        if self.logger:
            self.logger.info(f"Episode {self.episode_count} summary:")
            self.logger.info(f"  ISO total reward: {iso_summary['total_reward']:.2f}")
            self.logger.info(f"  PCS total reward: {pcs_summary['total_reward']:.2f}")
            self.logger.info(f"  Steps completed: {self.step_count}")
        
        return {
            'iso': iso_summary,
            'pcs': pcs_summary,
            'steps': self.step_count,
            'episode': self.episode_count
        }
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Generate plots of key metrics for analysis.
        
        Args:
            save_path: If provided, plots will be saved to this path
        """
        # Skip if no data
        if len(self.shared_metrics['times']) == 0:
            return
        
        # Set up figure - use 4x2 to accommodate action plots
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        times = self.shared_metrics['times']
        
        # ISO metrics
        axs[0, 0].plot(times, self.iso_metrics['predicted_demands'], label='Predicted Demand')
        axs[0, 0].plot(times, self.iso_metrics['realized_demands'], label='Realized Demand')
        axs[0, 0].plot(times, self.iso_metrics['pcs_demands'], label='PCS Demand')
        axs[0, 0].set_title('Demand')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Power (MW)')
        axs[0, 0].legend()
        
        axs[0, 1].plot(times, self.iso_metrics['buy_prices'], label='Buy Price')
        axs[0, 1].plot(times, self.iso_metrics['sell_prices'], label='Sell Price')
        axs[0, 1].set_title('ISO Prices')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Price ($/MWh)')
        axs[0, 1].legend()
        
        # PCS metrics
        axs[1, 0].plot(times, self.pcs_metrics['battery_levels'])
        axs[1, 0].set_title('Battery Level')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Level (MWh)')
        
        axs[1, 1].plot(times, self.pcs_metrics['energy_exchanges'])
        axs[1, 1].set_title('Energy Exchange')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Energy (MWh)')
        axs[1, 1].axhline(y=0, color='r', linestyle='--')
        
        # Rewards
        axs[2, 0].plot(times, self.iso_metrics['rewards'])
        axs[2, 0].set_title('ISO Rewards')
        axs[2, 0].set_xlabel('Time')
        axs[2, 0].set_ylabel('Reward')
        
        axs[2, 1].plot(times, self.pcs_metrics['rewards'])
        axs[2, 1].set_title('PCS Rewards')
        axs[2, 1].set_xlabel('Time')
        axs[2, 1].set_ylabel('Reward')
        
        # Agent Actions (new plots)
        if 'actions' in self.iso_metrics and len(self.iso_metrics['actions']) > 0:
            try:
                action_data = np.array(self.iso_metrics['actions'])
                if len(action_data.shape) > 1 and action_data.shape[1] > 1:
                    # Multiple dimensions in action
                    for i in range(min(3, action_data.shape[1])):  # Limit to first 3 dimensions
                        axs[3, 0].plot(times[:len(action_data)], action_data[:, i], 
                                     label=f'Dim {i}')
                else:
                    # Single dimension
                    axs[3, 0].plot(times[:len(action_data)], action_data, label='Action')
                
                axs[3, 0].set_title('ISO Actions')
                axs[3, 0].set_xlabel('Time')
                axs[3, 0].set_ylabel('Action Value')
                axs[3, 0].legend()
            except (ValueError, IndexError, TypeError) as e:
                if self.logger:
                    self.logger.warning(f"Error plotting ISO actions: {e}")
                axs[3, 0].set_title('ISO Actions (Error)')
                axs[3, 0].set_xlabel('Time')
                axs[3, 0].set_ylabel('Action Value')
        else:
            axs[3, 0].set_title('ISO Actions (No Data)')
            axs[3, 0].set_xlabel('Time')
            axs[3, 0].set_ylabel('Action Value')
        
        if 'actions' in self.pcs_metrics and len(self.pcs_metrics['actions']) > 0:
            try:
                action_data = np.array(self.pcs_metrics['actions'])
                if len(action_data.shape) > 1 and action_data.shape[1] > 1:
                    # Multiple dimensions in action
                    for i in range(min(3, action_data.shape[1])):  # Limit to first 3 dimensions
                        axs[3, 1].plot(times[:len(action_data)], action_data[:, i], 
                                     label=f'Dim {i}')
                else:
                    # Single dimension
                    axs[3, 1].plot(times[:len(action_data)], action_data, label='Action')
                
                axs[3, 1].set_title('PCS Actions')
                axs[3, 1].set_xlabel('Time')
                axs[3, 1].set_ylabel('Action Value')
                axs[3, 1].legend()
            except (ValueError, IndexError, TypeError) as e:
                if self.logger:
                    self.logger.warning(f"Error plotting PCS actions: {e}")
                axs[3, 1].set_title('PCS Actions (Error)')
                axs[3, 1].set_xlabel('Time')
                axs[3, 1].set_ylabel('Action Value')
        else:
            axs[3, 1].set_title('PCS Actions (No Data)')
            axs[3, 1].set_xlabel('Time')
            axs[3, 1].set_ylabel('Action Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            if self.logger:
                self.logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close(fig)
        
    def plot_legacy_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Generate the exact three plots from the old pipeline for compatibility.
        
        Plot 1: ISO Metrics (Demand and Prices)
        Plot 2: PCS Metrics (Battery Level and Energy Exchange)
        Plot 3: Rewards (ISO and PCS)
        
        Args:
            save_path: If provided, plots will be saved to this path
        """
        # Skip if no data
        if len(self.shared_metrics['times']) == 0:
            return
            
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        times = self.shared_metrics['times']
        
        # Plot 1: ISO Metrics - Demand and Prices
        ax1 = axs[0]
        
        # Demand on left y-axis
        ax1.plot(times, self.iso_metrics['predicted_demands'], 'b-', label='Predicted Demand')
        ax1.plot(times, self.iso_metrics['realized_demands'], 'g--', label='Realized Demand')
        ax1.plot(times, self.iso_metrics['pcs_demands'], 'r-.', label='PCS Demand')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Power (MW)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Prices on right y-axis
        ax1_twin = ax1.twinx()
        ax1_twin.plot(times, self.iso_metrics['buy_prices'], 'm-', label='Buy Price')
        ax1_twin.plot(times, self.iso_metrics['sell_prices'], 'c--', label='Sell Price')
        ax1_twin.set_ylabel('Price ($/MWh)', color='m')
        ax1_twin.tick_params(axis='y', labelcolor='m')
        
        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        
        ax1.set_title('ISO Metrics: Demand and Prices')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: PCS Metrics - Battery Level and Energy Exchange
        ax2 = axs[1]
        
        # Battery level on left y-axis
        ax2.plot(times, self.pcs_metrics['battery_levels'], 'g-', label='Battery Level')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Battery Level (MWh)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Energy exchange on right y-axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(times, self.pcs_metrics['energy_exchanges'], 'b-', label='Energy Exchange')
        ax2_twin.axhline(y=0, color='r', linestyle='--')
        ax2_twin.set_ylabel('Energy Exchange (MWh)', color='b')
        ax2_twin.tick_params(axis='y', labelcolor='b')
        
        # Combine legends
        lines_1, labels_1 = ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        
        ax2.set_title('PCS Metrics: Battery Level and Energy Exchange')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rewards - ISO and PCS
        ax3 = axs[2]
        
        ax3.plot(times, self.iso_metrics['rewards'], 'b-', label='ISO Reward')
        ax3.plot(times, self.pcs_metrics['rewards'], 'g-', label='PCS Reward')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Reward')
        ax3.legend(loc='upper right')
        ax3.set_title('Agent Rewards')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # If save_path includes file extension, remove it to add the _legacy suffix
            if save_path.endswith('.png') or save_path.endswith('.jpg') or save_path.endswith('.pdf'):
                save_path = save_path.rsplit('.', 1)[0]
            
            legacy_save_path = f"{save_path}_legacy.png"
            plt.savefig(legacy_save_path)
            if self.logger:
                self.logger.info(f"Legacy metrics plot saved to {legacy_save_path}")
        else:
            plt.show()
            
        plt.close(fig)

    def update_dispatch_level(self, dispatch_level: float) -> None:
        """
        Update dispatch level in the metrics tracker.
        
        Args:
            dispatch_level: The ISO-determined dispatch level
        """
        # Store in metrics for shortfall calculation
        if 'dispatch_levels' not in self.iso_metrics:
            self.iso_metrics['dispatch_levels'] = []
        
        self.iso_metrics['dispatch_levels'].append(dispatch_level)
        
        if self.logger:
            self.logger.debug(f"Updated dispatch level: {dispatch_level:.4f}")
