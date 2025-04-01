"""
ISO Metrics Handler Module

This module handles all calculations related to grid state, costs, rewards, and metrics
for the ISO controller. It encapsulates the complex logic of determining grid performance,
calculating costs associated with different operations, and computing rewards for the
reinforcement learning agent.

Key functions:
1. Grid state calculation (demand realization, shortfall determination)
2. Cost component calculations (dispatch costs, reserve costs, PCS exchange costs)
3. Reward computation based on configurable parameters
4. Info dictionary building for tracking and visualization

By isolating these calculations in a separate module, we improve modularity and testability
of the ISO controller while keeping the core logic clean.
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

class ISOMetricsHandler:
    """
    Handles the calculation of grid state, costs, rewards, and metrics for the ISO controller.
    
    This class is responsible for:
    1. Calculating grid state variables (e.g., realized demand, shortfall)
    2. Computing costs (dispatch, reserve)
    3. Building the info dictionary for the step function
    4. Computing rewards
    5. Logging relevant metrics
    
    By extracting this logic from the ISO controller, we make the controller cleaner and more focused
    on its core responsibilities, while making the metrics calculations more maintainable and testable.
    """
    
    def __init__(self, config: Dict[str, Any], reward_calculator, logger: Optional[logging.Logger] = None):
        """
        Initialize the metrics handler.
        
        Args:
            config: Dictionary containing environment configuration
                Expected keys include:
                - dispatch_price: Cost per unit of dispatched energy
                - reserve_price: Cost per unit of reserve energy needed
                - demand_uncertainty: Dictionary with sigma parameter for demand noise
            reward_calculator: The reward calculator instance that computes agent rewards
            logger: Logger instance for logging metrics
        """
        self.config = config
        self.reward_calculator = reward_calculator
        self.logger = logger
        
        # Extract configuration values that might be used across multiple calculations
        self.dispatch_price = config.get('dispatch_price', 0.0)
        self.reserve_price = config.get('reserve_price', 0.0)
        self.sigma = config.get('demand_uncertainty', {}).get('sigma', 0.0)
        
        if self.logger:
            self.logger.info(f"Initialized metrics handler with dispatch_price: ${self.dispatch_price:.2f}, reserve_price: ${self.reserve_price:.2f}")
    
    def calculate_grid_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate grid state, costs, and metrics based on the current state.
        
        This method performs the following calculations:
        1. Applies random noise to predicted demand to simulate forecast uncertainty
        2. Calculates net demand by combining realized demand and PCS demand
        3. Determines shortfall (when net demand exceeds dispatch)
        4. Calculates component costs (dispatch, reserve, PCS exchange)
        5. Computes reward using the reward calculator
        6. Builds a comprehensive info dictionary for monitoring and visualization
        
        The calculations follow the principles of power grid operations, where dispatch
        decisions are made based on demand forecasts, and reserve power addresses
        any shortfall at a premium cost.
        
        Args:
            state: Dictionary containing the current state variables
                Required keys:
                - predicted_demand: Current predicted demand
                - pcs_demand: Current PCS demand
                - iso_buy_price: Current ISO buy price
                - iso_sell_price: Current ISO sell price
                - dispatch: Current dispatch level
                - count: Current step count
                - current_time: Current simulation time
                Optional keys:
                - production: Current production
                - consumption: Current consumption
                - battery_level: Current battery level
                - battery_actions: Current battery actions
                
        Returns:
            Dictionary containing calculated metrics:
                - reward: Calculated reward
                - info: Info dictionary for the step function
                - realized_demand: Realized demand (predicted + noise)
                - shortfall: Amount by which net demand exceeds dispatch
                - dispatch_price: Price per unit of dispatched power
                - reserve_price: Price per unit of reserve power
        """
        # Extract required values from state
        predicted_demand = state['predicted_demand']
        pcs_demand = state['pcs_demand']
        iso_buy_price = state['iso_buy_price']
        iso_sell_price = state['iso_sell_price']
        dispatch = state['dispatch']
        count = state['count']
        production = state.get('production', 0.0)
        consumption = state.get('consumption', 0.0)
        battery_level = state.get('battery_level', [])
        battery_actions = state.get('battery_actions', [])
        
        # Calculate grid state
        noise = np.random.normal(0, self.sigma)
        realized_demand = float(predicted_demand + noise)
        net_demand = realized_demand + pcs_demand
        
        if self.logger:
            self.logger.debug(f"Net demand: {net_demand:.2f} MWh")
        
        # Calculate costs
        dispatch_cost = self.dispatch_price * dispatch
        shortfall = max(0.0, net_demand - dispatch)
        
        if pcs_demand > 0: 
            price = iso_sell_price
        else:
            price = iso_buy_price
        
        pcs_costs = pcs_demand * price
        reserve_cost = self.reserve_price * shortfall
        
        if self.logger:
            self.logger.info(
                f"Cost Calculation:\n"
                f"  - Dispatch: {dispatch:.2f} MWh @ ${self.dispatch_price:.2f}/MWh = ${dispatch_cost:.2f}\n"
                f"  - Shortfall: {shortfall:.2f} MWh @ ${self.reserve_price:.2f}/MWh = ${reserve_cost:.2f}\n"
                f"  - PCS Exchange: {pcs_demand:.2f} MWh @ ${price:.2f}/MWh = ${pcs_costs:.2f}"
            )
            
            self.logger.warning(
                f"Grid Shortfall:\n"
                f"  - Amount: {shortfall:.2f} MWh\n"
                f"  - Reserve Cost: ${reserve_cost:.2f}"
            )
        
        # Build info dictionary
        info = {
            'predicted_demand': predicted_demand,
            'realized_demand': realized_demand,
            'pcs_demand': pcs_demand,
            'net_demand': net_demand,
            'dispatch': dispatch,
            'shortfall': shortfall,
            'dispatch_cost': dispatch_cost,
            'reserve_cost': reserve_cost,
            'pcs_costs': pcs_costs,
            'production': production,
            'consumption': consumption,
            'battery_level': battery_level.tolist() if isinstance(battery_level, np.ndarray) else battery_level,
            'battery_actions': battery_actions.tolist() if isinstance(battery_actions, np.ndarray) else battery_actions,
            'buy_price': iso_buy_price,
            'sell_price': iso_sell_price,
            'iso_buy_price': iso_buy_price,
            'iso_sell_price': iso_sell_price,
            'net_exchange': pcs_demand,
            'pcs_cost': pcs_costs,
            'pcs_actions': battery_actions.tolist() if isinstance(battery_actions, np.ndarray) else battery_actions
        }
        
        # Calculate reward using reward calculator
        reward = self.reward_calculator.compute_reward(info)
        
        if self.logger:
            self.logger.info(f"Step reward: {reward:.2f}")
            self.logger.debug(f"Reward calculation inputs: reserve_cost={info.get('reserve_cost', 0.0)}, pcs_demand={info.get('pcs_demand', 0.0)}, dispatch_cost={info.get('dispatch_cost', 0.0)}")
            
            # Log grid state
            self.logger.info(
                f"Grid State Step {count}:\n"
                f"  Time: {state.get('current_time', 'N/A')}\n"
                f"  Predicted Demand: {predicted_demand:.2f} MWh\n"
                f"  Realized Demand: {realized_demand:.2f} MWh\n"
                f"  PCS Demand: {pcs_demand:.2f} MWh\n"
                f"  Net Demand: {net_demand:.2f} MWh\n"
                f"  Shortfall: {shortfall:.2f} MWh"
            )
            
            # Log financial metrics
            self.logger.info(
                f"Financial Metrics:\n"
                f"  Dispatch Cost: ${dispatch_cost:.2f}\n"
                f"  Reserve Cost: ${reserve_cost:.2f}\n" 
                f"  Total Cost: ${(dispatch_cost + reserve_cost):.2f}"
            )
        
        return {
            'reward': reward,
            'info': info,
            'realized_demand': realized_demand,
            'shortfall': shortfall,
            'dispatch_price': self.dispatch_price,
            'reserve_price': self.reserve_price
        }