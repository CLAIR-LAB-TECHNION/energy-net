"""
Energy Net Controller Module

This module implements a unified controller for the Energy Net environment,
combining ISO (Independent System Operator) and PCS (Power Consumption & Storage)
components into a single, sequential simulation.

Key responsibilities:
1. Managing both ISO and PCS components in a unified timeline
2. Processing actions from both agents in the correct sequence
3. Tracking shared state variables and energy exchanges
4. Calculating rewards for both agents
5. Generating observations for both agents
6. Providing direct access to comprehensive metrics

The controller follows a sequential flow where:
1. ISO agent sets energy prices
2. PCS agent responds with battery control actions
3. Energy exchanges occur
4. State updates and rewards are calculated

This unified approach eliminates the need for manual transfers between 
separate environments and provides a more realistic simulation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
import yaml
import os
from typing import Dict, Any, Tuple, Union, List, Optional
from stable_baselines3 import PPO

from energy_net.utils.logger import setup_logger
from energy_net.market.pricing_policy import PricingPolicy
from energy_net.market.iso.cost_types import CostType, calculate_costs
from energy_net.market.iso.demand_patterns import DemandPattern, calculate_demand
from energy_net.controllers.iso.pricing_strategy import PricingStrategyFactory
from energy_net.controllers.unified_metrics_handler import UnifiedMetricsHandler
from energy_net.controllers.pcs.battery_manager import BatteryManager

# Import PCSUnit for full functionality
from energy_net.components.pcsunit import PCSUnit
from energy_net.dynamics.energy_dynamcis import EnergyDynamics, ModelBasedDynamics
from energy_net.dynamics.production_dynamics.deterministic_production import DeterministicProduction
from energy_net.dynamics.consumption_dynamics.deterministic_consumption import DeterministicConsumption
from energy_net.dynamics.storage_dynamics.deterministic_battery import DeterministicBattery

# Import reward classes for reference in metrics handler
from energy_net.rewards.base_reward import BaseReward
from energy_net.rewards.cost_reward import CostReward


class EnergyNetController:
    """
    Unified controller for the Energy Net environment, integrating both the
    ISO (Independent System Operator) and PCS (Power Consumption & Storage) components.
    
    This controller manages the sequential simulation of energy market dynamics,
    where the ISO sets prices and the PCS responds with battery actions.
    
    The controller maintains a single timeline and shared state variables,
    eliminating the need for manual transfers between separate environments.
    
    Key features:
    - Unified observation and action spaces for both agents
    - Sequential processing of agent actions
    - Direct access to comprehensive metrics
    - Shared state tracking for consistent simulation
    
    Observation Space:
        ISO: [time, predicted_demand, pcs_demand]
        PCS: [battery_level, time, iso_buy_price, iso_sell_price]
        
    Action Space:
        ISO: Depends on pricing policy (ONLINE, QUADRATIC, CONSTANT)
        PCS: Battery charging/discharging rate
    """
    
    def __init__(
        self,
        cost_type=None,
        pricing_policy=None,
        demand_pattern=None,
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
        Initialize the unified Energy Net controller.
        
        Args:
            cost_type: How grid operation costs are calculated
            pricing_policy: Policy for determining energy prices (ONLINE, QUADRATIC, CONSTANT)
            demand_pattern: Pattern of demand variation over time
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
        # Set up logger
        self.log_file = log_file  # Store log_file as instance attribute
        self.logger = setup_logger('EnergyNetController', log_file)
        self.logger.info(f"Initializing EnergyNetController with {pricing_policy.value} pricing policy")
        
        # Store configuration parameters
        self.pricing_policy = pricing_policy
        self.cost_type = cost_type
        self.demand_pattern = demand_pattern
        self.num_pcs_agents = num_pcs_agents
        self.logger.info(f"Using demand pattern: {demand_pattern.value}")
        self.logger.info(f"Using cost type: {cost_type.value}")

        # Load configurations
        self.env_config = self._load_config(env_config_path)
        self.iso_config = self._load_config(iso_config_path)
        self.pcs_unit_config = self._load_config(pcs_unit_config_path)

        # Initialize shared state variables
        self.current_time = 0.0
        self.count = 0  # Use count instead of step_count to match single agent controllers
        self.terminated = False
        self.truncated = False
        self.first_action_taken = False
        
        # ISO control variables
        self.use_dispatch_action = self.iso_config.get('dispatch', {}).get('use_dispatch_action', False)
        self.predicted_demand = 0.0
        self.dispatch = 0.0  # Track the dispatch value
        
        # Energy exchange tracking
        self.iso_buy_price = 0.0
        self.iso_sell_price = 0.0
        self.energy_bought = 0.0
        self.energy_sold = 0.0
        
        # Initialize time parameters using the same approach as in the single agent controllers
        self.time_steps_per_day_ratio = self.env_config['time']['time_steps_per_day_ratio']
        self.time_step_duration = self.env_config['time']['step_duration']
        self.max_steps_per_episode = self.env_config['time'].get('max_steps_per_episode', 48)
        
        # Get costs from cost type
        self.reserve_price, self.dispatch_price = calculate_costs(
            cost_type,
            self.env_config
        )
        
        # Initialize ISO components
        self._init_iso_components(dispatch_config)
        
        # Initialize PCS components
        self._init_pcs_components()
        
        # Create observation and action spaces
        self._create_observation_spaces()
        self._create_action_spaces()
        
        self.logger.info("EnergyNetController initialized successfully")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def _init_iso_components(self, dispatch_config: Optional[Dict[str, Any]]):
        """Initialize ISO-specific components"""
        # Define price bounds from ISO config
        pricing_config = self.iso_config.get('pricing', {})
        price_params = pricing_config.get('parameters', {})
        self.min_price = price_params.get('min_price', pricing_config.get('default_sell_price', 1.0))
        self.max_price = price_params.get('max_price', pricing_config.get('default_buy_price', 10.0))
        
        # Check if dispatch should be included in actions
        dispatch_config_from_file = self.iso_config.get('dispatch', {})
        if dispatch_config:
            self.use_dispatch_action = dispatch_config.get('use_dispatch_action', 
                                       dispatch_config_from_file.get('use_dispatch_action', False))
        else:
            self.use_dispatch_action = dispatch_config_from_file.get('use_dispatch_action', False)
        
        # Initialize pricing strategy
        action_spaces_config = self.iso_config.get('action_spaces', {})
        self.pricing_strategy = PricingStrategyFactory.create_strategy(
            pricing_policy=self.pricing_policy,
            min_price=self.min_price,
            max_price=self.max_price,
            max_steps_per_episode=self.max_steps_per_episode,
            action_spaces_config=action_spaces_config,
            logger=self.logger
        )
        
        # Initialize demand prediction
        self.predicted_demand = 0.0
        self.actual_demand = 0.0
        self.pcs_demand = 0.0
        
        self.logger.info("ISO components initialized")

    def _init_pcs_components(self):
        """Initialize PCS-specific components"""
        # Initialize PCSUnit with dynamics and configuration
        self.pcs_unit = PCSUnit(
            config=self.pcs_unit_config,
            log_file=self.log_file
        )
        self.logger.info("Initialized PCSUnit with all components")
        
        # Initialize battery manager with reference to PCSUnit
        energy_config = self.pcs_unit_config['battery']['model_parameters']
        self.battery_manager = BatteryManager(
            battery_config=energy_config,
            pcsunit=self.pcs_unit,  # Pass PCSUnit reference for full functionality
            logger=self.logger
        )
        
        # Initialize multi-action support flags for future use
        self.multi_action = self.pcs_unit_config.get('action', {}).get('multi_action', False)
        self.production_action_enabled = self.pcs_unit_config.get('action', {}).get('production_action', {}).get('enabled', False)
        self.consumption_action_enabled = self.pcs_unit_config.get('action', {}).get('consumption_action', {}).get('enabled', False)
        self.logger.info(f"Multi-action support: {self.multi_action}, Production: {self.production_action_enabled}, Consumption: {self.consumption_action_enabled}")
        
        # Initialize the unified metrics handler with reward references
        # This allows the metrics handler to use the reward functions internally
        reward_type = self.pcs_unit_config.get('reward', {}).get('type', 'cost')
        self.reward_function = self._initialize_reward(reward_type)
        
        self.metrics = UnifiedMetricsHandler(
            env_config=self.env_config,
            iso_config=self.iso_config,
            pcs_config=self.pcs_unit_config,
            cost_type=self.cost_type,
            reward_function=self.reward_function,  # Pass reward function reference
            logger=self.logger
        )
        
        # Initialize battery state
        self.battery_level = self.battery_manager.get_level()
        
        self.logger.info("PCS components initialized")
        
    def _initialize_reward(self, reward_type: str) -> BaseReward:
        """Initialize reward function based on specified type"""
        if reward_type.lower() == 'cost':
            return CostReward()  # CostReward doesn't take a config parameter
        else:
            self.logger.warning(f"Unknown reward type: {reward_type}, defaulting to CostReward")
            return CostReward()

    def _create_observation_spaces(self):
        """Create observation spaces for both agents"""
        # ISO observation space
        iso_obs_config = self.iso_config.get('observation_space', {})
        time_config = iso_obs_config.get('time', {})
        demand_config = iso_obs_config.get('predicted_demand', {})
        pcs_config = iso_obs_config.get('pcs_demand', {})
        
        # Convert 'inf' strings from yaml to numpy.inf
        def convert_inf(value):
            if value == 'inf':
                return np.inf
            elif value == '-inf':
                return -np.inf
            return value
        
        self.iso_observation_space = spaces.Box(
            low=np.array([
                time_config.get('min', 0.0),
                demand_config.get('min', 0.0),
                convert_inf(pcs_config.get('min', -np.inf))
            ], dtype=np.float32),
            high=np.array([
                time_config.get('max', 1.0),
                convert_inf(demand_config.get('max', np.inf)),
                convert_inf(pcs_config.get('max', np.inf))
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # PCS observation space
        pcs_obs_config = self.pcs_unit_config.get('observation_space', {})
        energy_config = self.pcs_unit_config['battery']['model_parameters']
        
        # Get battery level bounds from battery config if specified
        battery_level_config = pcs_obs_config.get('battery_level', {})
        battery_min = energy_config['min'] if battery_level_config.get('min') == "from_battery_config" else battery_level_config.get('min', energy_config['min'])
        battery_max = energy_config['max'] if battery_level_config.get('max') == "from_battery_config" else battery_level_config.get('max', energy_config['max'])
        
        # Get other observation space bounds from config
        pcs_time_config = pcs_obs_config.get('time', {})
        buy_price_config = pcs_obs_config.get('iso_buy_price', {})
        sell_price_config = pcs_obs_config.get('iso_sell_price', {})
        
        self.pcs_observation_space = spaces.Box(
            low=np.array([
                battery_min,
                pcs_time_config.get('min', 0.0),
                buy_price_config.get('min', 0.0),
                sell_price_config.get('min', 0.0)
            ], dtype=np.float32),
            high=np.array([
                battery_max,
                pcs_time_config.get('max', 1.0),
                buy_price_config.get('max', 100.0),
                sell_price_config.get('max', 100.0)
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        self.logger.info(f"ISO observation space: {self.iso_observation_space}")
        self.logger.info(f"PCS observation space: {self.pcs_observation_space}")

    def _create_action_spaces(self):
        """Create action spaces for both agents"""
        # ISO action space based on pricing strategy
        self.iso_action_space = self.pricing_strategy.create_action_space(
            use_dispatch_action=self.use_dispatch_action
        )
        
        # PCS action space
        energy_config = self.pcs_unit_config['battery']['model_parameters']
        self.pcs_action_space = spaces.Box(
            low=np.array([
                -energy_config['discharge_rate_max']
            ], dtype=np.float32),
            high=np.array([
                energy_config['charge_rate_max']
            ], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )
        
        self.logger.info(f"ISO action space: {self.iso_action_space}")
        self.logger.info(f"PCS action space: {self.pcs_action_space}")

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple containing:
            - Initial observations for both agents
            - Info dictionary with initial state information
        """
        # Reset random generator if seed is provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset shared state variables
        self.current_time = 0.0
        self.count = 0  # Using count instead of step_count for consistency
        self.terminated = False
        self.truncated = False
        
        # Reset energy exchange tracking
        self.iso_buy_price = 0.0
        self.iso_sell_price = 0.0
        self.energy_bought = 0.0
        self.energy_sold = 0.0
        
        # Reset ISO components - Use _update_time_and_demand for consistency
        self._update_time_and_demand()
        self.actual_demand = self.predicted_demand  # At reset, these are the same
        self.pcs_demand = 0.0
        
        # Reset PCS components
        self.pcs_unit.reset()  # Reset the PCSUnit first
        self.battery_manager.reset()
        self.battery_level = self.battery_manager.get_level()
        
        # Reset metrics handler
        self.metrics.reset()
        
        # Generate initial observations
        iso_obs = self._get_iso_observation()
        pcs_obs = self._get_pcs_observation()
        
        # Generate info
        info = self._get_info()
        
        # Set initialization flag as in single agent controllers
        self.init = True
        
        return [iso_obs, pcs_obs], info

    def step(self, iso_action, pcs_action):
        """
        Take a step in the energy net environment with both ISO and PCS actions.
        
        Args:
            iso_action: Action from the ISO agent
            pcs_action: Action from the PCS agent
            
        Returns:
            tuple: (obs, reward, terminated, truncated, info) - standard gym environment return format
        """
        # Track the actions in metrics
        self.metrics.update_iso_action(iso_action)
        if hasattr(self.metrics, 'update_pcs_action'):
            self.metrics.update_pcs_action(pcs_action)
        
        # Update time and step count - Using the same approach as in PCSUnitController
        self.count += 1
        self.current_time = (self.count * self.time_step_duration) / self.env_config['time']['minutes_per_day']
        
        # Update demand prediction for this time step
        self._update_time_and_demand()
        
        # Process agent actions
        self._process_iso_action(iso_action)
        self._process_pcs_action(pcs_action)
        
        # Update grid state
        self._update_grid_state()
        
        # Calculate rewards
        iso_reward = self._calculate_iso_reward()
        pcs_reward = self._calculate_pcs_reward()
        self.metrics.update_episode_metrics(iso_reward, pcs_reward)
        
        # Check for termination - match single agent controllers' approach
        self.terminated = False  # No specific termination condition
        self.truncated = (self.count >= self.max_steps_per_episode)
        
        # Generate detailed info dict for monitoring
        info = self._get_detailed_info()
        
        # Log step completion
        if self.logger:
            self.logger.info(
                f"Step {self.count}: time={self.current_time:.2f}, "
                f"ISO reward={iso_reward:.4f}, PCS reward={pcs_reward:.4f}, "
                f"battery={self.battery_level:.2f}, prices: buy=${self.iso_buy_price:.2f}, sell=${self.iso_sell_price:.2f}"
            )
        
        # Return observation for both agents, rewards, termination flags, and info
        return self._get_obs(), (iso_reward, pcs_reward), self.terminated, self.truncated, info

    def _process_iso_action(self, iso_action):
        """
        Process ISO action to set prices.
        
        Args:
            iso_action: Action from ISO agent
            
        Returns:
            Tuple (buy_price, sell_price, dispatch): The prices and dispatch set by ISO
        """
        # Use pricing strategy to process action
        # Get new prices and dispatch based on ISO action
        self.iso_buy_price, self.iso_sell_price, dispatch, self.first_action_taken = self.pricing_strategy.process_action(
            action=iso_action,
            step_count=self.count,
            first_action_taken=self.first_action_taken,
            predicted_demand=self.predicted_demand,
            use_dispatch_action=self.use_dispatch_action
        )
        
        # Store the dispatch value
        self.dispatch = dispatch
        
        # Log prices
        self.logger.debug(f"ISO set prices: buy={self.iso_buy_price}, sell={self.iso_sell_price}, dispatch={self.dispatch}")
        
        # Update metrics with new prices
        self.metrics.update_prices(self.iso_buy_price, self.iso_sell_price)
        
        # Update metrics with dispatch level
        if hasattr(self.metrics, 'update_dispatch_level'):
            self.metrics.update_dispatch_level(self.dispatch)
        else:
            # Directly update the dispatch_levels list if method doesn't exist
            if hasattr(self.metrics, 'iso_metrics') and 'dispatch_levels' in self.metrics.iso_metrics:
                self.metrics.iso_metrics['dispatch_levels'].append(self.dispatch)
        
        return self.iso_buy_price, self.iso_sell_price, dispatch

    def _process_pcs_action(self, pcs_action):
        """Process PCS action for battery control"""
        # Check if using multi-action mode
        if self.multi_action:
            # Use PCSUnit to process the multi-action
            # This handles production, consumption, and storage actions together
            result = self.pcs_unit.process_action(pcs_action)
            
            # Get energy needed from the PCSUnit result
            energy_needed = result.get('grid_exchange', 0.0)
            
            # Update battery level from PCSUnit
            self.battery_level = self.pcs_unit.get_battery_level()
        else:
            # Legacy single-action mode
            # Extract battery command (charging/discharging rate)
            if isinstance(pcs_action, np.ndarray) and pcs_action.shape == (1,):
                battery_command = pcs_action[0]
            else:
                battery_command = pcs_action
            
            # CRITICAL FIX: Directly update the PCSUnit with the battery action
            time_fraction = self.count * self.time_step_duration / self.env_config['time']['minutes_per_day']
            self.pcs_unit.update(time=time_fraction, battery_action=battery_command)
            
            # Calculate energy change using the correct method
            energy_change, new_battery_level = self.battery_manager.calculate_energy_change(battery_command)
            
            # Update battery state
            actual_energy_change = self.battery_manager.update(battery_command)
            self.battery_level = self.battery_manager.get_level()
            
            # Set energy needed to the energy change for grid exchange calculations
            energy_needed = energy_change
            
            # Track charging/discharging rates
            if battery_command > 0:  # Charging
                self.metrics.pcs_metrics['charge_rates'].append(battery_command)
                self.metrics.pcs_metrics['discharge_rates'].append(0.0)
            else:  # Discharging or idle
                self.metrics.pcs_metrics['charge_rates'].append(0.0)
                self.metrics.pcs_metrics['discharge_rates'].append(abs(battery_command))
                
            # Track efficiency losses
            efficiency_loss = abs(energy_change - actual_energy_change)
            self.metrics.pcs_metrics['efficiency_losses'].append(efficiency_loss)
        
        # Execute energy exchange
        if energy_needed > 0:  # Buying from grid
            self.energy_bought += energy_needed
            cost = self.iso_buy_price * energy_needed
        else:  # Selling to grid
            self.energy_sold += abs(energy_needed)
            cost = self.iso_sell_price * energy_needed  # Note: energy_needed is negative
        
        # Update metrics with energy exchange and battery level
        self.metrics.update_energy_exchange(energy_needed, cost)
        self.metrics.update_battery_level(self.battery_level)
        
        # Track the action in metrics
        self.metrics.pcs_metrics['actions'].append(pcs_action)
        
        # Calculate time_step for conversion - matching PCSUnitController approach
        time_step = self.time_step_duration / self.env_config['time']['minutes_per_day']
        
        # Update PCS demand for ISO observation
        self.pcs_demand = energy_needed / time_step  # Convert energy to power
        
        self.logger.debug(f"PCS energy exchange: {energy_needed}, cost: {cost}, battery level: {self.battery_level:.4f}")

    def _update_time_and_demand(self):
        """Update time and predict demand for this step"""
        # No need to update time here, as it's already updated in the step method
        
        # Update demand prediction for this step
        self.predicted_demand = calculate_demand(
            time=self.current_time,
            pattern=self.demand_pattern,
            config=self.env_config['predicted_demand']
        )
        
        # Log updated time and demand prediction
        self.logger.debug(f"Updated time: {self.current_time}, step: {self.count}, predicted demand: {self.predicted_demand}")
        
        # Update metrics with the new time
        self.metrics.update_step_time(self.current_time)

    def _update_grid_state(self):
        """Update grid state based on energy exchange and calculate impacts"""
        # Update metrics with demand information - pass None for actual_demand to use noise
        self.metrics.update_demand(self.predicted_demand, None)
        
        # Get the realized demand with noise from metrics
        self.actual_demand = self.metrics.realized_demand
        
        # Calculate shortfall (if net demand exceeds what was prepared for)
        shortfall = max(0.0, self.actual_demand - self.dispatch)
        self.metrics.iso_metrics['shortfalls'].append(shortfall)
        
        # Calculate reserve cost (cost of addressing shortfall)
        reserve_cost = shortfall * self.reserve_price
        self.metrics.iso_metrics['reserve_costs'].append(reserve_cost)
        
        # Calculate dispatch cost (based on dispatch value, not predicted demand)
        dispatch_cost = self.dispatch * self.dispatch_price
        self.metrics.iso_metrics['dispatch_costs'].append(dispatch_cost)
        
        # Calculate total cost
        total_cost = dispatch_cost + reserve_cost
        self.metrics.iso_metrics['total_costs'].append(total_cost)
        
        # Calculate grid stability (negative of shortfall cost)
        grid_stability = -reserve_cost
        self.metrics.iso_metrics['grid_stability'].append(grid_stability)
        
        # Calculate ISO revenue from energy exchange with PCS
        iso_revenue = self.metrics.calculate_iso_revenue()
        self.metrics.iso_metrics['revenues'].append(iso_revenue)
        
        # Calculate and track PCS costs
        pcs_cost = self.metrics.calculate_total_pcs_cost()
        self.metrics.pcs_metrics['costs'].append(pcs_cost)
        
        # Calculate and track battery utilization
        battery_utilization = self.metrics.calculate_battery_utilization()
        self.metrics.pcs_metrics['battery_utilization'].append(battery_utilization)
        
        # Save net demand for tracking
        self.metrics.iso_metrics['net_demands'].append(self.actual_demand)
        
        self.logger.debug(
            f"Updated grid state: actual demand={self.actual_demand:.4f}, " +
            f"predicted={self.predicted_demand:.4f}, dispatch={self.dispatch:.4f}, " +
            f"shortfall={shortfall:.4f}, grid stability={grid_stability:.4f}"
        )

    def _check_termination(self):
        """Check if episode should terminate"""
        # Terminate if reached max steps - use same approach as single agent controllers
        self.terminated = False  # No specific termination condition
        self.truncated = (self.count >= self.max_steps_per_episode)
        
        if self.truncated:
            self.logger.info(f"Episode truncated after {self.count} steps")

    def _get_iso_observation(self):
        """Generate observation for ISO agent"""
        return np.array([
            self.current_time,
            self.predicted_demand,
            self.pcs_demand
        ], dtype=np.float32)

    def _get_pcs_observation(self):
        """Generate observation for PCS agent"""
        return np.array([
            self.battery_level,
            self.current_time,
            self.iso_buy_price,
            self.iso_sell_price
        ], dtype=np.float32)

    def _calculate_iso_reward(self):
        """Calculate reward for ISO agent"""
        # Use unified metrics handler to calculate ISO reward
        iso_reward = self.metrics.calculate_iso_reward()
        
        return iso_reward

    def _calculate_pcs_reward(self):
        """Calculate reward for PCS agent"""
        # Use unified metrics handler to calculate PCS reward
        pcs_reward = self.metrics.calculate_pcs_reward()
        
        return pcs_reward

    def _get_info(self):
        """Generate info dictionary with metrics"""
        # Get comprehensive metrics from unified handler
        metrics = self.metrics.get_metrics()
        
        # Add iso_total_reward at the top level for compatibility with test script
        metrics['iso_total_reward'] = self.metrics.total_iso_reward
        
        return metrics

    def _get_detailed_info(self):
        """Generate a comprehensive info dictionary for monitoring and evaluation"""
        # Get basic info
        info = self._get_info()
        
        # Calculate time_step for energy conversion
        time_step = self.time_step_duration / self.env_config['time']['minutes_per_day']
        
        # Add more detailed metrics for both agents
        info.update({
            # ISO specific info
            'iso_buy_price': self.iso_buy_price,
            'iso_sell_price': self.iso_sell_price,
            'predicted_demand': self.predicted_demand,
            'realized_demand': self.actual_demand,
            'net_demand': self.actual_demand,  # includes PCS contribution
            'dispatch': self.dispatch,  # Use tracked dispatch value instead of predicted_demand
            'shortfall': max(0.0, self.actual_demand - self.dispatch),
            'dispatch_cost': self.dispatch * self.dispatch_price,
            'reserve_cost': max(0.0, self.actual_demand - self.dispatch) * self.reserve_price,
            'price_spread': self.iso_sell_price - self.iso_buy_price,
            'iso_action': self.metrics.last_iso_action if hasattr(self.metrics, 'last_iso_action') else None,
            
            # PCS specific info
            'battery_level': self.battery_level,
            'battery_action': self.battery_manager.get_last_action() if hasattr(self.battery_manager, 'get_last_action') else 0.0,
            'net_exchange': self.pcs_demand * time_step,  # convert power to energy
            'pcs_exchange_cost': self.metrics.pcs_metrics['costs'][-1] if self.metrics.pcs_metrics['costs'] else 0.0,
            'pcs_action': self.metrics.last_pcs_action if hasattr(self.metrics, 'last_pcs_action') else None,
            
            # Shared metrics
            'time': self.current_time,
            'step': self.count,
            'terminated': self.terminated,
            'truncated': self.truncated,
            'episode_iso_reward': self.metrics.total_iso_reward,
            'episode_pcs_reward': self.metrics.total_pcs_reward
        })
        
        return info

    def get_metrics(self):
        """Get comprehensive metrics for both agents"""
        return self.metrics.get_metrics()

    def get_iso_observation_space(self):
        """Get ISO observation space"""
        return self.iso_observation_space
        
    def get_pcs_observation_space(self):
        """Get PCS observation space"""
        return self.pcs_observation_space
        
    def get_iso_action_space(self):
        """Get ISO action space"""
        return self.iso_action_space
        
    def get_pcs_action_space(self):
        """Get PCS action space"""
        return self.pcs_action_space

    def _get_obs(self):
        """
        Get observations for both ISO and PCS agents.
        
        Returns:
            tuple: (iso_obs, pcs_obs) - observations for both agents
        """
        iso_obs = self._get_iso_observation()
        pcs_obs = self._get_pcs_observation()
        
        return [iso_obs, pcs_obs]
