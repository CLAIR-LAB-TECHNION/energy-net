"""
PCSUnit Controller Module

This module implements the controller for Power Consumption and Storage Units (PCSUnit).
It manages the battery charging/discharging strategies in response to electricity prices
from the ISO and coordinates the various components involved in PCS operations.

Key responsibilities:
1. Interfacing with the PCSUnit component to manage battery operations
2. Processing agent actions and translating them to battery commands
3. Calculating energy production, consumption, and grid exchange
4. Managing state transitions and reward calculations
5. Providing standardized observations for the RL agent

The controller serves as the bridge between the reinforcement learning agent
and the physical battery simulation, enabling intelligent energy management
that can respond to market conditions.
"""

from energy_net.components.grid_entity import GridEntity
from typing import Optional, Tuple, Dict, Any, Union, Callable
import numpy as np
import os
from stable_baselines3 import PPO

from gymnasium import spaces
import yaml
import logging

from energy_net.components.pcsunit import PCSUnit
from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.dynamics.energy_dynamcis import ModelBasedDynamics
from energy_net.dynamics.production_dynamics.deterministic_production import DeterministicProduction
from energy_net.dynamics.consumption_dynamics.deterministic_consumption import DeterministicConsumption
from energy_net.dynamics.storage_dynamics.deterministic_battery import DeterministicBattery
from energy_net.dynamics.energy_dynamcis import DataDrivenDynamics
from energy_net.utils.iso_factory import iso_factory
from energy_net.utils.logger import setup_logger  

from energy_net.market.iso.demand_patterns import DemandPattern, calculate_demand  
from energy_net.market.iso.cost_types import CostType, calculate_costs
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO  

from energy_net.rewards.base_reward import BaseReward
from energy_net.rewards.cost_reward import CostReward

# Import controller components
from energy_net.controllers.pcs.metrics_handler import PCSMetricsHandler
from energy_net.controllers.pcs.battery_manager import BatteryManager
from energy_net.controllers.pcs.market_interface import MarketInterface


class PCSUnitController:
    """
    Power Consumption & Storage Unit Controller
    
    Manages a PCS unit's interaction with the power grid by controlling:
    1. Battery charging/discharging
    2. Energy production (optional)
    3. Energy consumption (optional)
    
    The controller handles:
    - Battery state management
    - Price-based decision making
    - Energy exchange with grid
    - Production/consumption coordination
    
    Actions:
        Type: Box
            - If multi_action=False:
                Charging/Discharging Power: continuous scalar
            - If multi_action=True:
                [Charging/Discharging Power, Consumption Action, Production Action]

    Observation:
        Type: Box(4)
            Energy storage level (MWh): [0, ENERGY_MAX]
            Time (fraction of day): [0, 1]
            ISO Buy Price ($/MWh): [0, inf]
            ISO Sell Price ($/MWh): [0, inf]
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
        trained_iso_model_path: Optional[str] = None  
    ):
        """
        Constructs an instance of PCSunitEnv.

        Args:
            render_mode: Optional rendering mode.
            env_config_path: Path to the environment YAML configuration file.
            iso_config_path: Path to the ISO YAML configuration file.
            pcs_unit_config_path: Path to the PCSUnit YAML configuration file.
            log_file: Path to the log file for environment logging.
            reward_type: Type of reward function to use.
        """
        super().__init__()  # Initialize the parent class

        # Store new parameters
        self.cost_type = cost_type
        self.demand_pattern = demand_pattern
        
        # Set up logger
        self.logger = setup_logger('PCSUnitController', log_file)
        self.logger.info(f"Using demand pattern: {demand_pattern.value}")
        self.logger.info(f"Using cost type: {cost_type.value}")

        # Load configurations
        self.env_config: Dict[str, Any] = self.load_config(env_config_path)
        self.iso_config: Dict[str, Any] = self.load_config(iso_config_path)
        self.pcs_unit_config: Dict[str, Any] = self.load_config(pcs_unit_config_path)
        
        # Initialize PCSUnit with dynamics and configuration
        self.PCSUnit: PCSUnit = PCSUnit(
            config=self.pcs_unit_config,
            log_file=log_file
        )
        self.logger.info("Initialized PCSUnit with all components.")

        # Define observation and action spaces
        energy_config: Dict[str, Any] = self.pcs_unit_config['battery']['model_parameters']
        obs_config = self.pcs_unit_config.get('observation_space', {})

        # Get battery level bounds from battery config if specified
        battery_level_config = obs_config.get('battery_level', {})
        battery_min = energy_config['min'] if battery_level_config.get('min') == "from_battery_config" else battery_level_config.get('min', energy_config['min'])
        battery_max = energy_config['max'] if battery_level_config.get('max') == "from_battery_config" else battery_level_config.get('max', energy_config['max'])

        # Get other observation space bounds from config
        time_config = obs_config.get('time', {})
        buy_price_config = obs_config.get('iso_buy_price', {})
        sell_price_config = obs_config.get('iso_sell_price', {})

        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([
                battery_min,
                time_config.get('min', 0.0),
                buy_price_config.get('min', 0.0),
                sell_price_config.get('min', 0.0)
            ], dtype=np.float32),
            high=np.array([
                battery_max,
                time_config.get('max', 1.0),
                buy_price_config.get('max', 100.0),
                sell_price_config.get('max', 100.0)
            ], dtype=np.float32),
            dtype=np.float32
        )
        self.logger.info(f"Defined observation space: low={self.observation_space.low}, high={self.observation_space.high}")

        # Define Action Space
        self.multi_action: bool = self.pcs_unit_config.get('action', {}).get('multi_action', False)
        self.production_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('production_action', {}).get('enabled', False)
        self.consumption_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('consumption_action', {}).get('enabled', False)

        self.action_space: spaces.Box = spaces.Box(
            low=np.array([
                -energy_config['discharge_rate_max']
            ], dtype=np.float32),
            high=np.array([
                energy_config['charge_rate_max']
            ], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )
        self.logger.info(f"Defined action space: low={-energy_config['discharge_rate_max']}, high={energy_config['charge_rate_max']}")

        # Initialize state variables
        self.time = 0.0
        
        # Internal State
        self.init: bool = False
        self.rng = np.random.default_rng()
        self.count: int = 0        # Step counter
        self.terminated: bool = False
        self.truncated: bool = False

        # Initialize component modules
        
        # Initialize battery manager with reference to PCSUnit for direct access
        self.battery_manager = BatteryManager(
            battery_config=self.pcs_unit_config['battery']['model_parameters'],
            pcsunit=self.PCSUnit,  # Pass PCSUnit reference to use direct battery access
            logger=self.logger
        )
        self.battery_level = self.battery_manager.get_level()  # Get initial battery level from manager
        self.logger.info(f"Initialized battery manager with level: {self.battery_level}")
        
        # Initialize market interface
        self.market_interface = MarketInterface(
            env_config=self.env_config,
            iso_config=self.iso_config,
            pcs_config=self.pcs_unit_config,
            logger=self.logger
        )
        self.logger.info("Initialized market interface")

        # Initialize timing parameters
        self.time_steps_per_day_ratio = self.env_config['time']['time_steps_per_day_ratio']
        self.time_step_duration = self.env_config['time']['step_duration']
        self.max_steps_per_episode = self.env_config['time']['max_steps_per_episode']

        # Initialize the Reward Function
        self.logger.info(f"Setting up reward function: {reward_type}")
        self.reward: BaseReward = self.initialize_reward(reward_type)
        
        # Initialize the Metrics Handler
        self.metrics_handler = PCSMetricsHandler(
            config=self.pcs_unit_config,
            reward_function=self.reward,
            logger=self.logger
        )
        self.logger.info("Initialized metrics handler")
                
        # Load trained ISO model if provided
        if trained_iso_model_path:
            try:
                trained_iso_agent = PPO.load(trained_iso_model_path)
                self.market_interface.set_trained_iso_agent(trained_iso_agent)
                self.logger.info(f"Loaded ISO model: {trained_iso_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load ISO model: {e}")

        self.logger.info("PCSunitEnv initialization complete.")
                
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads a YAML configuration file.

        Args:
            config_path (str): Path to the YAML config file.

        Returns:
            Dict[str, Any]: Configuration parameters.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found at {config_path}")
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as file:
            config: Dict[str, Any] = yaml.safe_load(file)
            self.logger.debug(f"Loaded configuration from {config_path}: {config}")

        return config        

    def initialize_reward(self, reward_type: str) -> BaseReward:
        """
        Initializes the reward function based on the specified type.

        Args:
            reward_type (str): Type of reward ('cost').

        Returns:
            BaseReward: An instance of a reward class.
        
        Raises:
            ValueError: If an unsupported reward_type is provided.
        """
        if reward_type == 'cost':
            return CostReward()
        else:
            self.logger.error(f"Unsupported reward type: {reward_type}")
            raise ValueError(f"Unsupported reward type: {reward_type}")

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
        self.logger.info("Resetting environment.")

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.logger.debug(f"Random number generator seeded with: {seed}")
        else:
            self.rng = np.random.default_rng()
            self.logger.debug("Random number generator initialized without seed.")

        # If this is not the first reset and we have metrics, end the previous episode
        if self.init and hasattr(self, 'metrics_handler'):
            self.metrics_handler.end_episode()

        # Reset PCSUnit and ISO
        self.PCSUnit.reset()
        self.logger.debug("PCSUnit has been reset.")

        # Reset market interface
        self.market_interface.reset()
        self.logger.debug("Market interface reset")
        
        # Reset step counter and state
        self.count = 0
        self.time = 0.0
        
        # Reset metrics handler and reward function
        self.metrics_handler.reset()
        self.reward.reset()  # Reset reward function state
        self.logger.debug("Metrics handler and reward function reset")
        
        # Initialize time step
        time = self.time
        
        # Get initial market state
        predicted_demand = self.calculate_predicted_demand(time)
        self.market_interface.update_market_prices(time, predicted_demand, 0.0)
        realized_demand = self.market_interface.update_realized_demand()
        market_state = {
            'iso_buy_price': self.market_interface.iso_buy_price,
            'iso_sell_price': self.market_interface.iso_sell_price,
            'predicted_demand': predicted_demand,
            'realized_demand': realized_demand,
            'pcs_demand': 0.0
        }
        
        # Get initial production and consumption
        production = self.PCSUnit.get_self_production()
        consumption = self.PCSUnit.get_self_consumption()
        
        # Build state dictionary
        battery_state = self.battery_manager.get_state()
        self.state = {
            **battery_state,
            'time': time,
            'current_time': time * self.env_config['time']['minutes_per_day'],  # Convert to minutes
            'production': production,
            'consumption': consumption,
            'battery_action': 0.0,
            'net_exchange': 0.0,
            **market_state  # Include all market state metrics
        }
        
        # Create initial observation
        observation = np.array([
            self.battery_level,
            time,
            self.market_interface.iso_buy_price,
            self.market_interface.iso_sell_price
        ], dtype=np.float32)

        self.logger.info(f"Environment reset complete. Initial observation: {observation}")
        
        # Set initialization flag
        self.init = True

        return observation, self.get_info()

    def validate_battery_action(self, action: float) -> float:
        """
        Simple validation for battery action - only prevents attempting to discharge more energy
        than is available in the battery. The battery component's internal update method will 
        handle other constraints like charge/discharge rates and capacity limits.
        
        Args:
            action: Proposed battery action (+ for charging, - for discharging)
            
        Returns:
            Validated action within allowable bounds
        """
        # Get current battery state
        current_level = self.PCSUnit.battery.get_state()
        
        # Only validate discharge actions - charging validation happens in the battery component
        if action < 0:  # Discharging
            # Simple physics constraint: can't discharge more than available
            max_possible_discharge = -current_level
            
            if action < max_possible_discharge:
                self.logger.warning(
                    f"Physics constraint: Attempted to discharge {abs(action):.4f} MWh "
                    f"with only {current_level:.4f} MWh available. "
                    f"Limiting to {abs(max_possible_discharge):.4f} MWh."
                )
                return max_possible_discharge
        
        # For all other cases (charging or valid discharge), let the battery component handle it
        return action
    
    def get_battery_state(self) -> Dict[str, float]:
        """
        Get current battery state information.
        
        Retrieves comprehensive information about the current state of the battery,
        including current level, physical constraints, and rate limits. This information
        is essential for understanding the battery's operational context and constraints.
        
        Returns:
            Dictionary containing:
                - battery_level: Current state of charge (MWh)
                - energy_change: Most recent change in energy level (MWh)
                - battery_min: Minimum allowed battery level (MWh)
                - battery_max: Maximum battery capacity (MWh)
                - charge_rate_max: Maximum charging rate (MWh/step)
                - discharge_rate_max: Maximum discharging rate (MWh/step)
        """
        return {
            'battery_level': self.battery_level,
            'energy_change': self.PCSUnit.battery.energy_change if hasattr(self.PCSUnit.battery, 'energy_change') else 0.0,
            'battery_min': self.PCSUnit.battery.energy_min,
            'battery_max': self.PCSUnit.battery.energy_max,
            'charge_rate_max': self.PCSUnit.battery.charge_rate_max,
            'discharge_rate_max': self.PCSUnit.battery.discharge_rate_max
        }
    
    def step(self, action: Union[float, np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one time step of the PCS unit environment.
        
        This is the core simulation method that advances the state by one time step.
        The detailed process flow is:
        
        1. Update time and get current ISO prices based on the time and demand
        2. Process and validate the battery action (ensure it respects physical constraints)
        3. Update PCS unit state including battery level, production, and consumption
        4. Calculate net exchange with the grid (buying or selling electricity)
        5. Determine financial impacts (revenue, costs) based on market prices
        6. Calculate grid impacts (demand, shortfall, dispatch costs)
        7. Compute reward based on the updated state
        8. Build observation vector for the agent
        
        Args:
            action: Battery charging/discharging power. Can be:
                   - float: Simple battery action (positive = charging, negative = discharging)
                   - np.ndarray: For single action [battery] or multi-action [battery, consumption, production]
                   - int: Will be converted to float for battery action
                   
        Returns:
            observation: Current state [battery_level, time, buy_price, sell_price]
            reward: Cost-based reward for this step (typically negative, representing costs)
            done: Whether episode is complete (always False for this environment)
            truncated: Whether episode was truncated due to reaching max steps
            info: Dictionary containing detailed metrics including:
                  - Battery state (level, energy_change)
                  - Market position (net_exchange, revenue)
                  - Grid impact (shortfall, reserve_cost)
                  - Production and consumption values
        
        Raises:
            ValueError: If the action doesn't match the expected format or dimensions
            TypeError: If the action type is invalid
        """
        assert self.init, "Environment must be reset before stepping."
        
        # 1. Update time and state
        self.count += 1
        self.time = (self.count * self.time_step_duration) / self.env_config['time']['minutes_per_day']
        self.logger.debug(f"Time updated to {self.time:.3f} (day fraction)")

        # Calculate predicted demand for this timestep
        predicted_demand = self.calculate_predicted_demand(self.time)
        
        # Get PCS demand from previous step (or 0.0 if first step)
        pcs_demand = self.state.get('pcs_demand', 0.0)
        
        # 2. Update market prices based on time and demand
        self.market_interface.update_market_prices(self.time, predicted_demand, pcs_demand)
        
        # Update realized demand with noise
        realized_demand = self.market_interface.update_realized_demand()

        # 3. Process action: Convert to proper format
        self.logger.debug(f"Processing PCS action: {action}")
        if isinstance(action, np.ndarray):
            if self.multi_action and action.shape != (3,):
                raise ValueError(f"Action array must have shape (3,) for multi-action mode")
            elif not self.multi_action and action.shape != (1,):
                raise ValueError(f"Action array must have shape (1,) for single-action mode")
            
            if not self.action_space.contains(action):
                self.logger.warning(f"Action {action} outside bounds, clipping to valid range")
                action = np.clip(action, self.action_space.low, self.action_space.high)
                
            if self.multi_action:
                battery_action, consumption_action, production_action = action
            else:
                battery_action = action.item()
                consumption_action = None
                production_action = None
                
        elif isinstance(action, float) or isinstance(action, int):
            if self.multi_action:
                raise TypeError("Expected array action for multi-action mode")
            battery_action = float(action)  # Ensure it's a float
            consumption_action = None
            production_action = None
        else:
            raise TypeError(f"Invalid action type: {type(action)}")

        # Store original action for reference
        original_action = battery_action
        
        # 4. Validate battery action (using simplified approach following the old controller)
        # Note: We only validate the basic physics constraint here (can't discharge more than available)
        # The battery component's internal update method will handle the remaining constraints
        # like charge/discharge rates and capacity limits
        previous_battery_level = self.PCSUnit.battery.get_state()
        validated_battery_action = self.validate_battery_action(battery_action)
        
        # 5. Update the PCSUnit with validated action
        self.PCSUnit.update(time=self.time, battery_action=validated_battery_action)
        
        # 6. Get updated battery state
        self.battery_level = self.PCSUnit.battery.get_state()
        energy_change = self.PCSUnit.battery.energy_change if hasattr(self.PCSUnit.battery, 'energy_change') else (self.battery_level - previous_battery_level)
        
        # Get current production and consumption
        production = self.PCSUnit.get_self_production()
        consumption = self.PCSUnit.get_self_consumption()
        
        # 7. Calculate net exchange using the validated battery action
        # Store the old calculation method for comparison
        if validated_battery_action > 0:  # Charging
            old_net_exchange = (consumption + validated_battery_action) - production
        elif validated_battery_action < 0:  # Discharging
            old_net_exchange = consumption - (production + abs(validated_battery_action))
        else:
            old_net_exchange = consumption - production

        # New, clearer calculation with same logic but explicit components
        battery_effect = validated_battery_action if validated_battery_action > 0 else -abs(validated_battery_action)
        consumption_total = consumption + max(0, battery_effect)  # Add charging to consumption
        production_total = production + max(0, -battery_effect)   # Add discharging to production
        net_exchange = consumption_total - production_total
        
        # Verify calculations match for safety
        if abs(net_exchange - old_net_exchange) > 1e-6:
            self.logger.warning(
                f"Net exchange calculation mismatch: old={old_net_exchange:.6f}, new={net_exchange:.6f}, "
                f"diff={abs(net_exchange - old_net_exchange):.6f}"
            )
            
        # Save for the next step
        self.pcs_demand = net_exchange
        
        # 8. Create market position dictionary
        market_position = {
            'net_exchange': net_exchange,
            'consumption_total': consumption_total,
            'production_total': production_total,
            'predicted_demand': predicted_demand,
            'realized_demand': realized_demand,
            'iso_buy_price': self.market_interface.iso_buy_price,
            'iso_sell_price': self.market_interface.iso_sell_price
        }
        
        # Calculate financial impact with clearer diagnostics
        if net_exchange > 0:  # Buying from grid
            revenue = -net_exchange * self.market_interface.iso_sell_price  # Negative revenue (cost)
            self.logger.debug(
                f"Buying {net_exchange:.4f} MWh @ ${self.market_interface.iso_sell_price:.2f}/MWh = ${revenue:.2f}"
            )
        else:  # Selling to grid
            revenue = -net_exchange * self.market_interface.iso_buy_price  # Positive revenue
            self.logger.debug(
                f"Selling {abs(net_exchange):.4f} MWh @ ${self.market_interface.iso_buy_price:.2f}/MWh = ${revenue:.2f}"
            )
            
        market_position['revenue'] = revenue
        
        # Calculate grid impact
        net_demand = realized_demand + net_exchange
        market_position['net_demand'] = net_demand
        
        # Calculate dispatch cost and reserve costs
        dispatch = predicted_demand
        dispatch_cost = self.market_interface.dispatch_price * dispatch
        shortfall = max(0.0, net_demand - dispatch)
        reserve_cost = self.market_interface.reserve_price * shortfall
        
        market_position['dispatch'] = dispatch
        market_position['dispatch_cost'] = dispatch_cost
        market_position['shortfall'] = shortfall
        market_position['reserve_cost'] = reserve_cost
        
        # 9. Get battery state directly
        battery_state = self.get_battery_state()
        
        # 10. Update state with current metrics
        self.state = {
            **battery_state,  # Include all battery state metrics 
            'time': self.time,
            'current_time': self.time * self.env_config['time']['minutes_per_day'],  # Convert to minutes
            'production': production,
            'consumption': consumption,
            'battery_action': validated_battery_action,  # Use validated action
            'original_action': original_action,  # Store original action for reference
            'energy_change': energy_change,  # Store actual energy change
            **market_position  # Include market position metrics
        }
        
        # 11. Calculate reward based on state
        reward = self.reward.compute_reward(info=self.state)
        
        # Add reward to state for logging
        self.state['reward'] = reward
        self.state['step'] = self.count
        
        # Create current observation
        observation = np.array([
            self.battery_level,
            self.time,
            self.market_interface.iso_buy_price,
            self.market_interface.iso_sell_price
        ], dtype=np.float32)

        # Check for episode termination
        self.terminated = False  # PCS environment doesn't have a specific termination condition
        self.truncated = (self.count >= self.max_steps_per_episode)
        
        # Add debugging for truncation condition
        if self.truncated:
            self.logger.info(f"Episode truncated at step {self.count}/{self.max_steps_per_episode}")
            # Get episode summary from metrics handler
            summary = self.metrics_handler.get_metrics_summary()  # Just get summary without ending episode
            self.logger.info(f"Episode summary: {summary}")
            
            # Store the summary in state for the final info dict
            self.state['episode_summary'] = summary
        
        # Simple but comprehensive debugging
        self.logger.debug(
            f"Step {self.count}: Action {original_action:.4f}→{validated_battery_action:.4f}, "
            f"Battery {previous_battery_level:.2f}→{self.battery_level:.2f}, "
            f"Net Exchange {net_exchange:.4f}, Reward {reward:.4f}"
        )
        
        return observation, reward, self.terminated, self.truncated, self.get_info()

    def get_info(self) -> Dict[str, float]:
        """
        Provides additional information about the environment's state.

        Returns:
            Dict[str, float]: Dictionary containing environment summary metrics and state values.
        """
        # Get the metrics summary
        metrics = self.metrics_handler.get_metrics_summary()
        
        # Add the current state values to the info
        info = {
            **metrics,  # Include the metrics summary
            **self.state  # Include all the state values (production, consumption, etc.)
        }
        
        # If we have an episode summary (from truncation), include it
        if 'episode_summary' in self.state:
            info['episode'] = self.state['episode_summary']
        
        return info
 
    def close(self):
        """
        Cleanup method. Closes loggers and releases resources.
        """
        self.logger.info("Closing environment.")
        
        # Get final metrics summary
        metrics_summary = self.metrics_handler.get_metrics_summary()
        self.logger.info(f"Final metrics summary: {metrics_summary}")

        logger_names = ['PCSunitEnv', 'Battery', 'ProductionUnit', 'ConsumptionUnit', 'PCSUnit'] 
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        self.logger.info("Environment closed successfully.")
            
    def calculate_predicted_demand(self, time: float) -> float:
        """
        Calculate predicted demand using selected pattern.
        
        This method determines the predicted electricity demand for a given time using
        the configured demand pattern (e.g., SINUSOIDAL, RANDOM). The predicted demand
        is used by the ISO to set prices and prepare dispatch.
        
        Args:
            time: Current time as a fraction of day (0.0 to 1.0)
            
        Returns:
            float: Predicted demand value (MWh) for the given time
        """
        return calculate_demand(
            time=time,
            pattern=self.demand_pattern,
            config=self.env_config['predicted_demand']
        )

