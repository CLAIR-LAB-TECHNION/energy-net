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
from energy_net.dynamics.storage_dynamics.deterministic_battery import DeterministicBattery  # Import the new dynamics
from energy_net.dynamics.energy_dynamcis import DataDrivenDynamics
from energy_net.utils.iso_factory import iso_factory
from energy_net.utils.logger import setup_logger  


# Import all reward classes
from energy_net.rewards.base_reward import BaseReward
from energy_net.rewards.cost_reward import CostReward



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

        # Set up logger
        self.logger = setup_logger('PCSunitEnv', log_file)
        self.logger.info("Initializing PCSunitEnv.")

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

        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([
                energy_config['min'],
                0.0,
                0.0,
                0.0
            ], dtype=np.float32),
            high=np.array([
                energy_config['max'],
                1.0,
                100.0,
                100.0
            ], dtype=np.float32),
            dtype=np.float32
        )
        self.logger.info(f"Defined observation space: low={self.observation_space.low}, high={self.observation_space.high}")

        # Define Action Space
        self.multi_action: bool = self.pcs_unit_config.get('action', {}).get('multi_action', False)
        self.production_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('production_action', {}).get('enabled', False)
        self.consumption_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('consumption_action', {}).get('enabled', False)
        pricing_config = self.iso_config.get('pricing', {})
        price_params = pricing_config.get('parameters', {})
        self.min_price = price_params.get('min_price', pricing_config.get('default_sell_price', 1.0))
        self.max_price = price_params.get('max_price', pricing_config.get('default_buy_price', 10.0))

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
        self.predicted_demand = 0.0
        self.pcs_demand = 0.0  
        # Tracking variables for ISO state
        self.iso_sell_price = self.min_price  
        self.iso_buy_price = self.min_price  
        # Internal State
        self.init: bool = False
        self.rng = np.random.default_rng()
        self.avg_price: float = 0.0
        self.battery_level: float = energy_config['init']
        self.reward_type: int = 0
        self.count: int = 0        # Step counter
        self.terminated: bool = False
        self.truncated: bool = False

        # reference to trained ISO agent
        self.trained_iso_agent = None

        uncertainty_config = self.env_config.get('demand_uncertainty', {})
        self.sigma = uncertainty_config.get('sigma', 0.0)
        self.reserve_price = self.env_config.get('reserve_price', 0.0)
        self.dispatch_price = self.env_config.get('dispatch_price', 0.0)

        # Extract other configurations if necessary
        self.pricing_eta = self.env_config['pricing']['eta']
        self.time_steps_per_day_ratio = self.env_config['time']['time_steps_per_day_ratio']
        self.time_step_duration = self.env_config['time']['step_duration']
        self.max_steps_per_episode = self.env_config['time']['max_steps_per_episode']

        # Initialize the Reward Function
        self.logger.info(f"Setting up reward function: {reward_type}")
        self.reward: BaseReward = self.initialize_reward(reward_type)
                
        # Load trained ISO model if provided
        if trained_iso_model_path:
            try:
                trained_iso_agent = PPO.load(trained_iso_model_path)
                self.set_trained_iso_agent(trained_iso_agent)
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

        # Reset PCSUnit and ISO
        self.PCSUnit.reset()
        self.logger.debug("PCSUnit has been reset.")

        # Reset internal state
        energy_config = self.pcs_unit_config['battery']['model_parameters']
        self.avg_price = 0.0
        self.energy_lvl = energy_config['init']
        #self.battery_level = self.rng.uniform(low=energy_config['min'], high=energy_config['max'])
        
        # Pass the random initial energy level to PCSUnit
        self.PCSUnit.reset(initial_battery_level=self.battery_level)  # Changed: pass initial level

        self.reward_type = 0  # Default reward type

        # Handle options
        if options and 'reward' in options:
            if options.get('reward') == 1:
                self.reward_type = 1
                self.logger.debug("Reward type set to 1 based on options.")
            else:
                self.logger.debug(f"Reward type set to {self.reward_type} based on options.")
        else:
            self.logger.debug("No reward type option provided; using default.")

        # Reset step counter
        self.count = 0
        self.terminated = False
        self.truncated = False
        self.init = True

        # Initialize current time (fraction of day)
        self.time = 0.0
        time: float = (self.count * self.time_step_duration) / 1440  # 1440 minutes in a day
        self.logger.debug(f"Initial time set to {time} fraction of day.")


        self.predicted_demand = self.calculate_predicted_demand(self.time)
        self.pcs_demand = 0.0
        noise = np.random.normal(0, self.sigma)
        self.realized_demand = float(self.predicted_demand + noise)


        # Update PCSUnit with current time and no action
        self.PCSUnit.update(time=time, battery_action=0.0)
        self.logger.debug("PCSUnit updated with initial time and no action.")

        # Fetch self-production and self-consumption
        production: float = self.PCSUnit.get_self_production()
        consumption: float = self.PCSUnit.get_self_consumption()
        self.logger.debug(f"Initial pcs-production: {production}, pcs-consumption: {consumption}")
        
        if self.trained_iso_agent is not None:
            try:
                prices = self.trained_iso_agent.predict(self.translate_to_iso_observation(), deterministic=True)[0]
                self.iso_sell_price, self.iso_buy_price = prices
                self.logger.info(f"Updated ISO agent prices on reset: Sell {self.iso_sell_price:.2f}, Buy {self.iso_buy_price:.2f}")
            except Exception as e:
                self.logger.error(f"Failed to update ISO prices on reset: {e}")

        observation: np.ndarray = np.array([
            self.battery_level,
            time,
            self.iso_buy_price,   
            self.iso_sell_price   
        ], dtype=np.float32)
        self.logger.debug(f"Initial observation: {observation}")

        info = {"status": "reset"}
        self.logger.debug(f"Initial info: {info}")

        return (observation, info)

    def step(self, action: Union[float, np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one time step of the PCS unit.
        
        Process flow:
        1. Update time and get current ISO prices
        2. Validate and process battery action
        3. Update PCS unit state
        4. Calculate costs and rewards
        
        Args:
            action: Battery charging/discharging power
                   Positive = charging, Negative = discharging
                   
        Returns:
            observation: Current state [battery_level, time, buy_price, sell_price]
            reward: Cost-based reward for this step
            done: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional metrics and state information
        """
        assert self.init, "Environment must be reset before stepping."
        
        # 1. Update time and predicted demand

        # Update time and state
        self.count += 1
        self.time = (self.count * self.time_step_duration) / self.env_config['time']['minutes_per_day']
        self.logger.debug(f"Time updated to {self.time:.3f} (day fraction)")

        # Update predicated and realized demand
        self.predicted_demand = self.calculate_predicted_demand(self.time)
        noise = np.random.normal(0, self.sigma)
        self.realized_demand = float(self.predicted_demand + noise)
        self.logger.debug(f"Predicted demand: {self.predicted_demand:.2f} MWh")

        # 2. Process ISO action
        if self.trained_iso_agent is not None:
            iso_obs = self.translate_to_iso_observation()
            self.logger.debug(f"Sending observation to ISO: {iso_obs}")
            try:
                prices = self.trained_iso_agent.predict(iso_obs, deterministic=True)[0]
                self.logger.debug(f"Raw ISO prediction: {prices}")
                
                self.iso_sell_price, self.iso_buy_price = prices
                if self.iso_sell_price == 0 and self.iso_buy_price == 0:
                    self.logger.warning("ISO agent returned zero prices - this might indicate an issue")
                
                self.logger.info(
                    f"Using ISO agent prices:\n"
                    f"  - ISO Sell Price: {self.iso_sell_price:.2f} $/MWh\n"
                    f"  - ISO Buy Price: {self.iso_buy_price:.2f} $/MWh"
                )
            except Exception as e:
                self.logger.error(f"Failed to get prices from ISO agent: {e}")
                self.iso_sell_price = self.iso_config.get('pricing', {}).get('default_buy_price', 50.0)
                self.iso_buy_price = self.iso_config.get('pricing', {}).get('default_sell_price', 45.0)
                self.logger.warning(
                    f"Falling back to default prices:\n"
                    f"  - ISO Sell Price: {self.iso_sell_price:.2f} $/MWh\n"
                    f"  - ISO Buy Price: {self.iso_buy_price:.2f} $/MWh"
                )
        else:
            self.iso_sell_price = self.iso_buy_price = 10.0
            if (10<=self.count<=30):
                self.iso_sell_price = 4.0
            self.iso_buy_price = 0.8 * self.iso_sell_price
            self.logger.info(
                f"Using default prices (no ISO agent):\n"
                f"  - ISO Sell Price: {self.iso_sell_price:.2f} $/MWh\n"
                f"  - ISO Buy Price: {self.iso_buy_price:.2f} $/MWh"
            )


        # 3. Get PCS response
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
                
        elif isinstance(action, float):
            if self.multi_action:
                raise TypeError("Expected array action for multi-action mode")
            battery_action = action
            consumption_action = None
            production_action = None
        else:
            raise TypeError(f"Invalid action type: {type(action)}")

                        

        # Validate battery action based on current state
        current_battery_level = self.PCSUnit.battery.get_state()
        
        # If trying to discharge (negative action) with insufficient battery level
        if battery_action < 0:  # Discharging
            max_possible_discharge = -current_battery_level  # Maximum amount we can discharge
            if battery_action < max_possible_discharge:
                self.logger.warning(
                    f"Attempted to discharge {abs(battery_action):.2f} MWh with only {current_battery_level:.2f} MWh available. "
                    "Limiting discharge to available energy."
                )
                battery_action = max_possible_discharge
        
        if self.multi_action:
            self.PCSUnit.update(
                time=self.time,
                battery_action=battery_action,
                consumption_action=consumption_action,
                production_action=production_action
            )
        else:
            self.PCSUnit.update(
                time=self.time,
                battery_action=battery_action
            )

        # Get updated production and consumption
        production = self.PCSUnit.get_self_production()
        consumption = self.PCSUnit.get_self_consumption()
        
        # 4. Calculate grid state and costs
        # Calculate net exchange based on battery action
        if battery_action > 0:  # Charging
            net_exchange = (consumption + battery_action) - production
        elif battery_action < 0:  # Discharging
            net_exchange = consumption - (production + abs(battery_action))
        else:
            net_exchange = consumption - production
            

        # Update energy level and tracking variables
        self.battery_level = self.PCSUnit.battery.get_state()
        self.pcs_demand = net_exchange
        net_demand = self.realized_demand + net_exchange
        self.logger.debug(f"Net demand: {net_demand:.2f} MWh")
        # Calculate dispatch and costs (matching ISO calculations)
        dispatch = self.predicted_demand
        if (dispatch<=100):
            dispatch_cost = 5.0 * dispatch
        elif (100<dispatch<170):
            dispatch_cost = 7.0 * dispatch
        elif (170<=dispatch):
            dispatch_cost = 8.0 * dispatch
        else:
            dispatch_cost = 0.0
        #dispatch_cost = self.dispatch_price * dispatch
        shortfall = max(0.0, net_demand - dispatch)
        reserve_cost = self.reserve_price * shortfall
    
        # Update info dictionary with all cost components
        info = {
            'iso_sell_price': self.iso_sell_price,
            'iso_buy_price': self.iso_buy_price,
            'predicted_demand': self.predicted_demand,
            'realized_demand': self.realized_demand,
            'production': production,
            'consumption': consumption,
            'battery_level': self.battery_level,
            'net_exchange': net_exchange,
            'pcs_demand': self.pcs_demand,
            'dispatch_cost': dispatch_cost,
            'shortfall': shortfall,
            'reserve_cost': reserve_cost,
            'dispatch': dispatch,
            'net_demand': net_demand
        }

        # Add detailed logging for battery actions
        self.logger.info(
            f"PCS State Step {self.count}:\n"
            f"  Time: {self.time:.3f}\n"
            f"  Battery Level: {self.battery_level:.2f} MWh\n"
            f"  Battery Action: {battery_action:.2f} MWh\n"
            f"  Production: {production:.2f} MWh\n"
            f"  Consumption: {consumption:.2f} MWh\n"
            f"  Net Exchange: {net_exchange:.2f} MWh"
        )

        # Add financial logging
        self.logger.info(
            f"Financial Metrics:\n"
            f"  ISO Buy Price: ${self.iso_buy_price:.2f}/MWh\n"
            f"  ISO Sell Price: ${self.iso_sell_price:.2f}/MWh\n"
            f"  Energy Cost: ${abs(net_exchange * (self.iso_buy_price if net_exchange > 0 else self.iso_sell_price)):.2f}"
        )

        # 5. Compute reward
        reward = self.reward.compute_reward(info)
        self.logger.info(f"Step reward: {reward:.2f}")

        # 6. Create next observation
        
        observation = np.array([
            self.battery_level,
            self.time,
            self.iso_buy_price,
            self.iso_sell_price
        ], dtype=np.float32)

        # Check if episode is done
        done = self.count >= self.max_steps_per_episode
        if done:
            self.logger.info("Episode complete")
        
        return observation, float(reward), done, False, info


    def get_info(self) -> Dict[str, float]:
        """
        Provides additional information about the environment's state.

        Returns:
            Dict[str, float]: Dictionary containing the running average price.
        """
        return {"running_avg": self.avg_price}
 
    def close(self):
        """
        Cleanup method. Closes loggers and releases resources.
        """
        self.logger.info("Closing environment.")

        logger_names = ['PCSunitEnv', 'Battery', 'ProductionUnit', 'ConsumptionUnit', 'PCSUnit'] 
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        self.logger.info("Environment closed successfully.")
            
    def calculate_predicted_demand(self, time: float) -> float:
        """Calculate base grid demand using cosine function"""
        demand_config = self.env_config['predicted_demand']
        interval = time * demand_config['interval_multiplier']
        predicted_demand = demand_config['base_load'] + demand_config['amplitude'] * np.cos(
            (interval + demand_config['phase_shift']) * np.pi / demand_config['period_divisor']
        )
        self.logger.debug(f"Calculated predicated demand for time {time}: {predicted_demand}")
        return predicted_demand

    def set_trained_iso_agent(self, iso_agent):
        """Set the trained ISO agent for price determination"""
        self.trained_iso_agent = iso_agent
    
        # Test that the agent works
        test_obs = self.translate_to_iso_observation()
        try:
            prices = self.trained_iso_agent.predict(test_obs, deterministic=True)[0]
            self.logger.info(f"ISO agent test successful - got prices: {prices}")
        except Exception as e:
            self.logger.error(f"ISO agent validation failed: {e}")
            self.trained_iso_agent = None  # Reset if validation fails
            raise e

    def translate_to_iso_observation(self) -> np.ndarray:
        """
        Converts current PCS state to ISO observation format.
        
        Creates observation vector containing:
        - Current time of day
        - Predicted grid demand
        - PCS net demand (consumption - production + battery)
        
        Returns:
            np.ndarray: Observation for ISO agent
        """
        iso_observation = np.array([
            self.time,
            self.predicted_demand,
            self.pcs_demand 
        ], dtype=np.float32)
        
        self.logger.debug(
            f"ISO Observation:\n"
            f"  Time: {self.time:.3f}\n"
            f"  Predicted Demand: {self.predicted_demand:.2f} MWh\n"
            f"  PCS Demand: {self.pcs_demand:.2f} MWh"
        )
        return iso_observation

