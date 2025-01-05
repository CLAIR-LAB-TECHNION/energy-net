from energy_net.components.grid_entity import GridEntity
from typing import Optional, Tuple, Dict, Any, Union, Callable
import numpy as np
import os
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
    Actions:
        Type: Box
            - If multi_action=False:
                Charging/Discharging Power: continuous scalar
            - If multi_action=True:
                [Charging/Discharging Power, Consumption Action, Production Action]

    Observation:
        Type: Box(4)
                                        Min                     Max
        Energy storage level (MWh)            0                       ENERGY_MAX
        Time (fraction of day)               0                       1
        Self Production (MWh)                0                       Inf
        Self Consumption (MWh)               0                       Inf
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',  # Path to the log file
        reward_type: str = 'cost',  # New parameter to specify the reward type
        trained_iso_model_path: Optional[str] = None  # Add this parameter
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

        # Initialize ISO using the factory - This section should be commented out or removed
        '''
        iso_type: str = self.iso_config.get('type', 'HourlyPricingISO')
        iso_parameters: Dict[str, Any] = self.iso_config.get('parameters', {})
        self.ISO = iso_factory(iso_type, iso_parameters)
        self.logger.info(f"Initialized ISO with type: {iso_type} and parameters: {iso_parameters}")
        '''
        
        # Initialize PCSUnit with dynamics and configuration
        self.PCSUnit: PCSUnit = PCSUnit(
            config=self.pcs_unit_config,
            log_file=log_file
        )
        
        self.logger.info("Initialized PCSUnit with all components.")

        # Define Action Space
        energy_config: Dict[str, Any] = self.pcs_unit_config['battery']['model_parameters']
        
        
        # Load action configurations
        self.multi_action: bool = self.pcs_unit_config.get('action', {}).get('multi_action', False)
        self.production_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('production_action', {}).get('enabled', False)
        self.consumption_action_enabled: bool = self.pcs_unit_config.get('action', {}).get('consumption_action', {}).get('enabled', False)

        
        self.action_space: spaces.Box = spaces.Box(
            low=np.array([-energy_config['discharge_rate_max']], dtype=np.float32),
            high=np.array([energy_config['charge_rate_max']], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )
        self.logger.info(f"Defined action space: low={-energy_config['discharge_rate_max']}, high={energy_config['charge_rate_max']}")

        # Define observation space with explicit float32 dtype
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
                np.inf,
                np.inf
            ], dtype=np.float32),
            dtype=np.float32
        )
        self.logger.info(f"Defined observation space: low={self.observation_space.low}, high={self.observation_space.high}")

        # Metadata for Gymnasium (optional, but recommended)
        self.metadata = {"render_modes": [], "render_fps": 4}

        # Internal State
        self.init: bool = False
        self.rng = np.random.default_rng()
        self.avg_price: float = 0.0
        self.energy_lvl: float = energy_config['init']
        self.reward_type: int = 0
        self.count: int = 0        # Step counter
        self.terminated: bool = False
        self.truncated: bool = False

        # Extract other configurations if necessary
        self.pricing_eta = self.env_config['pricing']['eta']
        self.time_steps_per_day_ratio = self.env_config['time']['time_steps_per_day_ratio']
        self.time_step_duration = self.env_config['time']['step_duration']
        self.max_steps_per_episode = self.env_config['time']['max_steps_per_episode']

        # Initialize the Reward Function
        self.logger.info(f"Setting up reward function: {reward_type}")
        self.reward: BaseReward = self.initialize_reward(reward_type)
        
        # Add reference to trained ISO agent
        self.trained_iso_agent = None
        
        # Load trained ISO model if provided
        if trained_iso_model_path:
            try:
                trained_iso_agent = PPO.load(trained_iso_model_path)
                self.set_trained_iso_agent(trained_iso_agent)
                self.logger.info(f"Loaded ISO model: {trained_iso_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load ISO model: {e}")
        
        self.logger.info("PCSunitEnv initialization complete.")
        
        # Add state tracking variables
        self.current_time = 0.0
        self.predicated_demand = 0.0
        self.pcs_demand = 0.0  # Track PCS's impact on grid


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

    #! Add method to set trained ISO agent
    def translate_to_iso_observation(self) -> np.ndarray:
        """
        Convert current state to ISO observation format
        """
        iso_observation = np.array([
            self.current_time,
            self.predicated_demand,
            self.pcs_demand 
        ], dtype=np.float32)
        
        self.logger.debug(f"Translated to ISO observation: {iso_observation}")
        return iso_observation
        

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
        self.logger.debug("PCSUnit and ISO have been reset.")

        # Reset internal state
        energy_config: Dict[str, Any] = self.pcs_unit_config['battery']['model_parameters']
        self.avg_price = 0.0
        self.energy_lvl = energy_config['init']
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
        current_time: float = (self.count * self.time_step_duration) / 1440  # 1440 minutes in a day
        self.logger.debug(f"Initial time set to {current_time} fraction of day.")

        # Update PCSUnit with current time and no action
        self.PCSUnit.update(time=current_time, battery_action=0.0)
        self.logger.debug("PCSUnit updated with initial time and no action.")

        # Fetch self-production and self-consumption
        production: float = self.PCSUnit.get_self_production()
        consumption: float = self.PCSUnit.get_self_consumption()
        self.logger.debug(f"Initial pcs-production: {production}, pcs-consumption: {consumption}")

        # Create initial observation
        observation: np.ndarray = np.array([
            self.energy_lvl,
            current_time,
            production,
            consumption
        ], dtype=np.float32)
        self.logger.debug(f"Initial observation: {observation}")

        info: Dict[str, float] = self.get_info()
        self.logger.debug(f"Initial info: {info}")

        # Reset state tracking
        self.current_time = 0.0
        self.predicated_demand = self.calculate_predicated_demand(self.current_time)
        self.pcs_demand = 0.0

        return (observation, info)

    #! Add method to calculate predicated demand
    def calculate_predicated_demand(self, time: float) -> float:
        """Calculate base grid demand using cosine function"""
        demand_config = self.env_config['predicated_demand']
        interval = time * demand_config['interval_multiplier']
        predicated_demand = demand_config['base_load'] + demand_config['amplitude'] * np.cos(
            (interval + demand_config['phase_shift']) * np.pi / demand_config['period_divisor']
        )
        self.logger.debug(f"Calculated predicated demand for time {time}: {predicated_demand}")
        return predicated_demand

    def step(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes a single time step in this order:
        1. Get current prices from ISO agent or use defaults
        2. Process PCS action based on prices
        3. Calculate net exchange and rewards
        """
        assert self.init, "Environment must be reset before stepping."
        
        # 1. Get or set prices (either from ISO agent or defaults)
        if self.trained_iso_agent is not None:
            iso_obs = self.translate_to_iso_observation()
            self.logger.debug(f"Getting prices from ISO agent with observation: {iso_obs}")
            try:
                prices = self.trained_iso_agent.predict(iso_obs, deterministic=True)[0]
                buy_price, sell_price = prices
                self.logger.info(
                    f"Using ISO agent prices:\n"
                    f"  - Buy Price: {buy_price:.2f} $/MWh\n"
                    f"  - Sell Price: {sell_price:.2f} $/MWh"
                )
            except Exception as e:
                self.logger.error(f"Failed to get prices from ISO agent: {e}")
                buy_price = self.iso_config.get('pricing', {}).get('default_buy_price', 50.0)
                sell_price = self.iso_config.get('pricing', {}).get('default_sell_price', 45.0)
                self.logger.warning(
                    f"Falling back to default prices:\n"
                    f"  - Buy Price: {buy_price:.2f} $/MWh\n"
                    f"  - Sell Price: {sell_price:.2f} $/MWh"
                )
        else:
            # Always ensure we have prices, even without an ISO agent
            buy_price = self.iso_config.get('pricing', {}).get('default_buy_price', 50.0)
            sell_price = self.iso_config.get('pricing', {}).get('default_sell_price', 45.0)
            self.logger.info(
                f"Using default prices (no ISO agent):\n"
                f"  - Buy Price: {buy_price:.2f} $/MWh\n"
                f"  - Sell Price: {sell_price:.2f} $/MWh"
            )

        # 2. Process and validate PCS action
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

        # Update time and state
        self.count += 1
        self.current_time = (self.count * self.time_step_duration) / self.env_config['time']['minutes_per_day']
        self.logger.debug(f"Time updated to {self.current_time:.3f} (day fraction)")
        
        # Update predicated demand
        self.predicated_demand = self.calculate_predicated_demand(self.current_time)
        
        # 3. Update PCS state with action
        if self.multi_action:
            self.PCSUnit.update(
                time=self.current_time,
                battery_action=battery_action,
                consumption_action=consumption_action,
                production_action=production_action
            )
        else:
            self.PCSUnit.update(
                time=self.current_time,
                battery_action=battery_action
            )

        # Get updated production and consumption
        production = self.PCSUnit.get_self_production()
        consumption = self.PCSUnit.get_self_consumption()
        
        # Calculate net exchange based on battery action
        if battery_action > 0:  # Charging
            net_exchange = (consumption + battery_action) - production
        elif battery_action < 0:  # Discharging
            net_exchange = consumption - (production + abs(battery_action))
        else:
            net_exchange = consumption - production

        # Get pricing function based on current ISO prices
        def pricing_function(quantity: float) -> float:
            """Returns appropriate price based on quantity."""
            if quantity >= 0:  # Buying from grid
                return buy_price
            else:  # Selling to grid
                return sell_price
            
    # Create info with pricing function for reward calculation
        info = {
            'net_exchange': net_exchange,
            'pricing_function': pricing_function  
        }
        
        reward = self.reward.compute_reward(info)
        self.logger.info(f"Step reward: {reward:.2f}")
        
        # Update energy level and tracking variables
        self.energy_lvl = self.PCSUnit.battery.get_state()
        self.pcs_demand = net_exchange
        
        # Create next observation
        observation = np.array([
            self.energy_lvl,
            self.current_time,
            production,
            consumption
        ], dtype=np.float32)

        # Check if episode is done
        done = self.count >= self.max_steps_per_episode
        if done:
            self.logger.info("Episode complete")
        
        return observation, float(reward), done, False, info

    def get_iso_pricing_function(self, quantity: float) -> float:
        # Get current observation for ISO agent
        iso_observation = self.translate_to_iso_observation()
        
        if self.trained_iso_agent is not None:
            buy_price, sell_price = self.trained_iso_agent.predict(iso_observation, deterministic=True)[0]
        else:
            buy_price = self.iso_config.get('default_buy_price', 50.0)  
            sell_price = self.iso_config.get('default_sell_price', 45.0)  
            self.logger.warning("trained_iso_agent is not set. Using default buy and sell prices.")
        
        def price_function(quantity: float) -> float:
            """Returns appropriate price based on quantity."""
            if quantity >= 0:  # Buying from grid
                return buy_price
            else:  # Selling to grid
                return sell_price
                
        return price_function

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

        # Close loggers if necessary
        # Example:
        logger_names = ['PCSunitEnv', 'Battery', 'ProductionUnit', 'ConsumptionUnit', 'PCSUnit'] 
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        self.logger.info("Environment closed successfully.")