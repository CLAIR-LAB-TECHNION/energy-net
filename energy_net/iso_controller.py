from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import os
import yaml
import logging
from stable_baselines3 import PPO
from gymnasium import spaces

from energy_net.utils.logger import setup_logger
from energy_net.rewards.base_reward import BaseReward
from energy_net.rewards.iso_reward import ISOReward
from energy_net.components.pcsunit import PCSUnit


class ISOController:
    """
    Independent System Operator (ISO) Controller responsible for setting electricity prices.
    Can operate with a trained PPO model or other pricing mechanisms.
    
    Observation Space:
        [time, predicated_demand, pcs_demand]
        
    Action Space:
        [buy_price, sell_price]
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',
        reward_type: str = 'iso',
        model_path: Optional[str] = None

    ):
        # Set up logger
        self.logger = setup_logger('ISOController', log_file)
        self.logger.info("Initializing ISO Controller")
        
        # Load configurations
        self.env_config = self.load_config(env_config_path)
        self.iso_config = self.load_config(iso_config_path)
        self.pcs_unit_config = self.load_config(pcs_unit_config_path)

        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.inf], dtype=np.float32),  # Explicitly set dtype
            high=np.array([1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define price bounds from ISO config
        price_bounds = self.iso_config.get('parameters', {})
        self.min_price = price_bounds.get('min_price', 0.0)
        self.max_price = price_bounds.get('max_price', 100.0)
        
        self.action_space = spaces.Box(
            low=np.array([self.min_price, self.min_price], dtype=np.float32),
            high=np.array([self.max_price, self.max_price], dtype=np.float32),
            dtype=np.float32
        )
        
        # Load PPO model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                self.logger.info(f"Loaded PPO model from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load PPO model: {e}")
                
        # Initialize state variables
        self.current_time = 0.0
        self.predicated_demand = 0.0
        self.pcs_demand = 0.0
        self.reset_called = False

        # Add tracking variables for PCS state
        self.production = 0.0
        self.consumption = 0.0

        # Add reference to trained PCS agent
        self.trained_pcs_agent = None

        uncertainty_config = self.env_config.get('demand_uncertainty', {})
        self.sigma = uncertainty_config.get('sigma', 0.0)
        self.reserve_price = self.env_config.get('reserve_price', 0.0)

        # Add missing initialization
        self.buy_price = self.min_price
        self.sell_price = self.min_price
        self.last_action = np.array([self.min_price, self.min_price])

        # Add time management variables from env_config
        self.time_step_duration = self.env_config.get('time', {}).get('step_duration', 5)  # in minutes
        self.count = 0  # Add step counter

        self.predicated_demand = self.calculate_predicated_demand(0.0)

        # Initialize the Reward Function
        self.logger.info(f"Setting up reward function: {reward_type}")
        self.reward: BaseReward = self.initialize_reward(reward_type)

        pcs_config = self.load_config(pcs_unit_config_path)
        battery_config = pcs_config['battery']['model_parameters']
        
        self.PCSUnit = PCSUnit(
            config=pcs_config,
            log_file=log_file
        )
        self.logger.info("Initialized PCSUnit component")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file."""
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.logger.debug(f"Loaded configuration from {config_path}")
        return config

    def build_observation(self) -> np.ndarray:
        """Construct observation vector for the model."""
        return np.array([
            self.current_time,
            self.predicated_demand,
            self.pcs_demand,
        ], dtype=np.float32)

    def calculate_predicated_demand(self, time: float) -> float:
        """Calculate base grid demand using cosine function"""
        # Convert time fraction to equivalent interval
        demand_config = self.env_config['predicated_demand']
        interval = time * demand_config['interval_multiplier']
        predicated_demand = demand_config['base_load'] + demand_config['amplitude'] * np.cos(
            (interval + demand_config['phase_shift']) * np.pi / demand_config['period_divisor']
        )
        return float(predicated_demand)

    def translate_to_pcs_observation(self) -> np.ndarray:
        """
        Match PCS controller's observation format exactly using PCSUnit
        """
        pcs_observation = np.array([
            self.PCSUnit.battery.get_state(),     # Energy level from actual battery
            self.current_time,                    # Time
            self.PCSUnit.get_self_production(),   # Production from PCSUnit
            self.PCSUnit.get_self_consumption()   # Consumption from PCSUnit
        ], dtype=np.float32)
        
        self.logger.debug(f"PCS observation from PCSUnit: energy={pcs_observation[0]}, "
                         f"time={pcs_observation[1]}, production={pcs_observation[2]}, "
                         f"consumption={pcs_observation[3]}")
        return pcs_observation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the controller state."""
        self.current_time = 0.0
        self.predicated_demand = 0.0
        self.pcs_demand = 0.0
        self.reset_called = True
        
        self.current_time = 0.0
        self.buy_price = self.min_price
        self.sell_price = self.min_price
        self.last_action = np.array([self.min_price, self.min_price])
        self.predicated_demand = self.calculate_predicated_demand(self.current_time)
        self.pcs_demand = 0.0
        
        # Reset PCSUnit 
        self.PCSUnit.reset()  

        observation = self.build_observation()
        info = {"status": "reset"}
        
        self.count = 0  # Reset step counter

        return observation, info

    def step(self, action: Union[np.ndarray, float]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step of the ISO controller.
        
        The step process follows this order:
        1. Update time and calculate new base demand
        2. Set electricity prices (buy/sell) based on ISO action
        3. Allow PCS units to respond to new prices
        4. Calculate final grid state and rewards
        
        Args:
            action (Union[np.ndarray, float]): Price setting action from the agent
                                             As array: [buy_price, sell_price]
                                             As float: Same price for buy/sell
        
        Returns:
            observation (np.ndarray): Next state observation [time, predicated_demand, pcs_demand]
            reward (float): Reward for this step based on grid stability and costs
            done (bool): Whether episode has ended (full day completed)
            truncated (bool): Whether episode was artificially terminated
            info (dict): Additional information about the step
        """
        if not self.reset_called:
            self.logger.warning("Step called before reset()")
            self.reset()

        # 1. Update time and demand prediction
        self.count += 1
        self.current_time = (self.count * self.time_step_duration) / (
            self.env_config['time']['minutes_per_day']
        )
        self.logger.debug(f"Advanced time to {self.current_time:.3f} (day fraction)")
        
        self.predicated_demand = self.calculate_predicated_demand(self.current_time)
        self.logger.debug(f"Predicted base demand: {self.predicated_demand:.2f} MWh")

        # 2. Process and set prices
        self.logger.debug(f"Processing raw action: {action}")
        if isinstance(action, np.ndarray):
            action = action.flatten()
        else:
            action = np.array([action, action])
            self.logger.debug(f"Converted scalar action to array: {action}")

        if not self.action_space.contains(action):
            self.logger.warning(f"Action {action} outside bounds, clipping to valid range")
            action = np.clip(action, self.action_space.low, self.action_space.high)

        self.buy_price, self.sell_price = action
        self.last_action = action
        
        self.logger.info(
            f"Step {self.count} - Price Settings:\n"
            f"  - Buy Price: {self.buy_price:.2f} $/MWh\n"
            f"  - Sell Price: {self.sell_price:.2f} $/MWh"
        )
        
        # 3. Get PCS response to the new prices
        self.pcs_demand = 0.0  # Reset PCS demand
        if self.trained_pcs_agent is not None:
            self.logger.debug("Getting PCS unit response to new prices...")
            pcs_obs = self.translate_to_pcs_observation()
            battery_action = self.simulate_pcs_response(pcs_obs)
            
            # Update PCSUnit state
            self.PCSUnit.update(
                time=self.current_time,
                battery_action=battery_action
            )
            
            # Calculate net exchange with grid
            self.production = self.PCSUnit.get_self_production()
            self.consumption = self.PCSUnit.get_self_consumption()
            
            # Calculate net grid exchange based on battery action
            if battery_action > 0:  # Charging
                net_exchange = (self.consumption + battery_action) - self.production
                self.logger.debug(f"Battery charging: {battery_action:.2f} MWh")
            elif battery_action < 0:  # Discharging
                net_exchange = self.consumption - (self.production + abs(battery_action))
                self.logger.debug(f"Battery discharging: {abs(battery_action):.2f} MWh")
            else:
                net_exchange = self.consumption - self.production
                self.logger.debug("Battery idle (no charge/discharge)")
                    
            self.pcs_demand = net_exchange
            
            self.logger.info(
                f"PCS Unit State:\n"
                f"  - Battery Action: {battery_action:.3f} MWh\n"
                f"  - Production: {self.production:.3f} MWh\n"
                f"  - Consumption: {self.consumption:.3f} MWh\n"
                f"  - Net Grid Exchange: {net_exchange:.3f} MWh"
            )

        # 4. Calculate final state and reward
        noise = np.random.normal(0, self.sigma)
        self.realized_demand = float(self.predicated_demand + noise)
        net_demand = self.realized_demand + self.pcs_demand
        
        self.logger.debug(
            f"Grid State:\n"
            f"  - Base Demand: {self.predicated_demand:.2f} MWh\n"
            f"  - Demand Noise: {noise:.2f} MWh\n"
            f"  - PCS Impact: {self.pcs_demand:.2f} MWh\n"
            f"  - Net Demand: {net_demand:.2f} MWh"
        )
        
        # Calculate dispatch and costs
        dispatch = self.predicated_demand
        shortfall = max(0.0, net_demand - dispatch)
        reserve_cost = self.reserve_price * shortfall
        
        self.logger.warning(
            f"Grid Shortfall:\n"
            f"  - Amount: {shortfall:.2f} MWh\n"
            f"  - Reserve Cost: ${reserve_cost:.2f}"
            )

        # Create observation for next step
        observation = np.array([
            float(self.current_time),
            float(self.predicated_demand),
            float(self.pcs_demand)
        ], dtype=np.float32)

        # Calculate reward
        reward_info = {
            'predicated_demand': self.predicated_demand,
            'realized_demand': self.realized_demand,
            'net_demand': net_demand,
            'dispatch': dispatch,
            'shortfall': shortfall,
            'reserve_cost': reserve_cost,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'pcs_demand': self.pcs_demand,
        }
        
        reward = self.reward.compute_reward(reward_info)
        self.logger.info(f"Step reward: {reward:.2f}")
        
        # Check if episode is done
        done = self.current_time >= 1.0
        if done:
            self.logger.info("Episode complete - Full day simulated")
        
        truncated = False
        
        info = {
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'predicated_demand': self.predicated_demand,
            'realized_demand': self.realized_demand,
            'pcs_demand': self.pcs_demand,
            'shortfall': shortfall,
            'reserve_cost': reserve_cost,
        }

        return observation, reward, done, truncated, info

    def get_pricing_function(self, observation_dict: Dict[str, Any]):
        """
        Returns a pricing function based on current state.
        
        Args:
            observation_dict: Dictionary containing current observation
            
        Returns:
            Callable: Function that returns price for given quantity
        """
        # Extract relevant information from observation
        time = observation_dict.get('time', 0.0)
        predicated_demand = observation_dict.get('predicated_demand', 0.0)
        pcs_demand = observation_dict.get('pcs_demand', 0.0)
        
        # Get current prices
        buy_price, sell_price = self.step(time, predicated_demand, pcs_demand)
        
        # Return a function that implements the pricing logic
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
        """
        return {}

    def close(self):
        """
        Cleanup method. Closes loggers and releases resources.
        """
        self.logger.info("Closing environment.")

        # Close loggers if necessary
        # Example:
        logger_names = [] 
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        self.logger.info("Environment closed successfully.")

    def set_trained_pcs_agent(self, pcs_agent):
        """Set the trained PCS agent for interaction during training"""
        self.trained_pcs_agent = pcs_agent
        try:
            test_obs = np.array([0.5, 0.5, 50.0, 50.0], dtype=np.float32)  
            test_action, _ = self.trained_pcs_agent.predict(test_obs, deterministic=True)
            self.logger.info(f"PCS agent test - observation: {test_obs}, action: {test_action}")
        except Exception as e:
            self.logger.error(f"PCS agent validation failed: {e}")
            raise e 
        
    def simulate_pcs_response(self, observation: np.ndarray) -> float:
        """
        Simulate PCS unit's response to current grid conditions using trained agent.
        
        The PCS unit makes decisions about battery charging/discharging based on:
        - Current battery level
        - Time of day
        - Current production and consumption levels
        - Grid prices (implicitly through training)
        
        Args:
            observation (np.ndarray): Current state observation for PCS unit
                                    [battery_level, time, production, consumption]
        
        Returns:
            float: Battery action (positive=charging, negative=discharging)
        """
        if self.trained_pcs_agent is None:
            self.logger.warning("No trained PCS agent available - returning no action")
            return 0.0
                
        self.logger.debug(f"PCS Agent Input State: {observation}")
        action, _ = self.trained_pcs_agent.predict(observation, deterministic=True)
        self.logger.debug(f"Raw PCS Agent Action: {action}")
        
        battery_action = action.item()
        
        energy_config = self.pcs_unit_config['battery']['model_parameters']
        self.logger.debug(
            f"Battery Constraints:\n"
            f"  - Charge Rate Max: {energy_config['charge_rate_max']}\n"
            f"  - Discharge Rate Max: {energy_config['discharge_rate_max']}\n"
            f"  - Selected Action: {battery_action}"
        )

        return battery_action
        
    def initialize_reward(self, reward_type: str) -> BaseReward:
        """
        Initializes the reward function based on the specified type.
        
        Args:
            reward_type (str): Type of reward ('iso' or 'cost').
            
        Returns:
            BaseReward: An instance of a reward class.
        """
        if reward_type in ['iso', 'cost']:  # Allow both 'iso' and 'cost' to work
            return ISOReward()
        else:
            self.logger.error(f"Unsupported reward type: {reward_type}")
            raise ValueError(f"Unsupported reward type: {reward_type}")