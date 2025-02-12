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
        [time, predicted_demand, pcs_demand]
        
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
        model_path: Optional[str] = None,
        trained_pcs_model_path: Optional[str] = None
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
            low=np.array([0.0, 0.0, -np.inf], dtype=np.float32),  # [time, predicted_demand, pcs_demand]
            high=np.array([1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define price bounds from ISO config
        pricing_config = self.iso_config.get('pricing', {})
        price_params = pricing_config.get('parameters', {})
        self.min_price = price_params.get('min_price', pricing_config.get('default_sell_price', 1.0))
        self.max_price = price_params.get('max_price', pricing_config.get('default_buy_price', 10.0))
        self.max_steps_per_episode = self.env_config['time'].get('max_steps_per_episode', 48)
        self.logger.info(f"Price bounds set to: min={self.min_price}, max={self.max_price}")
        
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
        self.predicted_demand = 0.0
        self.pcs_demand = 0.0
        self.reset_called = False

        # Tracking variables for PCS state
        self.production = 0.0
        self.consumption = 0.0

        # Reference to trained PCS agent (to simulate PCS response)
        self.trained_pcs_agent = None

        uncertainty_config = self.env_config.get('demand_uncertainty', {})
        self.sigma = uncertainty_config.get('sigma', 0.0)
        self.reserve_price = self.env_config.get('reserve_price', 0.0)
        self.dispatch_price = self.env_config.get('dispatch_price', 0.0)

        # Initialize ISO prices with default values
        self.iso_sell_price = self.min_price
        self.iso_buy_price = self.min_price

        # Time management variables
        self.time_step_duration = self.env_config.get('time', {}).get('step_duration', 5)  # in minutes
        self.count = 0
        self.predicted_demand = self.calculate_predicted_demand(0.0)

        self.logger.info(f"Setting up reward function: {reward_type}")
        self.reward: BaseReward = self.initialize_reward(reward_type)

        # Initialize PCSUnit component
        pcs_config = self.load_config(pcs_unit_config_path)
        self.PCSUnit = PCSUnit(
            config=pcs_config,
            log_file=log_file
        )
        self.logger.info("Initialized PCSUnit component")

        # Load trained PCS agent if provided
        if trained_pcs_model_path:
            try:
                self.trained_pcs_agent = PPO.load(trained_pcs_model_path)
                self.logger.info(f"Loaded trained PCS agent from {trained_pcs_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load trained PCS agent: {e}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.logger.debug(f"Loaded configuration from {config_path}")
        return config

    def build_observation(self) -> np.ndarray:
        return np.array([
            self.current_time,
            self.predicted_demand,
            self.pcs_demand,
        ], dtype=np.float32)

    def calculate_predicted_demand(self, time: float) -> float:
        demand_config = self.env_config['predicted_demand']
        interval = time * demand_config['interval_multiplier']
        predicted_demand = demand_config['base_load'] + demand_config['amplitude'] * np.cos(
            (interval + demand_config['phase_shift']) * np.pi / demand_config['period_divisor']
        )
        return float(predicted_demand)

    def translate_to_pcs_observation(self) -> np.ndarray:
        """
        Converts current state to PCS observation format.
        
        Returns:
            np.ndarray: Observation array containing:
                - Current battery level
                - Time of day
                - Current production
                - Current consumption
        """
        pcs_observation = np.array([
            self.PCSUnit.battery.get_state(),
            self.current_time,
            self.PCSUnit.get_self_production(),
            self.PCSUnit.get_self_consumption()
        ], dtype=np.float32)
        
        self.logger.debug(
            f"PCS Observation:\n"
            f"  Battery Level: {pcs_observation[0]:.2f} MWh\n"
            f"  Time: {pcs_observation[1]:.3f}\n"
            f"  Production: {pcs_observation[2]:.2f} MWh\n"
            f"  Consumption: {pcs_observation[3]:.2f} MWh"
        )
        
        return pcs_observation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Resets the ISO controller state.
        """
        self.logger.info("Resetting ISO Controller environment.")

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.logger.debug(f"Random number generator seeded with: {seed}")
        else:
            self.rng = np.random.default_rng()
            self.logger.debug("Random number generator initialized without seed.")

        # Reset PCSUnit
        self.PCSUnit.reset()
        self.logger.debug("PCSUnit has been reset.")

        # Reset internal state
        let_energy = self.pcs_unit_config['battery']['model_parameters']
        self.avg_price = 0.0
        self.energy_lvl = let_energy['init']
        self.PCSUnit.reset(initial_battery_level=self.energy_lvl)
        self.reward_type = 0
        if options and 'reward' in options:
            if options.get('reward') == 1:
                self.reward_type = 1
                self.logger.debug("Reward type set to 1 based on options.")
            else:
                self.logger.debug(f"Reward type set to {self.reward_type} based on options.")
        else:
            self.logger.debug("No reward type option provided; using default.")

        self.count = 0
        self.terminated = False
        self.truncated = False
        self.init = True

        self.current_time = 0.0
        self.predicted_demand = self.calculate_predicted_demand(self.current_time)
        self.pcs_demand = 0.0

        if self.trained_pcs_agent is not None:
            try:
                pcs_obs = self.translate_to_pcs_observation()
                battery_action = self.simulate_pcs_response(pcs_obs)
                self.PCSUnit.update(time=self.current_time, battery_action=battery_action)
                self.production = self.PCSUnit.get_self_production()
                self.consumption = self.PCSUnit.get_self_consumption()
                self.pcs_demand = self.consumption - self.production
                self.logger.info(f"Updated PCS state on reset: battery_action={battery_action:.3f}, production={self.production:.3f}, consumption={self.consumption:.3f}")
            except Exception as e:
                self.logger.error(f"Failed to update PCS state on reset: {e}")
        else:
            self.logger.info("No trained PCS agent available on reset; using default PCS state.")

        self.logger.info(
            f"Environment Reset:\n"
            f"  Time: {self.current_time:.3f}\n"
            f"  Initial Demand: {self.predicted_demand:.2f} MWh\n"
            f"  PCS Demand: {self.pcs_demand:.2f} MWh\n"
            f"  ISO Sell Price: ${self.iso_sell_price:.2f}/MWh\n"
            f"  ISO Buy Price: ${self.iso_buy_price:.2f}/MWh"
        )

        observation = np.array([
            self.current_time,
            self.predicted_demand,
            self.pcs_demand
        ], dtype=np.float32)
        self.logger.debug(f"Initial observation: {observation}")

        info = {"status": "reset"}
        return observation, info

    def step(self, action: Union[float, np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes a single time step in this order:
        1. Update time and predicted demand
        2. Process ISO action
        3. Get PCS response (if trained PCS agent is available)
        4. Calculate grid state and costs
        5. Compute reward
        """
        assert self.init, "Environment must be reset before stepping."

        self.count += 1
        self.current_time = (self.count * self.time_step_duration) / self.env_config['time']['minutes_per_day']
        self.logger.debug(f"Advanced time to {self.current_time:.3f} (day fraction)")
        self.predicted_demand = self.calculate_predicted_demand(self.current_time)
        self.logger.debug(f"Predicted demand: {self.predicted_demand:.2f} MWh")

        self.logger.debug(f"Processing ISO action: {action}")
        if isinstance(action, np.ndarray):
            action = action.flatten()
        else:
            action = np.array([action, action])
            self.logger.debug(f"Converted scalar action to array: {action}")
        if not self.action_space.contains(action):
            self.logger.warning(f"Action {action} out of bounds; clipping.")
            action = np.clip(action, self.action_space.low, self.action_space.high)
        self.iso_sell_price, self.iso_buy_price = action
        self.logger.info(f"Step {self.count} - ISO Prices: Sell {self.iso_sell_price:.2f}, Buy {self.iso_buy_price:.2f}")

        if self.trained_pcs_agent is not None:
            try:
                pcs_obs = self.translate_to_pcs_observation()
                battery_action = self.simulate_pcs_response(pcs_obs)
                self.PCSUnit.update(time=self.current_time, battery_action=battery_action)
                self.production = self.PCSUnit.get_self_production()
                self.consumption = self.PCSUnit.get_self_consumption()
                if battery_action > 0:
                    net_exchange = (self.consumption + battery_action) - self.production
                    self.logger.debug(f"Battery charging: {battery_action:.2f} MWh")
                elif battery_action < 0:
                    net_exchange = self.consumption - (self.production + abs(battery_action))
                    self.logger.debug(f"Battery discharging: {abs(battery_action):.2f} MWh")
                else:
                    net_exchange = self.consumption - self.production
                    self.logger.debug("Battery idle (no charge/discharge)")
                self.pcs_demand = net_exchange
                self.logger.info(f"PCS response: battery_action={battery_action:.3f}, production={self.production:.3f}, consumption={self.consumption:.3f}, net_exchange={net_exchange:.3f}")
            except Exception as e:
                self.logger.error(f"Failed to get PCS response on step: {e}")
        else:
            # Simulate maximum charging behavior when no PCS agent is present
            # This helps validate that ISO responds correctly to constant buying behavior
            charge_rate_max = self.pcs_unit_config['battery']['model_parameters']['charge_rate_max']
            battery_action = charge_rate_max  # Always charge at maximum rate
            net_exchange = (self.consumption + battery_action) - self.production
            self.pcs_demand = net_exchange
            
            self.PCSUnit.update(time=self.current_time, battery_action=battery_action)
            self.production = self.PCSUnit.get_self_production()
            self.consumption = self.PCSUnit.get_self_consumption()
            
            # Net exchange will be consumption + charging - production
            self.pcs_demand = (self.consumption + battery_action) - self.production
            
            # Validate ISO's pricing response
            if self.iso_sell_price < self.max_price * 0.9:  # Allow for some wiggle room
                self.logger.warning(
                    f"Suboptimal ISO behavior detected: "
                    f"When PCS is constantly buying (charging at {battery_action:.2f} MWh), "
                    f"ISO sell price ({self.iso_sell_price:.2f}) should be closer to "
                    f"maximum ({self.max_price:.2f})"
                )
            
            self.logger.info(
                f"No PCS agent - simulating constant charging: "
                f"battery_action={battery_action:.2f}, "
                f"consumption={self.consumption:.2f}, "
                f"production={self.production:.2f}, "
                f"net_demand={self.pcs_demand:.2f}"
            )

        noise = np.random.normal(0, self.sigma)
        self.realized_demand = float(self.predicted_demand + noise)
        net_demand = self.realized_demand + self.pcs_demand
        self.logger.debug(f"Net demand: {net_demand:.2f} MWh")
        dispatch = self.predicted_demand
        dispatch_cost = self.dispatch_price * dispatch
        shortfall = max(0.0, net_demand - dispatch)
        reserve_cost = self.reserve_price * shortfall

        self.logger.warning(
            f"Grid Shortfall:\n"
            f"  - Amount: {shortfall:.2f} MWh\n"
            f"  - Reserve Cost: ${reserve_cost:.2f}"
        )

        info = {
            'iso_sell_price': self.iso_sell_price,
            'iso_buy_price': self.iso_buy_price,
            'predicted_demand': self.predicted_demand,
            'realized_demand': self.realized_demand,
            'production': self.production,
            'consumption': self.consumption,
            'battery_level': self.PCSUnit.battery.get_state(),
            'net_exchange': self.pcs_demand,
            'dispatch_cost': dispatch_cost,
            'shortfall': shortfall,
            'reserve_cost': reserve_cost,
            'pcs_demand': self.pcs_demand
        }

        reward = self.reward.compute_reward(info)
        self.logger.info(f"Step reward: {reward:.2f}")

        done = self.count >= self.max_steps_per_episode
        if done:
            self.logger.info("Episode complete - Full day simulated")
        truncated = False

        observation = np.array([
            self.current_time,
            self.predicted_demand,
            self.pcs_demand
        ], dtype=np.float32)

        self.logger.info(
            f"Grid State Step {self.count}:\n"
            f"  Time: {self.current_time:.3f}\n"
            f"  Predicted Demand: {self.predicted_demand:.2f} MWh\n"
            f"  Realized Demand: {self.realized_demand:.2f} MWh\n"
            f"  PCS Demand: {self.pcs_demand:.2f} MWh\n"
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

        return observation, float(reward), done, truncated, info

    def get_info(self) -> Dict[str, float]:
        return {"running_avg": self.avg_price}

    def close(self):
        self.logger.info("Closing ISO Controller environment.")
        for logger_name in []:
            logger = logging.getLogger(logger_name)
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        self.logger.info("ISO Controller environment closed successfully.")

    def set_trained_pcs_agent(self, pcs_agent):
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
        Simulates the PCS unit's response to current market conditions.
        
        Args:
            observation (np.ndarray): Current state observation for PCS unit.
            
        Returns:
            float: Battery action (positive for charging, negative for discharging).
        """
        if self.trained_pcs_agent is None:
            self.logger.warning("No trained PCS agent available - simulating default charging behavior")
            return self.pcs_unit_config['battery']['model_parameters']['charge_rate_max']
            
        self.logger.debug(f"Sending observation to PCS agent: {observation}")
        action, _ = self.trained_pcs_agent.predict(observation, deterministic=True)
        battery_action = action.item()
        
        energy_config = self.pcs_unit_config['battery']['model_parameters']
        self.logger.info(
            f"PCS Response:\n"
            f"  Battery Action: {battery_action:.2f} MWh\n"
            f"  Max Charge: {energy_config['charge_rate_max']:.2f} MWh\n"
            f"  Max Discharge: {energy_config['discharge_rate_max']:.2f} MWh"
        )
        return battery_action

    def initialize_reward(self, reward_type: str) -> BaseReward:
        """
        Creates the appropriate reward function instance.
        
        Args:
            reward_type (str): Type of reward ('iso' or 'cost')
            
        Returns:
            BaseReward: Configured reward function
            
        Raises:
            ValueError: If reward_type is not supported
        """
        if reward_type in ['iso', 'cost']:
            self.logger.info(f"Initializing {reward_type} reward function")
            return ISOReward()
        else:
            self.logger.error(f"Unsupported reward type: {reward_type}")
            raise ValueError(f"Unsupported reward type: {reward_type}")
