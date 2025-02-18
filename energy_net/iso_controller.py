from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import os
import yaml
import logging
from stable_baselines3 import PPO
from gymnasium import spaces
from energy_net.env import PricingPolicy  
from energy_net.market.iso.demand_patterns import DemandPattern, calculate_demand
from energy_net.market.iso.cost_types import CostType, calculate_costs

from energy_net.utils.logger import setup_logger
from energy_net.rewards.base_reward import BaseReward
from energy_net.rewards.iso_reward import ISOReward
from energy_net.components.pcsunit import PCSUnit
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO
from energy_net.market.iso.pcs_manager import PCSManager


class ISOController:
    """
    Independent System Operator (ISO) Controller responsible for setting electricity prices.
    Can operate with a trained PPO model or other pricing mechanisms.
    
    Observation Space:
        [time, predicted_demand, pcs_demand]
        
    Action Space:
        [b0, b1, b2, s0, s1, s2, dispatch_profile] 
        - b0, b1, b2: Polynomial coefficients for buy pricing
        - s0, s1, s2: Polynomial coefficients for sell pricing
        - dispatch_profile: 48 values for dispatch profile
    """
    
    def __init__(
        self,
        num_pcs_agents: int = 1, 
        pricing_policy: PricingPolicy = None,  
        demand_pattern=None,
        cost_type=None, 
        render_mode: Optional[str] = None,
        env_config_path: Optional[str] = 'configs/environment_config.yaml',
        iso_config_path: Optional[str] = 'configs/iso_config.yaml',
        pcs_unit_config_path: Optional[str] = 'configs/pcs_unit_config.yaml',
        log_file: Optional[str] = 'logs/environments.log',
        reward_type: str = 'iso',
        model_path: Optional[str] = None,
        trained_pcs_model_path: Optional[str] = None,
    ):
        # Set up logger
        self.logger = setup_logger('ISOController', log_file)
        self.logger.info(f"Initializing ISO Controller with {pricing_policy.value} policy")
        
        self.pricing_policy = pricing_policy
        self.demand_pattern = demand_pattern  # Store it as instance variable
        self.logger.info(f"Using demand pattern: {demand_pattern.value}")
        self.cost_type = cost_type
        self.logger.info(f"Using cost type: {cost_type.value}")

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

        # Get action space parameters from config based on pricing policy
        action_spaces_config = self.iso_config.get('action_spaces', {})
        policy_config = action_spaces_config.get(self.pricing_policy.value, {})
        
        if self.pricing_policy == PricingPolicy.QUADRATIC:
            # Define action space for quadratic pricing policy
            dispatch_config = policy_config.get('dispatch', {})
            poly_config = policy_config.get('polynomial', {})
            
            self.dispatch_min = dispatch_config.get('min', 0.0)
            self.dispatch_max = dispatch_config.get('max', 300.0)
            low_poly = poly_config.get('min', -100.0)
            high_poly = poly_config.get('max', 100.0)

            low_array = np.concatenate((
                np.full(6, low_poly, dtype=np.float32),
                np.full(self.max_steps_per_episode, self.dispatch_min, dtype=np.float32)
            ))
            high_array = np.concatenate((
                np.full(6, high_poly, dtype=np.float32),
                np.full(self.max_steps_per_episode, self.dispatch_max, dtype=np.float32)
            ))
                    
            self.action_space = spaces.Box(
                low=low_array,
                high=high_array,
                dtype=np.float32
            )
        elif self.pricing_policy == PricingPolicy.CONSTANT:
            dispatch_config = policy_config.get('dispatch', {})
            poly_config = policy_config.get('polynomial', {})
            self.dispatch_min = dispatch_config.get('min', 0.0)
            self.dispatch_max = dispatch_config.get('max', 300.0)
            low_const = poly_config.get('min', self.min_price)
            high_const = poly_config.get('max', self.max_price)

            low_array = np.concatenate((
                np.array([self.min_price, self.min_price], dtype=np.float32),
                np.full(self.max_steps_per_episode, self.dispatch_min, dtype=np.float32)
            ))
            high_array = np.concatenate((
                np.array([self.max_price, self.max_price], dtype=np.float32),
                np.full(self.max_steps_per_episode, self.dispatch_max, dtype=np.float32)
            ))
            
            self.action_space = spaces.Box(
                low=low_array,
                high=high_array,
                dtype=np.float32
            )
        elif self.pricing_policy == PricingPolicy.ONLINE:
            # Define action space for online pricing policy
            self.action_space = spaces.Box(
                low=np.array([self.min_price, self.min_price], dtype=np.float32),
                high=np.array([self.max_price, self.max_price], dtype=np.float32),
                dtype=np.float32
        )
                   

        self.buy_coef = np.zeros(3, dtype=np.float32)   # [b0, b1, b2]
        self.sell_coef = np.zeros(3, dtype=np.float32)  # [s0, s1, s2]
        self.dispatch_profile = np.zeros(self.max_steps_per_episode, dtype=np.float32)


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

        # Get costs from cost type
        self.reserve_price, self.dispatch_price = calculate_costs(
            cost_type,
            self.env_config
        )

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

        self.demand_pattern = demand_pattern
        self.logger.info(f"Using demand pattern: {demand_pattern.value}")

        # Replace single PCS with PCSManager
        self.pcs_manager = PCSManager(
            num_agents=num_pcs_agents,
            pcs_unit_config=self.pcs_unit_config,
            log_file=log_file
        )

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
        """
        Calculate predicted demand using selected pattern
        """
        return calculate_demand(
            time=time,
            pattern=self.demand_pattern,
            config=self.env_config['predicted_demand']
        )

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
        self.first_action_taken = False
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
        self.pcs_manager.reset_all()
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

        # 1. Update time and predicted demand
        self.count += 1            
        self.current_time = (self.count * self.time_step_duration) / self.env_config['time']['minutes_per_day']
        self.logger.debug(f"Advanced time to {self.current_time:.3f} (day fraction)")
        self.predicted_demand = self.calculate_predicted_demand(self.current_time)
        self.logger.debug(f"Predicted demand: {self.predicted_demand:.2f} MWh")

        # 2. Process ISO action
        if self.pricing_policy == PricingPolicy.QUADRATIC:
            if self.count == 1 and not self.first_action_taken:
                action = np.array(action).flatten()
                if len(action) != 6 + self.max_steps_per_episode: 
                    raise ValueError(
                        f"Expected action of length {6 + self.max_steps_per_episode}, "
                        f"got {len(action)}"
                    )
                
                self.buy_coef = action[0:3]    # [b0, b1, b2] 
                self.sell_coef = action[3:6]   # [s0, s1, s2] 
                self.dispatch_profile = action[6:]  # 48 values for dispatch profile
                self.dispatch_profile = np.clip(self.dispatch_profile, self.dispatch_min, self.dispatch_max)
            

                self.buy_iso = QuadraticPricingISO(
                    buy_a=float(self.buy_coef[0]),
                    buy_b=float(self.buy_coef[1]), 
                    buy_c=float(self.buy_coef[2])
                )
                self.sell_iso = QuadraticPricingISO(
                    buy_a=float(self.sell_coef[0]),
                    buy_b=float(self.sell_coef[1]),
                    buy_c=float(self.sell_coef[2])
                )

                self.first_action_taken = True
                self.logger.info(
                    f"Day-ahead polynomial for BUY: {self.buy_coef}, "
                    f"SELL: {self.sell_coef}, "
                    f"Dispatch profile: {self.dispatch_profile}"
                )
            else:
                self.logger.debug("Ignoring action - day-ahead polynomial & dispatch are already set.")
        
            buy_pricing_fn = self.buy_iso.get_pricing_function({'demand': self.predicted_demand})
            self.iso_buy_price = max(buy_pricing_fn(1.0),0)

            sell_pricing_fn = self.sell_iso.get_pricing_function({'demand': self.predicted_demand})
            self.iso_sell_price = max(sell_pricing_fn(1.0),0)
            dispatch = self.dispatch_profile[self.count - 1]
            self.logger.info(f"Step {self.count} - ISO Prices: Sell {self.iso_sell_price:.2f}, Buy {self.iso_buy_price:.2f}")

        elif self.pricing_policy == PricingPolicy.CONSTANT:
            if self.count == 1 and not self.first_action_taken:
                action = np.array(action).flatten()
                if len(action) != 2 + self.max_steps_per_episode:
                    raise ValueError(
                        f"Expected action of length {2 + self.max_steps_per_episode}, got {len(action)}"
                    )
                self.const_buy = float(action[0])
                self.const_sell = float(action[1])
                self.dispatch_profile = action[2:]
                self.buy_iso = QuadraticPricingISO(
                    buy_a=0.0,
                    buy_b=0.0,
                    buy_c=self.const_buy
                )
                self.sell_iso = QuadraticPricingISO(
                    buy_a=0.0,
                    buy_b=0.0,
                    buy_c=self.const_sell
                )   
                self.first_action_taken = True
                self.logger.info(
                    f"Day-ahead polynomial for BUY: {self.buy_coef}, "
                    f"SELL: {self.sell_coef}, "
                    f"Dispatch profile: {self.dispatch_profile}"
                )
            else:
                self.logger.debug("Ignoring action - day-ahead constant pricing & dispatch already set.")
            
            buy_pricing_fn = self.buy_iso.get_pricing_function({'demand': self.predicted_demand})
            self.iso_buy_price = buy_pricing_fn(1.0)

            sell_pricing_fn = self.sell_iso.get_pricing_function({'demand': self.predicted_demand})
            self.iso_sell_price = sell_pricing_fn(1.0)
            dispatch = self.dispatch_profile[self.count - 1]
            self.logger.info(f"Step {self.count} - ISO Prices: Sell {self.iso_sell_price:.2f}, Buy {self.iso_buy_price:.2f}")

        elif self.pricing_policy == PricingPolicy.ONLINE:
            self.logger.debug(f"Processing ISO action: {action}")
            dispatch = self.predicted_demand
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



        # 3. Get PCS responses
        self.production, self.consumption, self.pcs_demand = self.pcs_manager.simulate_step(
            current_time=self.current_time,
            iso_buy_price=self.iso_buy_price,
            iso_sell_price=self.iso_sell_price
        )

        # 4. Calculate grid state and costs
        noise = np.random.normal(0, self.sigma)
        self.realized_demand = float(self.predicted_demand + noise)
        net_demand = self.realized_demand + self.pcs_demand
        self.logger.debug(f"Net demand: {net_demand:.2f} MWh")
        dispatch_cost = self.dispatch_price * dispatch
        shortfall = max(0.0, net_demand - dispatch)
        if self.pcs_demand>0: 
            price = self.iso_sell_price
        else:
            price = self.iso_buy_price

        pcs_costs = self.pcs_demand*price
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
            'battery_level': self.pcs_manager.battery_levels[-1] if self.pcs_manager.battery_levels else [],  
            'net_exchange': self.pcs_demand,
            'dispatch_cost': dispatch_cost,
            'shortfall': shortfall,
            'reserve_cost': reserve_cost,
            'dispatch': dispatch,
            'pcs_demand': self.pcs_demand,
            'pcs_costs': pcs_costs,
            'pcs_actions': self.pcs_manager.battery_actions[-1] if self.pcs_manager.battery_actions else []  
        }

        # 5. Compute reward

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

    def set_trained_pcs_agent(self, agent_idx: int, pcs_agent_path: str):
        """Set trained agent for specific PCS unit"""
        success = self.pcs_manager.set_trained_agent(agent_idx, pcs_agent_path)
        if success:
            self.logger.info(f"Successfully set trained agent {agent_idx} from {pcs_agent_path}")
        else:
            self.logger.error(f"Failed to set trained agent {agent_idx} from {pcs_agent_path}")
        return success

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
