"""
Pricing Strategy Module for ISO Controllers

This module implements the Strategy pattern for different pricing policies used by the
Independent System Operator (ISO). It defines a common interface for all pricing strategies
and provides concrete implementations for:

1. Online Pricing: Real-time price setting at each time step
2. Quadratic Pricing: Polynomial-based pricing with coefficients set at the beginning
3. Constant Pricing: Fixed pricing throughout the episode

Each strategy handles:
- Action space definition based on its policy
- Processing agent actions into actual prices and dispatch
- Validation of actions within price boundaries
- Day-ahead vs. real-time action processing

This design allows for easy extension to new pricing policies by implementing
additional strategy classes.
"""

from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod
from energy_net.market.pricing_policy import PricingPolicy
from gymnasium import spaces
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO

class PricingStrategy(ABC):
    """
    Base strategy interface for pricing policies.
    
    This abstract class defines the interface for all pricing strategies.
    Each concrete strategy handles a specific pricing policy (Quadratic, Online, Constant).
    
    The Strategy pattern allows the ISO controller to use different pricing mechanisms
    without changing its core logic, by delegating pricing decisions to the appropriate
    strategy object.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            logger: Logger instance for logging
        """
        self.min_price = min_price
        self.max_price = max_price
        self.max_steps_per_episode = max_steps_per_episode
        self.logger = logger
    
    @abstractmethod
    def create_action_space(self, use_dispatch_action: bool = False) -> spaces.Space:
        """
        Create the appropriate action space for this pricing strategy.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A gymnasium Space object representing the action space
        """
        pass
    
    @abstractmethod
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the pricing strategy.
        
        Args:
            action: The action taken by the agent
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value (predicted_demand if not controlled by action)
            - first_action_taken: Updated first_action_taken flag
        """
        pass


class QuadraticPricingStrategy(PricingStrategy):
    """
    Strategy for the Quadratic pricing policy.
    
    This strategy uses polynomial coefficients to determine prices. The agent sets
    coefficients for quadratic functions at the beginning of an episode (day-ahead),
    and these coefficients are then used to calculate prices throughout the day
    based on demand.
    
    Pricing Formula:
        price = a * demand² + b * demand + c
    
    The agent sets the coefficients [a, b, c] for both buy and sell prices,
    resulting in 6 total coefficients.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the quadratic pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the quadratic pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        policy_config = config.get('quadratic', {})
        dispatch_config = policy_config.get('dispatch', {})
        poly_config = policy_config.get('polynomial', {})
        
        self.dispatch_min = dispatch_config.get('min', 0.0)
        self.dispatch_max = dispatch_config.get('max', 300.0)
        self.low_poly = poly_config.get('min', -100.0)
        self.high_poly = poly_config.get('max', 100.0)
        
        # Initialize price coefficients and dispatch profile
        self.buy_coef = np.zeros(3, dtype=np.float32)   # [b0, b1, b2]
        self.sell_coef = np.zeros(3, dtype=np.float32)  # [s0, s1, s2]
        
        # Initialize ISO pricing objects
        self.buy_iso = None
        self.sell_iso = None
    
    def create_action_space(self, use_dispatch_action: bool = False) -> spaces.Space:
        """
        Create the action space for quadratic pricing, optionally including dispatch.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A Box space with dimensions for polynomial coefficients and optionally dispatch
        """
        if use_dispatch_action:
            # Include dispatch in the action space (now just a single value)
            low_array = np.concatenate((
                np.full(6, self.low_poly, dtype=np.float32),
                np.array([self.dispatch_min], dtype=np.float32)
            ))
            high_array = np.concatenate((
                np.full(6, self.high_poly, dtype=np.float32),
                np.array([self.dispatch_max], dtype=np.float32)
            ))
        else:
            # Only include pricing coefficients
            low_array = np.full(6, self.low_poly, dtype=np.float32)
            high_array = np.full(6, self.high_poly, dtype=np.float32)
                
        return spaces.Box(
            low=low_array,
            high=high_array,
            dtype=np.float32
        )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the quadratic pricing strategy.
        
        In the quadratic pricing model, price coefficients are set on the first step
        (day-ahead pricing), but dispatch is now determined at each step.
        
        Action format when use_dispatch_action is False:
            [b0, b1, b2, s0, s1, s2]
            - b0, b1, b2: Buy price polynomial coefficients
            - s0, s1, s2: Sell price polynomial coefficients
            
        Action format when use_dispatch_action is True:
            [b0, b1, b2, s0, s1, s2, dispatch]
            - The dispatch value is used for the current step
            
        Args:
            action: The action taken by the agent 
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        iso_buy_price = 0.0
        iso_sell_price = 0.0
        dispatch = predicted_demand  # Default to predicted demand if dispatch not provided
        
        action = np.array(action).flatten()
        
        # Process pricing coefficients on first step
        if step_count == 1 and not first_action_taken:
            if use_dispatch_action:
                # Extract pricing coefficients (first 6 values)
                self.buy_coef = action[0:3]    # [b0, b1, b2] 
                self.sell_coef = action[3:6]   # [s0, s1, s2]
                # The dispatch is handled below, with current step's action
            else:
                # Only extract the pricing coefficients
                expected_length = 6  # 6 polynomial coefficients
                if len(action) < expected_length:
                    if self.logger:
                        self.logger.error(
                            f"Expected at least {expected_length} pricing coefficients, "
                            f"got {len(action)}"
                        )
                    raise ValueError(
                        f"Expected at least {expected_length} pricing coefficients, "
                        f"got {len(action)}"
                    )
                
                self.buy_coef = action[0:3]    # [b0, b1, b2] 
                self.sell_coef = action[3:6]   # [s0, s1, s2]
            
            # Initialize ISO pricing objects
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

            first_action_taken = True
            if self.logger:
                log_msg = f"Day-ahead polynomial for BUY: {self.buy_coef}, SELL: {self.sell_coef}"
                self.logger.info(log_msg)
        
        # Calculate prices using the polynomial coefficients
        buy_pricing_fn = self.buy_iso.get_pricing_function({'demand': predicted_demand}) if self.buy_iso else lambda x: 0
        iso_buy_price = max(buy_pricing_fn(predicted_demand), 0)

        sell_pricing_fn = self.sell_iso.get_pricing_function({'demand': predicted_demand}) if self.sell_iso else lambda x: 0
        iso_sell_price = 0.9*iso_buy_price
        #iso_sell_price = max(sell_pricing_fn(predicted_demand), 0)
        
        # Process dispatch at each step if enabled
        if use_dispatch_action:
            # If it's the first step, we already extracted the first 6 values above
            # For the dispatch, use the 7th value
            if len(action) >= 7:
                dispatch = action[6]
                # ENSURE STRICT CLIPPING of dispatch values
                dispatch = float(np.clip(dispatch, self.dispatch_min, self.dispatch_max))
                self.logger.info(f"Clipped dispatch to {dispatch:.2f} [range: {self.dispatch_min:.1f}-{self.dispatch_max:.1f}]")
            else:
                if self.logger:
                    self.logger.warning(f"Expected at least 7 values for action with dispatch, got {len(action)}. Using predicted demand as dispatch.")
                dispatch = predicted_demand
        
        if self.logger:
            self.logger.info(
                f"Step {step_count} - ISO Prices: Sell {iso_sell_price:.2f}, Buy {iso_buy_price:.2f}, " +
                f"Dispatch: {dispatch:.2f}"
            )
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class ConstantPricingStrategy(PricingStrategy):
    """
    Strategy for the Constant pricing policy.
    
    This strategy uses constant prices for an entire episode. The agent sets 
    fixed buy and sell prices at the beginning of an episode (day-ahead),
    and these prices remain unchanged throughout the day.
    
    This is the simplest pricing strategy and serves as a baseline for
    comparison with more dynamic strategies.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the constant pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the constant pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        policy_config = config.get('constant', {})
        dispatch_config = policy_config.get('dispatch', {})
        poly_config = policy_config.get('polynomial', {})
        
        self.dispatch_min = dispatch_config.get('min', 0.0)
        self.dispatch_max = dispatch_config.get('max', 300.0)
        self.low_const = poly_config.get('min', min_price)
        self.high_const = poly_config.get('max', max_price)
        
        # Initialize constant prices
        self.const_buy = 0.0
        self.const_sell = 0.0
        
        # Initialize ISO pricing objects
        self.buy_iso = None
        self.sell_iso = None
    
    def create_action_space(self, use_dispatch_action: bool = False) -> spaces.Space:
        """
        Create the action space for constant pricing, optionally including dispatch.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A Box space with dimensions for constant buy/sell prices and optionally dispatch
        """
        if use_dispatch_action:
            # Include a single dispatch value in the action space
            low_array = np.array([self.min_price, self.min_price, self.dispatch_min], dtype=np.float32)
            high_array = np.array([self.max_price, self.max_price, self.dispatch_max], dtype=np.float32)
        else:
            # Only include constant prices
            low_array = np.array([self.min_price, self.min_price], dtype=np.float32)
            high_array = np.array([self.max_price, self.max_price], dtype=np.float32)
        
        return spaces.Box(
            low=low_array,
            high=high_array,
            dtype=np.float32
        )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the constant pricing strategy.
        
        Prices are set on the first step and remain constant, but dispatch is now
        determined at each step.
        
        Args:
            action: The action taken by the agent (constant buy/sell prices + optional dispatch)
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        iso_buy_price = 0.0
        iso_sell_price = 0.0
        dispatch = predicted_demand  # Default to predicted demand if dispatch not provided
        
        action = np.array(action).flatten()
        
        # Process constant prices on first step
        if step_count == 1 and not first_action_taken:
            if use_dispatch_action:
                if len(action) >= 3:
                    # Extract constant prices (first 2 values)
                    self.const_buy = float(action[0])
                    self.const_sell = float(action[1])
                    # Dispatch is processed below for each step
                else:
                    if self.logger:
                        self.logger.error(f"Expected at least 3 values for action with dispatch, got {len(action)}.")
                    raise ValueError(f"Expected at least 3 values for action with dispatch, got {len(action)}.")
            else:
                # Only extract constant prices
                if len(action) >= 2:
                    self.const_buy = float(action[0])
                    self.const_sell = float(action[1])
                else:
                    if self.logger:
                        self.logger.error(f"Expected at least 2 price values, got {len(action)}.")
                    raise ValueError(f"Expected at least 2 price values, got {len(action)}.")
            
            # Initialize ISO pricing objects for constant prices
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
            
            first_action_taken = True
            if self.logger:
                log_msg = f"Day-ahead constant prices - BUY: {self.const_buy}, SELL: {self.const_sell}"
                self.logger.info(log_msg)
        
        # Calculate constant prices
        buy_pricing_fn = self.buy_iso.get_pricing_function({'demand': predicted_demand}) if self.buy_iso else lambda x: 0
        iso_buy_price = buy_pricing_fn(1.0)

        sell_pricing_fn = self.sell_iso.get_pricing_function({'demand': predicted_demand}) if self.sell_iso else lambda x: 0
        iso_sell_price = sell_pricing_fn(1.0)
        
        # Process dispatch at each step if enabled
        if use_dispatch_action:
            if len(action) >= 3:
                dispatch = action[2]
                # ENSURE STRICT CLIPPING of dispatch values
                dispatch = float(np.clip(dispatch, self.dispatch_min, self.dispatch_max))
                self.logger.info(f"Clipped dispatch to {dispatch:.2f} [range: {self.dispatch_min:.1f}-{self.dispatch_max:.1f}]")
            else:
                if self.logger:
                    self.logger.warning(f"Expected at least 3 values for action with dispatch, got {len(action)}. Using predicted demand as dispatch.")
                dispatch = predicted_demand
        
        if self.logger:
            self.logger.info(
                f"Step {step_count} - ISO Prices: Sell {iso_sell_price:.2f}, Buy {iso_buy_price:.2f}, " +
                f"Dispatch: {dispatch:.2f}"
            )
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class OnlinePricingStrategy(PricingStrategy):
    """
    Strategy for the Online pricing policy.
    
    This strategy allows the agent to update prices at each time step (real-time pricing).
    It provides the most flexibility, allowing the ISO to respond immediately to changing
    grid conditions.
    
    Action format when use_dispatch_action is False:
        [buy_price, sell_price]
        
    Action format when use_dispatch_action is True:
        [buy_price, sell_price, dispatch]
    
    Each action directly sets the prices (and optionally dispatch) for the current time step.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the online pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the online pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        # Extract specific bounds from the config
        online_config = config.get('online', {})
        self.buy_price_min = online_config.get('buy_price', {}).get('min', min_price)
        self.buy_price_max = online_config.get('buy_price', {}).get('max', max_price)
        self.sell_price_min = online_config.get('sell_price', {}).get('min', min_price)
        self.sell_price_max = online_config.get('sell_price', {}).get('max', max_price)
        self.dispatch_min = online_config.get('dispatch', {}).get('min', 0.0)
        self.dispatch_max = online_config.get('dispatch', {}).get('max', 300.0)
        
        if self.logger:
            self.logger.info(
                f"Initialized OnlinePricingStrategy with bounds: "
                f"Buy Price [{self.buy_price_min}, {self.buy_price_max}], "
                f"Sell Price [{self.sell_price_min}, {self.sell_price_max}]"
            )
    

    
    def create_action_space(self, use_dispatch_action: bool = False) -> spaces.Space:
        """
        Create the action space for online pricing, optionally including dispatch.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A Box space with dimensions for buy/sell prices and optionally dispatch
        """
        if use_dispatch_action:
            # Include dispatch in the action space
            return spaces.Box(
                low=np.array([self.buy_price_min, self.sell_price_min, self.dispatch_min], dtype=np.float32),
                high=np.array([self.buy_price_max, self.sell_price_max, self.dispatch_max], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Only include buy/sell prices
            return spaces.Box(
                low=np.array([self.buy_price_min, self.sell_price_min], dtype=np.float32),
                high=np.array([self.buy_price_max, self.sell_price_max], dtype=np.float32),
                dtype=np.float32
            )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the online pricing strategy.
        
        In the online pricing model, actions directly set the buy/sell prices
        for the current time step, allowing for real-time price adjustments.
        
        Action format when use_dispatch_action is False:
            [buy_price, sell_price]
            
        Action format when use_dispatch_action is True:
            [buy_price, sell_price, dispatch]
            
        Args:
            action: The action taken by the agent
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        if self.logger:
            self.logger.info(f"Processing ISO action: {action}")
        
        dispatch = predicted_demand  # Default to predicted demand if dispatch not provided
        
        if isinstance(action, np.ndarray):
            action = action.flatten()
        else:
            if self.logger:
                self.logger.info(f"Converting scalar action to array: {action}")
            action = np.array([action, action])
        
        if use_dispatch_action:
            # Extract prices and dispatch from action
            if len(action) >= 3:
                # Format: [buy_price, sell_price, dispatch]
                iso_buy_price = action[0]
                iso_sell_price = action[1]
                dispatch = action[2]
            else:
                if self.logger:
                    self.logger.warning(f"Expected 3 values for action with dispatch, got {len(action)}. Using only prices.")
                iso_buy_price = action[0]
                iso_sell_price = action[1] if len(action) > 1 else action[0]
        else:
            # Extract only prices
            iso_buy_price = action[0]
            iso_sell_price = action[1] if len(action) > 1 else action[0]
            
        # Ensure the prices are within bounds
        iso_buy_price = float(np.clip(iso_buy_price, self.buy_price_min, self.buy_price_max))
        iso_sell_price = float(np.clip(iso_sell_price, self.sell_price_min, self.sell_price_max))
        
        # Ensure dispatch is within bounds if provided
        if use_dispatch_action:
            dispatch = float(np.clip(dispatch, self.dispatch_min, self.dispatch_max))
        
        if self.logger:
            log_msg = (
                f"Step {step_count} - ISO Prices: "
                f"Buy {iso_buy_price:.2f} [{self.buy_price_min}-{self.buy_price_max}], "
                f"Sell {iso_sell_price:.2f} [{self.sell_price_min}-{self.sell_price_max}]"
            )
            if use_dispatch_action:
                log_msg += f", Dispatch: {dispatch:.2f}"
            self.logger.info(log_msg)
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class PricingStrategyFactory:
    """
    Factory class for creating pricing strategy instances.
    
    This factory implements the Factory pattern to create the appropriate
    pricing strategy based on the pricing policy enum value. It encapsulates
    the object creation logic and provides a clean interface for creating
    strategy objects.
    """
    
    @staticmethod
    def create_strategy(
        pricing_policy: PricingPolicy,
        min_price: float,
        max_price: float,
        max_steps_per_episode: int,
        action_spaces_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> PricingStrategy:
        """
        Create the appropriate pricing strategy based on the pricing policy.
        
        Args:
            pricing_policy: The pricing policy enum value
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            action_spaces_config: Configuration for action spaces
            logger: Logger instance for logging
            
        Returns:
            An instance of the appropriate pricing strategy
            
        Raises:
            ValueError: If the pricing policy is not supported
        """
        if pricing_policy == PricingPolicy.QUADRATIC:
            return QuadraticPricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        elif pricing_policy == PricingPolicy.CONSTANT:
            return ConstantPricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        elif pricing_policy == PricingPolicy.ONLINE:
            return OnlinePricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        else:
            if logger:
                logger.error(f"Unsupported pricing policy: {pricing_policy}")
            raise ValueError(f"Unsupported pricing policy: {pricing_policy}")