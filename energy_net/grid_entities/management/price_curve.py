from abc import ABC, abstractmethod
import numpy as np


class PriceCurveStrategy(ABC):
    """
    Abstract base class for different price curve strategies.

    Subclasses can either:
    1. Implement calculate_price() for symmetric buy/sell pricing
    2. Implement calculate_buy_price() AND calculate_sell_price() for asymmetric pricing
    """

    def calculate_price(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Default implementation that returns the same price for buy and sell.
        Override this for symmetric pricing, OR override both buy/sell methods.
        """
        raise NotImplementedError(
            "Must implement either calculate_price() or both "
            "calculate_buy_price() and calculate_sell_price()"
        )

    def calculate_buy_price(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Calculate buying price curve. Defaults to calculate_price().
        Override for asymmetric buy/sell pricing.
        """
        return self.calculate_price(observation)

    def calculate_sell_price(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Calculate selling price curve. Defaults to calculate_price().
        Override for asymmetric buy/sell pricing.
        """
        return self.calculate_price(observation)


class ActionBasedPriceStrategy(PriceCurveStrategy):
    """
    A strategy that uses a pre-computed action (from RL agent) to generate price curves.
    This fixes the training issue where RLPriceCurveStrategy re-queried the model,
    bypassing the exploratory action that the RL agent was trying to learn from.
    
    Use this during training when you have the action available.
    Use RLPriceCurveStrategy for evaluation when you need to query the model.
    """

    def __init__(self, action, price_min=0.0, price_max=0.20, 
                 use_asymmetric_pricing=False):
        """
        Args:
            action: The RL agent's action (already in [0, 1] range).
                - For symmetric pricing: action should have 48+ values
                - For asymmetric pricing: action should have 96+ values (48 buy, 48 sell)
            price_min (float): The minimum price in $/unit (when action=0).
            price_max (float): The maximum price in $/unit (when action=1).
            use_asymmetric_pricing (bool): If True, uses separate buy and sell prices.
        """
        self.action = np.asarray(action, dtype=np.float32)
        self.price_min = price_min
        self.price_max = price_max
        self.use_asymmetric_pricing = use_asymmetric_pricing

    def calculate_price(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Generates a 48-step price curve from the action.
        For symmetric pricing, uses first 48 values of action.
        """
        # Extract first 48 values (price signals in [0, 1])
        raw_prices = self.action[:48]
        
        # Scale from [0, 1] to [price_min, price_max]
        scaled_prices = self.price_min + (raw_prices * (self.price_max - self.price_min))
        
        return scaled_prices.astype(np.float32)
    
    def calculate_buy_price(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Calculate the price at which ISO buys energy from PCS.
        
        If asymmetric pricing is disabled, returns the same as calculate_price().
        If enabled, uses the first 48 action values as buy prices.
        """
        if not self.use_asymmetric_pricing:
            return self.calculate_price(observation)
        
        # Extract buy prices from first 48 action values
        raw_buy_prices = self.action[:48]
        
        # Scale from [0, 1] to [price_min, price_max]
        buy_prices = self.price_min + (raw_buy_prices * (self.price_max - self.price_min))
        
        return buy_prices.astype(np.float32)
    
    def calculate_sell_price(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Calculate the price at which ISO sells energy to PCS.
        
        If asymmetric pricing is disabled, returns the same as calculate_price().
        If enabled, uses the second 48 action values as sell prices.
        """
        if not self.use_asymmetric_pricing:
            return self.calculate_price(observation)
        
        # Extract sell prices from second 48 action values
        raw_sell_prices = self.action[48:96]
        
        # Scale from [0, 1] to [price_min, price_max]
        sell_prices = self.price_min + (raw_sell_prices * (self.price_max - self.price_min))
        
        return sell_prices.astype(np.float32)


class RLPriceCurveStrategy(PriceCurveStrategy):
    """
    A strategy that uses a trained ISO RL model to generate price curves.
    This replaces the old ISOPricingWrapper logic.
    
    Supports both symmetric (same buy/sell price) and asymmetric (separate buy/sell prices).
    
    NOTE: This should be used for EVALUATION only, not during training.
    During training, use ActionBasedPriceStrategy to avoid re-querying the model.
    """

    def __init__(self, iso_model, price_min=0.0, price_max=0.20, 
                 use_asymmetric_pricing=False):
        """
        Args:
            iso_model: The trained RL agent (e.g., PPO/SAC) representing the ISO.
            price_min (float): The minimum price in $/unit (when agent outputs 0).
            price_max (float): The maximum price in $/unit (when agent outputs 1).
            use_asymmetric_pricing (bool): If True, ISO learns separate buy and sell prices.
                - When False: ISO outputs 48 values used for both buy and sell (symmetric)
                - When True: ISO outputs 96 values - first 48 for buy prices, second 48 for sell prices
                Default is False for backward compatibility.
        """
        self.iso_model = iso_model
        self.price_min = price_min
        self.price_max = price_max
        self.use_asymmetric_pricing = use_asymmetric_pricing

    def calculate_price(self, observation: np.ndarray) -> np.ndarray:
        """
        Generates a 48-step price curve by querying the ISO model.

        Args:
            observation: The ISO's observation vector (predictions + features).

        Returns:
            np.ndarray: A 48-element array of scaled prices in $/unit.
        """
        # 1. Ensure the observation is in the correct format for the model
        obs = observation.astype(np.float32)

        # 2. Query the ISO policy deterministically to get the action
        # The ISO action contains [48 prices + 48 dispatch values]
        # ISO action space is [0, 1], so raw_prices will be in that range
        action, _ = self.iso_model.predict(obs, deterministic=True)

        # 3. Extract the first 48 entries (price signals already in [0, 1])
        raw_prices = action[:48]

        # 4. Direct linear scaling from [0, 1] to [price_min, price_max]
        # No min-max normalization - agent learns to output appropriate values
        scaled_prices = self.price_min + (raw_prices * (self.price_max - self.price_min))

        return scaled_prices.astype(np.float32)
    
    def calculate_buy_price(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Calculate the price at which ISO buys energy from PCS.
        
        If asymmetric pricing is disabled, returns the same as calculate_price().
        If enabled, uses the first 48 action outputs as independent buy prices.
        
        Args:
            observation: The ISO's observation vector.
            
        Returns:
            np.ndarray: A 48-element array of buy prices in $/unit.
        """
        if not self.use_asymmetric_pricing:
            return self.calculate_price(observation)
        
        # When asymmetric: ISO learns separate buy prices (first 48 of 96 total outputs)
        obs = observation.astype(np.float32)
        action, _ = self.iso_model.predict(obs, deterministic=True)
        
        # Extract buy prices from first 48 action values
        raw_buy_prices = action[:48]
        
        # Scale from [0, 1] to [price_min, price_max]
        buy_prices = self.price_min + (raw_buy_prices * (self.price_max - self.price_min))
        
        return buy_prices.astype(np.float32)
    
    def calculate_sell_price(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Calculate the price at which ISO sells energy to PCS.
        
        If asymmetric pricing is disabled, returns the same as calculate_price().
        If enabled, uses the second 48 action outputs as independent sell prices.
        
        Args:
            observation: The ISO's observation vector.
            
        Returns:
            np.ndarray: A 48-element array of sell prices in $/unit.
        """
        if not self.use_asymmetric_pricing:
            return self.calculate_price(observation)
        
        # When asymmetric: ISO learns separate sell prices (second 48 of 96 total outputs)
        obs = observation.astype(np.float32)
        action, _ = self.iso_model.predict(obs, deterministic=True)
        
        # Extract sell prices from second 48 action values
        raw_sell_prices = action[48:96]
        
        # Scale from [0, 1] to [price_min, price_max]
        sell_prices = self.price_min + (raw_sell_prices * (self.price_max - self.price_min))
        
        return sell_prices.astype(np.float32)
class SineWavePriceStrategy(PriceCurveStrategy):
    """
    Reproduces the original sine-wave pricing logic from PCSEnv.
    """
    def __init__(self, base_price=0.10, price_volatility=0.15, steps_per_day=48):
        self.base_price = base_price
        self.price_volatility = price_volatility
        self.T = steps_per_day

    def calculate_price(self, observation: np.ndarray = None) -> np.ndarray:
        """Generates a sine-wave price curve for one day (48 steps)."""
        prices = []
        dt = 1.0 / self.T
        for i in range(self.T):
            time_of_day = (i * dt * 24) % 24
            variation = np.sin(2 * np.pi * time_of_day / 24) * self.price_volatility
            price = self.base_price + variation
            prices.append(max(0.01, price))
        return np.array(prices, dtype=np.float32)

class FixedPriceStrategy(PriceCurveStrategy):
    """
    Generates a fixed price curve.
    """
    def __init__(self, fixed_price=0.10, steps_per_day=48):
        self.fixed_price = fixed_price
        self.steps_per_day = steps_per_day

    def calculate_price(self, observation: np.ndarray = None) -> np.ndarray:
        """Generates a fixed price curve for one day (48 steps)."""
        return np.full(self.steps_per_day, self.fixed_price, dtype=np.float32)