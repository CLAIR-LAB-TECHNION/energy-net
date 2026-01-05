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


class RLPriceCurveStrategy(PriceCurveStrategy):
    """
    A strategy that uses a trained ISO RL model to generate price curves.
    This replaces the old ISOPricingWrapper logic.
    """

    def __init__(self, iso_model, base_price=0.10, price_scale=0.20):
        """
        Args:
            iso_model: The trained RL agent (e.g., PPO/SAC) representing the ISO.
            base_price (float): The center point of the generated prices.
            price_scale (float): The total range (max - min) of the prices.
        """
        self.iso_model = iso_model
        self.base_price = base_price
        self.price_scale = price_scale

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
        action, _ = self.iso_model.predict(obs, deterministic=True)

        # 3. Extract the first 48 entries (raw price signals)
        raw_prices = action[:48]

        # 4. Perform Min-Max Normalization to scale raw signals to [0, 1]
        p_min, p_max = raw_prices.min(), raw_prices.max()
        denom = (p_max - p_min) + 1e-8  # Prevent division by zero
        normalized = (raw_prices - p_min) / denom

        # 5. Rescale to the realistic price range defined in __init__
        # Range will be [base - scale/2, base + scale/2]
        scaled_prices = (self.base_price - self.price_scale / 2) + (normalized * self.price_scale)

        return scaled_prices.astype(np.float32)
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