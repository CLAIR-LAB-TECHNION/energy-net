from typing import Dict, Any, Callable
from energy_net.market.iso.iso_base import ISOBase
import numpy as np

class QuadraticPricingISO(ISOBase):
    """
    ISO implementation that uses quadratic pricing function.
    """
    def __init__(self, buy_a: float = 0.0, buy_b: float = 0.0, buy_c: float = 50.0):
        """
        Args:
            buy_a (float): Quadratic coefficient
            buy_b (float): Linear coefficient
            buy_c (float): Constant term
        """
        self.buy_a = buy_a
        self.buy_b = buy_b
        self.buy_c = buy_c
        
    def get_pricing_function(self, state: Dict[str, Any]) -> Callable[[float], float]:
        """
        Returns a quadratic pricing function.
        
        Args:
            state (Dict[str, Any]): Current state (not used in this implementation)
            
        Returns:
            Callable[[float], float]: Pricing function that takes demand and returns price
        """
        def price_fn(demand: float) -> float:
            return self.buy_a * demand**2 + self.buy_b * demand + self.buy_c
            
        return price_fn
