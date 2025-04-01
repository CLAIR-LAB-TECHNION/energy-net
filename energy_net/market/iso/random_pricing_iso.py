from typing import Dict, Any, Callable
from energy_net.market.iso.iso_base import ISOBase
import numpy as np

class RandomPricingISO(ISOBase):
    """
    ISO implementation that uses random pricing within bounds.
    """
    def __init__(self, min_price: float = 10.0, max_price: float = 100.0, seed: int = None):
        """
        Args:
            min_price (float): Minimum possible price
            max_price (float): Maximum possible price
            seed (int): Random seed for reproducibility
        """
        self.min_price = min_price
        self.max_price = max_price
        self.rng = np.random.default_rng(seed)
        
    def get_pricing_function(self, state: Dict[str, Any]) -> Callable[[float], float]:
        """
        Returns a random pricing function.
        
        Args:
            state (Dict[str, Any]): Current state (not used in this implementation)
            
        Returns:
            Callable[[float], float]: Pricing function that takes demand and returns price
        """
        price = self.rng.uniform(self.min_price, self.max_price)
        
        def price_fn(demand: float) -> float:
            return price
            
        return price_fn
