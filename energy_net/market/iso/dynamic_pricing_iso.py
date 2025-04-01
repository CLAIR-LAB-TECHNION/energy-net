from typing import Dict, Any, Callable
from energy_net.market.iso.iso_base import ISOBase
import numpy as np

class DynamicPricingISO(ISOBase):
    """
    ISO implementation that uses dynamic pricing based on demand.
    """
    def __init__(self, base_price: float = 50.0, demand_sensitivity: float = 0.1):
        """
        Args:
            base_price (float): Base price when demand equals supply
            demand_sensitivity (float): How much price changes with demand
        """
        self.base_price = base_price
        self.demand_sensitivity = demand_sensitivity
        
    def get_pricing_function(self, state: Dict[str, Any]) -> Callable[[float], float]:
        """
        Returns a pricing function that increases with demand.
        
        Args:
            state (Dict[str, Any]): Current state including predicted demand
            
        Returns:
            Callable[[float], float]: Pricing function that takes demand and returns price
        """
        predicted_demand = state.get('predicted_demand', 0.0)
        
        def price_fn(demand: float) -> float:
            return self.base_price * (1 + self.demand_sensitivity * (demand - predicted_demand))
            
        return price_fn
