from typing import Dict, Any, Callable
from energy_net.market.iso.iso_base import ISOBase
import numpy as np

class HourlyPricingISO(ISOBase):
    """
    ISO implementation that uses hourly rates for pricing.
    """
    def __init__(self, hourly_rates: Dict[int, float]):
        """
        Args:
            hourly_rates (Dict[int, float]): Dictionary mapping hour (0-23) to price rate
        """
        self.hourly_rates = hourly_rates
        
    def get_pricing_function(self, state: Dict[str, Any]) -> Callable[[float], float]:
        """
        Returns a pricing function based on the current hour.
        
        Args:
            state (Dict[str, Any]): Current state including time
            
        Returns:
            Callable[[float], float]: Pricing function that takes demand and returns price
        """
        hour = int((state.get('time', 0.0) * 24) % 24)
        rate = self.hourly_rates.get(hour, 0.0)
        
        def price_fn(demand: float) -> float:
            return rate
            
        return price_fn
