from typing import Dict, Any, Callable
from energy_net.market.iso.iso_base import ISOBase
import numpy as np

class TimeOfUsePricingISO(ISOBase):
    """
    ISO implementation that uses time-of-use pricing.
    """
    def __init__(self, peak_price: float = 80.0, off_peak_price: float = 30.0, 
                 peak_start: float = 0.25, peak_end: float = 0.75):
        """
        Args:
            peak_price (float): Price during peak hours
            off_peak_price (float): Price during off-peak hours
            peak_start (float): Start of peak period (fraction of day)
            peak_end (float): End of peak period (fraction of day)
        """
        self.peak_price = peak_price
        self.off_peak_price = off_peak_price
        self.peak_start = peak_start
        self.peak_end = peak_end
        
    def get_pricing_function(self, state: Dict[str, Any]) -> Callable[[float], float]:
        """
        Returns a pricing function based on time of day.
        
        Args:
            state (Dict[str, Any]): Current state including time
            
        Returns:
            Callable[[float], float]: Pricing function that takes demand and returns price
        """
        time = state.get('time', 0.0)
        is_peak = self.peak_start <= time <= self.peak_end
        price = self.peak_price if is_peak else self.off_peak_price
        
        def price_fn(demand: float) -> float:
            return price
            
        return price_fn
