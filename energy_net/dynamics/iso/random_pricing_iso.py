import random
from typing import Callable, Dict
from energy_net.dynamics.iso.iso_base import ISOBase

class RandomPricingISO(ISOBase):
    """
    ISO implementation that generates random prices within a specified range.
    """

    def __init__(self, min_price: float = 40.0, max_price: float = 60.0):
        self.min_price = min_price
        self.max_price = max_price

    def reset(self) -> None:
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float], float]:
        price = random.uniform(self.min_price, self.max_price)
        

        def pricing(buy: float) -> float:
            return buy * price

        return pricing
