from typing import Callable, Dict
from energy_net.dynamics.iso.iso_base import ISOBase

class QuadraticPricingISO(ISOBase):
    """
    ISO implementation that uses a quadratic function to determine prices based on demand or other factors.
    """

    def __init__(self, buy_a: float = 1.0, buy_b: float = 0.0, buy_c: float = 50.0):
        if not isinstance(buy_a, (float, int)):
            raise TypeError(f"a must be a float or int, got {type(buy_a).__name__}")
        if not isinstance(buy_b, (float, int)):
            raise TypeError(f"b must be a float or int, got {type(buy_b).__name__}")
        if not isinstance(buy_c, (float, int)):
            raise TypeError(f"c must be a float or int, got {type(buy_c).__name__}")

        self.a = float(buy_a)
        self.b = float(buy_b)
        self.c = float(buy_c)

    def reset(self) -> None:
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float], float]:
        demand = observation.get('demand', 1.0)
        price = self.a * (demand ** 2) + self.b * demand + self.c
        

        def pricing(buy: float) -> float:
            return (buy * price) 

        return pricing
