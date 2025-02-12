from typing import Callable, Dict
from energy_net.dynamics.iso.iso_base import ISOBase

class DynamicPricingISO(ISOBase):
    """
    ISO implementation that adjusts prices dynamically based on real-time factors.
    """

    def __init__(
        self,
        base_price: float = 50.0,
        demand_factor: float = 1.0,
        supply_factor: float = 1.0,
        elasticity: float = 0.5
    ):
        self.base_price = base_price
        self.demand_factor = demand_factor
        self.supply_factor = supply_factor
        self.elasticity = elasticity

    def reset(self) -> None:
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float], float]:
        demand = observation.get('demand', 1.0)
        supply = observation.get('supply', 1.0)

        price = self.base_price * (1 + self.elasticity * (demand - supply))
        

        def pricing(buy: float) -> float:
            return (buy * price) 

        # Ensure we return the pricing callable
        return pricing
