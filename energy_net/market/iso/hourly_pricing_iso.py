from typing import Callable, Dict
from energy_net.dynamics.iso.iso_base import ISOBase

class HourlyPricingISO(ISOBase):
    """
    ISO implementation that sets prices based on predefined hourly rates.
    """

    def __init__(self, hourly_rates: Dict[int, float]):
        self.hourly_rates = hourly_rates

    def reset(self) -> None:
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float], float]:
        current_time_fraction = observation.get('time', 0.0)
        current_hour = int(current_time_fraction * 24) % 24

        price = self.hourly_rates.get(current_hour, 50.0)
        

        def pricing(buy: float) -> float:
            return buy * price

        return pricing
