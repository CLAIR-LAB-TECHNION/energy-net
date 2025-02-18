from typing import Callable, Dict
from energy_net.dynamics.iso.iso_base import ISOBase

class TimeOfUsePricingISO(ISOBase):
    """
    ISO implementation that sets prices based on time-of-use (TOU) periods.
    """

    def __init__(
        self,
        peak_hours: list,
        off_peak_hours: list,
        peak_price: float = 60.0,
        off_peak_price: float = 30.0
    ):
        self.peak_hours = peak_hours
        self.off_peak_hours = off_peak_hours
        self.peak_price = peak_price
        self.off_peak_price = off_peak_price

    def reset(self) -> None:
        pass

    def get_pricing_function(self, observation: Dict) -> Callable[[float], float]:
        current_time_fraction = observation.get('time', 0.0)
        current_hour = int(current_time_fraction * 24) % 24

        if current_hour in self.peak_hours:
            price = self.peak_price
            
        elif current_hour in self.off_peak_hours:
            price = self.off_peak_price
            
        else:
            price = (self.peak_price + self.off_peak_price) / 2
            

        def pricing(buy: float) -> float:
            return buy * price

        return pricing
