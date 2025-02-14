# energy_net/dynamics/iso/fixed_pricing_iso.py

from typing import Callable, Dict, List, Union
from energy_net.dynamics.iso.iso_base import ISOBase

class FixedPricingISO(ISOBase):
    """
    ISO with a fixed pricing policy. Pricing is determined at the start of the episode
    and remains constant throughout the episode.
    """

    def __init__(self, pricing_schedule: List[Union[float, int]]):
        """
        Initializes the FixedPricingISO with a predefined pricing schedule.

        Args:
            pricing_schedule (List[Union[float, int]]): List of prices for each time step in the episode.

        Raises:
            TypeError: If pricing_schedule is not a list.
            ValueError: If any element in pricing_schedule is not a number (int or float).
        """
        # Type check: pricing_schedule must be a list
        if not isinstance(pricing_schedule, list):
            raise TypeError(f"'pricing_schedule' must be a list, got {type(pricing_schedule).__name__} instead.")

        # Value check: all elements in pricing_schedule must be int or float
        if not all(isinstance(price, (int, float)) for price in pricing_schedule):
            invalid_types = {type(price).__name__ for price in pricing_schedule if not isinstance(price, (int, float))}
            raise ValueError(f"All elements in 'pricing_schedule' must be int or float. Invalid types found: {invalid_types}")

        self.pricing_schedule = pricing_schedule
        self.episode_length = len(pricing_schedule)
        self.current_step = 0

    def reset(self) -> None:
        """
        Resets the ISO's internal state.
        Called at the beginning of each new episode.
        """
        self.current_step = 0

    def get_pricing_function(self, observation: Dict) -> Callable[[float], float]:
        """
        Returns a pricing function based on the current observation.
        The pricing function calculates the reward given buy and sell amounts.

        Args:
            observation (Dict): Current state observation containing relevant information.

        Returns:
            Callable[[float], float]: A function that takes an action amount
                                      and returns the fixed price for the current step.
        """
        def pricing_function(action_amount: float) -> float:
            if self.current_step < self.episode_length:
                price = self.pricing_schedule[self.current_step]
                self.current_step += 1
                return float(price)
            else:
                # If current_step exceeds the schedule, return the last price
                return float(self.pricing_schedule[-1])

        return pricing_function
