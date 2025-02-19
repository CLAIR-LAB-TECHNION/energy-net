# iso_base.py

from abc import ABC, abstractmethod
from typing import Callable, Dict


class ISOBase(ABC):
    """
    Abstract Base Class for ISO (Independent System Operator) implementations.
    Defines the required interface for all ISO pricing strategies.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the ISO's internal state.
        Called at the beginning of each new episode.
        """
        pass

    @abstractmethod
    def get_pricing_function(self, observation: Dict) -> Callable[[float], float]:
        """
        Returns a pricing function based on the current observation.
        The pricing function calculates the reward given buy and sell amounts.

        Args:
            observation (Dict): Current state observation containing relevant information.

        Returns:
            Callable[[float, float], float]: A function that takes buy and sell amounts
                                              and returns the calculated reward.
        """
        pass


