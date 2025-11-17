# dynamics/energy_dynamics.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

class EnergyDynamics(ABC):
    """
    Abstract Base Class for defining energy dynamics.
    Represents the behavior of a component in the PCSUnit.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state of the dynamics.
        Called at the beginning of each new episode.
        """
        pass

    @abstractmethod
    def get_value(self, **kwargs) -> Any:
        """
        Retrieves the current value based on provided arguments.

        Args:
            **kwargs: Arbitrary keyword arguments specific to the dynamic.

        Returns:
            Any: The current value based on the dynamics.
        """
        pass
