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
    
    
    
class ModelBasedDynamics(EnergyDynamics):
    """
    Abstract Model-Based Dynamics class.
    Defines behavior through predefined mathematical models.
    Specific dynamics should inherit from this class and implement the get_value method.
    """

    def __init__(self, model_parameters: Dict[str, Any]):
        """
        Initializes the ModelBasedDynamics with specific model parameters.

        Args:
            model_parameters (Dict[str, Any]): Parameters defining the model behavior.
        """
        self.model_parameters = model_parameters

    def reset(self) -> None:
        """
        Resets the internal state of the model-based dynamics.
        """
        # Implement reset logic if necessary
        pass

    @abstractmethod
    def get_value(self, **kwargs) -> Any:
        """
        Retrieves the current value based on a predefined mathematical model.

        Args:
            **kwargs: Arbitrary keyword arguments specific to the dynamic.

        Returns:
            Any: The current value based on the model.
        """
        pass
    
    
    
class DataDrivenDynamics(EnergyDynamics):
    """
    Data-Driven Dynamics implementation.
    Defines behavior based on data from external sources.
    """

    def __init__(self, data_file: str, value_column: str):
        """
        Initializes the DataDrivenDynamics with data from a file.

        Args:
            data_file (str): Path to the data file (e.g., CSV).
            value_column (str): Name of the column to retrieve values from.
        """
        self.data = pd.read_csv(data_file)
        self.value_column = value_column
        self.current_index = 0

    def reset(self) -> None:
        """
        Resets the internal state of the data-driven dynamics.
        """
        self.current_index = 0

    def get_value(self, **kwargs) -> Any:
        """
        Retrieves the value from the data corresponding to the current time.

        Args:
            **kwargs: Should contain 'time' as a fraction of the day (0 to 1).

        Returns:
            Any: The value from the data corresponding to the current time.
        """
        time = kwargs.get('time', 0.0)
        total_steps = len(self.data)
        index = int(time * total_steps) % total_steps
        self.current_index = index
        value = self.data.iloc[index][self.value_column]
        return value
