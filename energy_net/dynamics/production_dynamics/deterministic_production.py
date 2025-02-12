# dynamics/deterministic_production.py

from typing import Any, Dict
from energy_net.dynamics.energy_dynamcis import ModelBasedDynamics
import math


class DeterministicProduction(ModelBasedDynamics):
    """
    Deterministic Production Dynamics.
    Production peaks at midday and decreases towards evening.
    """

    def __init__(self, model_parameters: Dict[str, Any]):
        """
        Initializes the DeterministicProduction with specific model parameters.

        Args:
            model_parameters (Dict[str, Any]):
                - peak_production (float): Maximum production capacity (MWh).
                - peak_time (float): Time of peak production as a fraction of the day (0 to 1).
                - width (float): Width of the production peak (controls how quickly production ramps up/down).
        """
        super().__init__(model_parameters)
        
        # Ensure all required parameters are provided
        required_params = ['peak_production', 'peak_time', 'width']
        for param in required_params:
            assert param in model_parameters, f"Missing required parameter '{param}' for DeterministicProduction."

    def get_value(self, **kwargs) -> float:
        """
        Calculates production based on the time of day.

        Args:
            **kwargs:
                - time (float): Current time as a fraction of the day (0 to 1).

        Returns:
            float: Production value in MWh.
        """
        time: float = kwargs.get('time')
        assert time is not None, "Time parameter is required for DeterministicProduction."

        peak_production: float = self.model_parameters['peak_production']
        peak_time: float = self.model_parameters['peak_time']
        width: float = self.model_parameters['width']

        # Using a Gaussian function to model the production peak
        production = peak_production * math.exp(-((time - peak_time) ** 2) / (2 * (width ** 2)))
        return production
