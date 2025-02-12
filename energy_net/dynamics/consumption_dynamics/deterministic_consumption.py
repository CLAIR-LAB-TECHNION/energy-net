# dynamics/deterministic_consumption.py

from typing import Any, Dict
from energy_net.dynamics.energy_dynamcis import ModelBasedDynamics
import math


class DeterministicConsumption(ModelBasedDynamics):
    """
    Deterministic Consumption Dynamics.
    Consumption has two peaks: midday and evening.
    """

    def __init__(self, model_parameters: Dict[str, Any]):
        """
        Initializes the DeterministicConsumption with specific model parameters.

        Args:
            model_parameters (Dict[str, Any]):
                - peak_consumption1 (float): First peak consumption (MWh).
                - peak_time1 (float): Time of first peak as a fraction of the day (0 to 1).
                - width1 (float): Width of the first consumption peak.
                - peak_consumption2 (float): Second peak consumption (MWh).
                - peak_time2 (float): Time of second peak as a fraction of the day (0 to 1).
                - width2 (float): Width of the second consumption peak.
        """
        super().__init__(model_parameters)
        
        # Ensure all required parameters are provided
        required_params = [
            'peak_consumption1', 'peak_time1', 'width1',
            'peak_consumption2', 'peak_time2', 'width2'
        ]
        for param in required_params:
            assert param in model_parameters, f"Missing required parameter '{param}' for DeterministicConsumption."

    def get_value(self, **kwargs) -> float:
        """
        Calculates consumption based on the time of day.

        Args:
            **kwargs:
                - time (float): Current time as a fraction of the day (0 to 1).

        Returns:
            float: Consumption value in MWh.
        """
        time: float = kwargs.get('time')
        assert time is not None, "Time parameter is required for DeterministicConsumption."

        peak_consumption1: float = self.model_parameters['peak_consumption1']
        peak_time1: float = self.model_parameters['peak_time1']
        width1: float = self.model_parameters['width1']
        peak_consumption2: float = self.model_parameters['peak_consumption2']
        peak_time2: float = self.model_parameters['peak_time2']
        width2: float = self.model_parameters['width2']

        # Using Gaussian functions for both peaks and summing them
        consumption1 = peak_consumption1 * math.exp(-((time - peak_time1) ** 2) / (2 * (width1 ** 2)))
        consumption2 = peak_consumption2 * math.exp(-((time - peak_time2) ** 2) / (2 * (width2 ** 2)))
        total_consumption = consumption1 + consumption2
        return total_consumption
