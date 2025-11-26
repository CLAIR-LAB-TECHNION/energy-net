import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from energy_net.common import defs
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import load_data_from_yaml, interpolate_value

class ProductionDynamics(EnergyDynamics):
    """
    Unified class for defining and handling energy production patterns and dynamics.
    """

    def __init__(self, params: Dict[str, float]):
        """
        Initializes the dynamics with a specific configuration.

        Args:
            params (Dict[str, float]): Configuration dictionary for the pattern.
        """
        self.params = self._validate_and_prepare_params(params)
    def _validate_and_prepare_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Default implementation for validating and preparing parameters.
        Subclasses can override this if specific validation is needed.

        Args:
            params (Dict[str, float]): Input parameters for the dynamics.

        Returns:
            Dict[str, float]: The same parameters without modification.
        """
        return params        

    def reset(self) -> None:
        """
        Resets the internal state of the dynamics.
        """
        pass

    @abstractmethod
    def get_value(self, **kwargs) -> float:
        """
        Retrieves the current production value.

        Args:
            **kwargs:
                - time (float): Current time as a fraction of the day (0.0 to 1.0).

        Returns:
            float: The calculated production value.
        """
        pass



class ConstantProductionDynamics(ProductionDynamics):
    """
    Production pattern with a constant value throughout the day.
    Useful for scenarios where energy usage does not fluctuate with time.
    """

    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)
        return self.params.get('base_load', defs.DEFAULT_PROD_BASE_LOAD)

class DoublePeakProductionDynamics(ProductionDynamics):
    """
    Production pattern with two distinct peaks during the day:
    one in the morning and one in the evening. The sharpness of peaks
    and their amplitude can be adjusted.
    """
    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)
        base_load = self.params.get('base_load', defs.DEFAULT_PROD_BASE_LOAD)
        amplitude = self.params.get('amplitude', defs.DEFAULT_PROD_AMPLITUDE)
        morning_peak = self.params.get('morning_peak', defs.DEFAULT_PROD_MORNING_PEAK)
        evening_peak = self.params.get('evening_peak', defs.DEFAULT_PROD_EVENING_PEAK)
        peak_sharpness = self.params.get('peak_sharpness', defs.DEFAULT_PROD_PEAK_SHARPNESS)

        morning_factor = np.exp(-peak_sharpness * ((time - morning_peak) ** 2))
        evening_factor = np.exp(-peak_sharpness * ((time - evening_peak) ** 2))
        return base_load + amplitude * (morning_factor + evening_factor)

class LinearGrowthProductionDynamics(ProductionDynamics):
    """
    Production pattern that grows linearly over time.
    Models scenarios where energy demand steadily increases during the day.
    """
    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)
        base_load = self.params.get('base_load', defs.DEFAULT_PROD_BASE_LOAD)
        growth_factor = self.params.get('growth_factor', defs.DEFAULT_PROD_GROWTH_FACTOR)
        return base_load + base_load * time * growth_factor


class GMMProductionDynamics(ProductionDynamics):
    """
    Production pattern modeled as a sum of two Gaussian peaks.
    Each peak has a configurable center (peak time), width, and height,
    allowing for flexible modeling of complex daily production profiles.
    """
    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)

        peak_production1 = self.params.get('peak_production1', defs.DEFAULT_PROD_PEAK1)
        peak_time1 = self.params.get('peak_time1', defs.DEFAULT_PROD_PEAK_TIME1)
        width1 = self.params.get('width1', defs.DEFAULT_PROD_WIDTH1)
        peak_production2 = self.params.get('peak_production2', defs.DEFAULT_PROD_PEAK2)
        peak_time2 = self.params.get('peak_time2', defs.DEFAULT_PROD_PEAK_TIME2)
        width2 = self.params.get('width2', defs.DEFAULT_PROD_WIDTH2)

        production1 = peak_production1 * np.exp(-((time - peak_time1) ** 2) / (2 * (width1 ** 2)))
        production2 = peak_production2 * np.exp(-((time - peak_time2) ** 2) / (2 * (width2 ** 2)))
        return production1 + production2


class DataDrivenProductionDynamics(ProductionDynamics):
    """
    Production pattern based on external data loaded from a file.
    Interpolates values between data points and applies an optional scaling factor.
    Useful for replicating measured or historical production profiles.
    """
    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)

        data_file = self.params.get('data_file')
        if not data_file:
            raise ValueError("No data file specified for DATA_DRIVEN pattern")
        scale_factor = self.params.get('scale_factor', defs.DEFAULT_PROD_SCALE_FACTOR)
        production_data = _load_production_data(data_file)
        return _interpolate_production(time, production_data) * scale_factor

def _load_production_data(data_file: str) -> Dict[float, float]:
    """
    Load production data from a YAML file.

    Args:
        data_file: Path to the YAML file.

    Returns:
        Dictionary mapping time fractions to production values.
    """
    return load_data_from_yaml(data_file, defs.PRODUCTION_DATA)


def _interpolate_production(time_fraction: float, production_data: Dict[float, float]) -> float:
    """
    Interpolate the production value based on time fraction.

    Args:
        time_fraction: Current time as a fraction of the day (0.0 to 1.0).
        production_data: Dictionary mapping time fractions to production values.

    Returns:
        Interpolated production value.
    """
    return interpolate_value(time_fraction, production_data)

# Add a new function to get raw production data for a file
def get_raw_production_data(data_file: str) -> Dict[float, float]:
    """
    Get raw production data from a file without any processing.
    This is useful for visualization purposes.

    Args:
        data_file: Path to the YAML file containing production data

    Returns:
        Dictionary mapping time fractions to production values
    """
    return _load_production_data(data_file)