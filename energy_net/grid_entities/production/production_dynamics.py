import numpy as np
from typing import Dict, Any, Optional, Tuple
from energy_net import defs
from energy_net.dynamics import EnergyDynamics
from energy_net.utils import load_data_from_yaml, interpolate_value

class ProductionDynamics(EnergyDynamics):
    """
    Unified class for defining and handling energy production patterns and dynamics.
    """

    def __init__(self, config: Dict[str, float]):
        """
        Initializes the dynamics with a specific configuration.

        Args:
            config (Dict[str, float]): Configuration dictionary for the pattern.
        """
        self.config = self._validate_and_prepare_config(config)

    def _validate_and_prepare_config(self, config: Dict[str, float]) -> Dict[str, float]:
        """
        Validates and preprocesses the configuration dictionary.

        Args:
            config (Dict[str, float]): The raw configuration dictionary.

        Returns:
            Dict[str, float]: The validated and preprocessed configuration.
        """
        config.setdefault('base_load', 100.0)
        config.setdefault('amplitude', 50.0)
        return config

    def calculate(self, time: float) -> float:
        """
        Calculate the production value based on time.

        Args:
            time (float): Current time as a fraction of the day (0.0 to 1.0).

        Returns:
            float: Calculated production value.
        """
        raise NotImplementedError("Subclasses must implement the `calculate` method.")

    def reset(self) -> None:
        """
        Resets the internal state of the dynamics.
        """
        pass

    def calculate(self, **kwargs) -> float:
        """
        Retrieves the current production value.

        Args:
            **kwargs:
                - time (float): Current time as a fraction of the day (0.0 to 1.0).

        Returns:
            float: The calculated production value.
        """
        time = kwargs.get('time', 0.0)
        return self.calculate(time)
class SinusoidalProductionDynamics(ProductionDynamics):
    """
    Production pattern modeled as a sinusoidal function over time.
    Represents daily fluctuations in production with configurable amplitude,
    phase, and period to mimic cyclical energy generation patterns.
    """
    def calculate(self, time: float) -> float:
        base_load = self.config.get('base_load', defs.DEFAULT_PROD_BASE_LOAD)
        amplitude = self.config.get('amplitude', defs.DEFAULT_PROD_AMPLITUDE)
        interval_multiplier = self.config.get('interval_multiplier', defs.DEFAULT_PROD_INTERVAL_MULTIPLIER)
        period_divisor = self.config.get('period_divisor', defs.DEFAULT_PROD_PERIOD_DIVISOR)
        phase_shift = self.config.get('phase_shift', defs.DEFAULT_PROD_PHASE_SHIFT)
        interval = time * interval_multiplier
        return base_load + amplitude * np.cos((interval + phase_shift) * np.pi / period_divisor)


class ConstantProductionDynamics(ProductionDynamics):
    """
    Production pattern with a constant output throughout the day.
    Useful for modeling generators or sources with stable production rates.
    """
    def calculate(self, time: float) -> float:
        return self.config.get('base_load', defs.DEFAULT_PROD_BASE_LOAD)


class DoublePeakProductionDynamics(ProductionDynamics):
    """
    Production pattern with two distinct peaks during the day:
    one in the morning and one in the evening. Peak timing, amplitude,
    and sharpness can be configured to model real-world variations.
    """
    def calculate(self, time: float) -> float:
        base_load = self.config.get('base_load', defs.DEFAULT_PROD_BASE_LOAD)
        amplitude = self.config.get('amplitude', defs.DEFAULT_PROD_AMPLITUDE)
        morning_peak = self.config.get('morning_peak', defs.DEFAULT_PROD_MORNING_PEAK)
        evening_peak = self.config.get('evening_peak', defs.DEFAULT_PROD_EVENING_PEAK)
        peak_sharpness = self.config.get('peak_sharpness', defs.DEFAULT_PROD_PEAK_SHARPNESS)

        morning_factor = np.exp(-peak_sharpness * ((time - morning_peak) ** 2))
        evening_factor = np.exp(-peak_sharpness * ((time - evening_peak) ** 2))
        return base_load + amplitude * (morning_factor + evening_factor)


class LinearGrowthProductionDynamics(ProductionDynamics):
    """
    Production pattern that increases linearly over time.
    Useful for modeling ramp-up of generators or production sources
    where output grows steadily during the day.
    """
    def calculate(self, time: float) -> float:
        base_load = self.config.get('base_load', defs.DEFAULT_PROD_BASE_LOAD)
        growth_factor = self.config.get('growth_factor', defs.DEFAULT_PROD_GROWTH_FACTOR)
        return base_load + base_load * time * growth_factor


class GMMProductionDynamics(ProductionDynamics):
    """
    Production pattern modeled as a sum of two Gaussian peaks.
    Each peak has a configurable center (peak time), width, and height,
    allowing flexible modeling of complex production profiles.
    """
    def calculate(self, time: float) -> float:
        peak_production1 = self.config.get('peak_production1', defs.DEFAULT_PROD_PEAK1)
        peak_time1 = self.config.get('peak_time1', defs.DEFAULT_PROD_PEAK_TIME1)
        width1 = self.config.get('width1', defs.DEFAULT_PROD_WIDTH1)
        peak_production2 = self.config.get('peak_production2', defs.DEFAULT_PROD_PEAK2)
        peak_time2 = self.config.get('peak_time2', defs.DEFAULT_PROD_PEAK_TIME2)
        width2 = self.config.get('width2', defs.DEFAULT_PROD_WIDTH2)

        prod1 = peak_production1 * np.exp(-((time - peak_time1) ** 2) / (2 * (width1 ** 2)))
        prod2 = peak_production2 * np.exp(-((time - peak_time2) ** 2) / (2 * (width2 ** 2)))
        return prod1 + prod2


class DataDrivenProductionDynamics(ProductionDynamics):
    """
    Production pattern based on external data loaded from a file.
    Interpolates production values between data points and applies an optional
    scaling factor. Useful for replicating measured or historical production profiles.
    """
    def calculate(self, time: float) -> float:
        data_file = self.config.get('data_file')
        if not data_file:
            raise ValueError("No data file specified for DATA_DRIVEN pattern")
        scale_factor = self.config.get('scale_factor', defs.DEFAULT_PROD_SCALE_FACTOR)
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