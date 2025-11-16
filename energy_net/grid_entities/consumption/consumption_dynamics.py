import numpy as np
from typing import Dict, Any, Optional, Tuple
from energy_net import defs
from energy_net.dynamics import EnergyDynamics
from energy_net.utils import load_data_from_yaml, interpolate_value

class ConsumptionDynamics(EnergyDynamics):
    """
    Unified class for defining and handling energy consumption patterns and dynamics.
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
        Calculate the consumption value based on time.

        Args:
            time (float): Current time as a fraction of the day (0.0 to 1.0).

        Returns:
            float: Calculated consumption value.
        """
        raise NotImplementedError("Subclasses must implement the `calculate` method.")

    def reset(self) -> None:
        """
        Resets the internal state of the dynamics.
        """
        pass

    def get_value(self, **kwargs) -> float:
        """
        Retrieves the current consumption value.

        Args:
            **kwargs:
                - time (float): Current time as a fraction of the day (0.0 to 1.0).

        Returns:
            float: The calculated consumption value.
        """
        time = kwargs.get('time', 0.0)
        return self.calculate(time)
class SinusoidalConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern modeled as a sinusoidal function over time.
    Represents daily fluctuations with configurable amplitude, phase,
    and period to mimic cyclical energy usage patterns.
    """
    def calculate(self, time: float) -> float:
        base_load = self.config.get('base_load', defs.DEFAULT_CONS_BASE_LOAD)
        amplitude = self.config.get('amplitude', defs.DEFAULT_CONS_AMPLITUDE)
        interval_multiplier = self.config.get('interval_multiplier', defs.DEFAULT_CONS_INTERVAL_MULTIPLIER)
        period_divisor = self.config.get('period_divisor', defs.DEFAULT_CONS_PERIOD_DIVISOR)
        phase_shift = self.config.get('phase_shift', defs.DEFAULT_CONS_PHASE_SHIFT)
        interval = time * interval_multiplier
        return base_load + amplitude * np.cos((interval + phase_shift) * np.pi / period_divisor)


class ConstantConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern with a constant value throughout the day.
    Useful for scenarios where energy usage does not fluctuate with time.
    """
    def calculate(self, time: float) -> float:
        return self.config.get('base_load', defs.DEFAULT_CONS_BASE_LOAD)


class DoublePeakConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern with two distinct peaks during the day:
    one in the morning and one in the evening. The sharpness of peaks
    and their amplitude can be adjusted.
    """
    def calculate(self, time: float) -> float:
        base_load = self.config.get('base_load', defs.DEFAULT_CONS_BASE_LOAD)
        amplitude = self.config.get('amplitude', defs.DEFAULT_CONS_AMPLITUDE)
        morning_peak = self.config.get('morning_peak', defs.DEFAULT_CONS_MORNING_PEAK)
        evening_peak = self.config.get('evening_peak', defs.DEFAULT_CONS_EVENING_PEAK)
        peak_sharpness = self.config.get('peak_sharpness', defs.DEFAULT_CONS_PEAK_SHARPNESS)

        morning_factor = np.exp(-peak_sharpness * ((time - morning_peak) ** 2))
        evening_factor = np.exp(-peak_sharpness * ((time - evening_peak) ** 2))
        return base_load + amplitude * (morning_factor + evening_factor)


class LinearGrowthConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern that grows linearly over time.
    Models scenarios where energy demand steadily increases during the day.
    """
    def calculate(self, time: float) -> float:
        base_load = self.config.get('base_load', defs.DEFAULT_CONS_BASE_LOAD)
        growth_factor = self.config.get('growth_factor', defs.DEFAULT_CONS_GROWTH_FACTOR)
        return base_load + base_load * time * growth_factor


class GMMConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern modeled as a sum of two Gaussian peaks.
    Each peak has a configurable center (peak time), width, and height,
    allowing for flexible modeling of complex daily consumption profiles.
    """
    def calculate(self, time: float) -> float:
        peak_consumption1 = self.config.get('peak_consumption1', defs.DEFAULT_CONS_PEAK_CONSUMPTION1)
        peak_time1 = self.config.get('peak_time1', defs.DEFAULT_CONS_PEAK_TIME1)
        width1 = self.config.get('width1', defs.DEFAULT_CONS_WIDTH1)
        peak_consumption2 = self.config.get('peak_consumption2', defs.DEFAULT_CONS_PEAK_CONSUMPTION2)
        peak_time2 = self.config.get('peak_time2', defs.DEFAULT_CONS_PEAK_TIME2)
        width2 = self.config.get('width2', defs.DEFAULT_CONS_WIDTH2)

        consumption1 = peak_consumption1 * np.exp(-((time - peak_time1) ** 2) / (2 * (width1 ** 2)))
        consumption2 = peak_consumption2 * np.exp(-((time - peak_time2) ** 2) / (2 * (width2 ** 2)))
        return consumption1 + consumption2


class DataDrivenConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern based on external data loaded from a file.
    Interpolates values between data points and applies an optional scaling factor.
    Useful for replicating measured or historical consumption profiles.
    """
    def calculate(self, time: float) -> float:
        data_file = self.config.get('data_file')
        if not data_file:
            raise ValueError("No data file specified for DATA_DRIVEN pattern")
        scale_factor = self.config.get('scale_factor', defs.DEFAULT_CONS_SCALE_FACTOR)
        consumption_data = _load_consumption_data(data_file)
        return _interpolate_consumption(time, consumption_data) * scale_factor

def _load_consumption_data(data_file: str) -> Dict[float, float]:
    """
    Load consumption data from a YAML file.

    Args:
        data_file: Path to the YAML file.

    Returns:
        Dictionary mapping time fractions to consumption values.
    """
    return load_data_from_yaml(data_file, defs.CONSUMPTION_DATA)


def _interpolate_consumption(time_fraction: float, consumption_data: Dict[float, float]) -> float:
    """
    Interpolate the consumption value based on time fraction.

    Args:
        time_fraction: Current time as a fraction of the day (0.0 to 1.0).
        consumption_data: Dictionary mapping time fractions to consumption values.

    Returns:
        Interpolated consumption value.
    """
    return interpolate_value(time_fraction, consumption_data)

# Add a new function to get raw consumption data for a file
def get_raw_consumption_data(data_file: str) -> Dict[float, float]:
    """
    Get raw consumption data from a file without any processing.
    This is useful for visualization purposes.

    Args:
        data_file: Path to the YAML file containing consumption data

    Returns:
        Dictionary mapping time fractions to consumption values
    """
    return _load_consumption_data(data_file)