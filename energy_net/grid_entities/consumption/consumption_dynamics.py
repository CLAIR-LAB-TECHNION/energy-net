import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from energy_net.common import defs
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import load_data_from_yaml, interpolate_value
import pandas as pd
class ConsumptionDynamics(EnergyDynamics):
    """
    Unified class for defining and handling energy consumption patterns and dynamics.
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
        Retrieves the current consumption value.

        Args:
            **kwargs:
                - time (float): Current time as a fraction of the day (0.0 to 1.0).

        Returns:
            float: The calculated consumption value.
        """
        pass



class ConstantConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern with a constant value throughout the day.
    Useful for scenarios where energy usage does not fluctuate with time.
    """

    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)
        return self.params.get('base_load', defs.DEFAULT_CONS_BASE_LOAD)

class DoublePeakConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern with two distinct peaks during the day:
    one in the morning and one in the evening. The sharpness of peaks
    and their amplitude can be adjusted.
    """
    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)
        base_load = self.params.get('base_load', defs.DEFAULT_CONS_BASE_LOAD)
        amplitude = self.params.get('amplitude', defs.DEFAULT_CONS_AMPLITUDE)
        morning_peak = self.params.get('morning_peak', defs.DEFAULT_CONS_MORNING_PEAK)
        evening_peak = self.params.get('evening_peak', defs.DEFAULT_CONS_EVENING_PEAK)
        peak_sharpness = self.params.get('peak_sharpness', defs.DEFAULT_CONS_PEAK_SHARPNESS)

        morning_factor = np.exp(-peak_sharpness * ((time - morning_peak) ** 2))
        evening_factor = np.exp(-peak_sharpness * ((time - evening_peak) ** 2))
        return base_load + amplitude * (morning_factor + evening_factor)

class LinearGrowthConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern that grows linearly over time.
    Models scenarios where energy demand steadily increases during the day.
    """
    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)
        base_load = self.params.get('base_load', defs.DEFAULT_CONS_BASE_LOAD)
        growth_factor = self.params.get('growth_factor', defs.DEFAULT_CONS_GROWTH_FACTOR)
        return base_load + base_load * time * growth_factor


class GMMConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern modeled as a sum of two Gaussian peaks.
    Each peak has a configurable center (peak time), width, and height,
    allowing for flexible modeling of complex daily consumption profiles.
    """
    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)

        peak_consumption1 = self.params.get('peak_consumption1', defs.DEFAULT_CONS_PEAK1)
        peak_time1 = self.params.get('peak_time1', defs.DEFAULT_CONS_PEAK_TIME1)
        width1 = self.params.get('width1', defs.DEFAULT_CONS_WIDTH1)
        peak_consumption2 = self.params.get('peak_consumption2', defs.DEFAULT_CONS_PEAK2)
        peak_time2 = self.params.get('peak_time2', defs.DEFAULT_CONS_PEAK_TIME2)
        width2 = self.params.get('width2', defs.DEFAULT_CONS_WIDTH2)

        consumption1 = peak_consumption1 * np.exp(-((time - peak_time1) ** 2) / (2 * (width1 ** 2)))
        consumption2 = peak_consumption2 * np.exp(-((time - peak_time2) ** 2) / (2 * (width2 ** 2)))
        return consumption1 + consumption2


class YAML_DataConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern based on external data loaded from a file.
    Interpolates values between data points and applies an optional scaling factor.
    Useful for replicating measured or historical consumption profiles.
    """
    def get_value(self, **kwargs) -> float:
        time = kwargs.get('time', 0.0)

        data_file = self.params.get('data_file')
        if not data_file:
            raise ValueError("No data file specified for DATA_DRIVEN pattern")
        scale_factor = self.params.get('scale_factor', defs.DEFAULT_CONS_SCALE_FACTOR)
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

class CSV_DataConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern based on external data loaded from a CSV file with Datetime and Consumption columns.
    Allows querying consumption at specific date/time values.
    """
    def __init__(self, params: Dict[str, str]):
        """
        Initializes the dynamics with a specific configuration.

        Args:
            params (Dict[str, EnergyDynamics]): Configuration dictionary for the pattern.
        """
        super().__init__(params)
        self.data_file = self.params.get('data_file')
        if not self.data_file:
            raise ValueError("No data file specified for DateTimeDrivenConsumptionDynamics")
        self.consumption_data = self._load_consumption_data(self.data_file)
        self.total_days = int(self.consumption_data.index[-1])+1



    def _load_consumption_data(self, data_file: str) -> pd.DataFrame:
        """
        Load consumption data from a CSV file.

        Args:
            data_file: Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing Datetime, Consumption, and TimeFraction columns.
        """
        try:
            df = pd.read_csv(data_file, parse_dates=['Datetime'])
            if 'Datetime' not in df.columns or 'Consumption' not in df.columns:
                raise ValueError("CSV file must contain 'Datetime' and 'Consumption' columns")
            df.set_index('Datetime', inplace=True)

            # Calculate the day number and time fraction
            df['DayNumber'] = (df.index - df.index[0]).days
            df['TimeFraction'] = df.index.hour / 24 + df.index.minute / 1440
            df['DayTimeFraction'] = df['DayNumber'] + df['TimeFraction']
            df.set_index('DayTimeFraction', inplace=True)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load consumption data: {e}")

    def get_value(self, **kwargs) -> float:
        """
        Retrieves the consumption value for a specific date/time.

        Args:
            **kwargs:
                - datetime (datetime): The specific date/time to query.

        Returns:
            float: The consumption value at the specified date/time.
        """
        query_time = kwargs.get('time')

        # Ensure the query time wraps around within the total days to handle dates that go past the data range
        query_time = query_time%self.total_days

        if query_time not in self.consumption_data.index:
            # Interpolate the missing value
            extended_index = self.consumption_data.index.union([query_time]).sort_values()
            interpolated_data = self.consumption_data.reindex(extended_index).interpolate(method='index')
            return interpolated_data.loc[query_time, 'Consumption']
            # Return the interpolated value
        return self.consumption_data.loc[query_time, 'Consumption']

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