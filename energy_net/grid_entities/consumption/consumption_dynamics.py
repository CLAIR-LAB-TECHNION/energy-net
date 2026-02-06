import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
from energy_net.common import defs
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import load_data_from_yaml, interpolate_value
import pandas as pd
from datetime import datetime, timedelta
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

class ModelDrivenConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern driven by an external predictive model (e.g., gradient boosting).
    This dynamics class accepts a trained predictor model and queries it in real-time
    to generate consumption values based on predicted demand patterns.
    
    The model should have a predict(date_str, time_str) method that returns consumption values.
    Examples include the EnergyPredictor class from consumption_prediction module.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the dynamics with a trained predictive model.
        
        Args:
            params (Dict[str, Any]): Configuration dictionary containing:
                - model: Trained predictor model with predict(date_str, time_str) method (required)
                - reference_date: Starting date for the simulation in 'YYYY-MM-DD' format (required)
                - datetime_format: Format string for parsing reference_date (optional, default '%Y-%m-%d')
        
        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        super().__init__(params)
        
        # Validate required parameters
        if 'model' not in self.params:
            raise ValueError("Missing 'model' parameter in ModelDrivenConsumptionDynamics")
        if 'reference_date' not in self.params:
            raise ValueError("Missing 'reference_date' parameter in ModelDrivenConsumptionDynamics")
        
        self.model = self.params['model']
        
        # Validate model has predict method
        if not hasattr(self.model, 'predict') or not callable(getattr(self.model, 'predict')):
            raise ValueError("Model must have a callable 'predict' method")
        
        # Parse reference date
        datetime_format = self.params.get('datetime_format', '%Y-%m-%d')
        try:
            self.reference_date = datetime.strptime(self.params['reference_date'], datetime_format)
        except Exception as e:
            raise ValueError(f"Failed to parse reference_date '{self.params['reference_date']}' with format '{datetime_format}': {e}")
    
    def _convert_time_to_datetime(self, time: float) -> tuple:
        """
        Converts simulation time (in days as float) to actual date and time strings.
        
        Args:
            time (float): Simulation time where integer part is day number and 
                         fractional part is time within the day (0.0-1.0)
        
        Returns:
            tuple: (date_str, time_str) formatted for model prediction
                  e.g., ('2025-01-05', '14:30')
        """
        # Extract day offset and time within day
        day_offset = int(time)
        time_within_day = time - day_offset
        
        # Convert time_within_day to hours and minutes with rounding
        total_minutes = round(time_within_day * 24 * 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        # Calculate actual datetime
        actual_datetime = self.reference_date + timedelta(days=day_offset, hours=hours, minutes=minutes)
        
        # Format for model prediction
        date_str = actual_datetime.strftime('%Y-%m-%d')
        time_str = actual_datetime.strftime('%H:%M')
        
        return date_str, time_str
    
    def get_value(self, **kwargs) -> float:
        """
        Retrieves the consumption value by querying the predictive model.
        
        Args:
            **kwargs:
                - time (float): Current simulation time as days (e.g., 0.0 = day 0 midnight,
                               1.5 = day 1 noon, 5.25 = day 5 at 6:00 AM)
        
        Returns:
            float: The predicted consumption value from the model.
        """
        time = kwargs.get('time', 0.0)
        
        # Convert time to date and time strings
        date_str, time_str = self._convert_time_to_datetime(time)
        
        # Query the model
        try:
            consumption = self.model.predict(date_str, time_str)
            return float(consumption)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed for {date_str} {time_str}: {e}")

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
