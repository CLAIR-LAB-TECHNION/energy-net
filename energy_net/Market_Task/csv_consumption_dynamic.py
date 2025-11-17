from datetime import timedelta

import pandas as pd
from typing import Dict, Any
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.grid_entities.consumption.consumption_dynamics import ConsumptionDynamics


class DTDataConsumptionDynamics(ConsumptionDynamics):
    """
    Consumption pattern based on external data loaded from a CSV file with Datetime and Consumption columns.
    Allows querying consumption at specific date/time values.
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the dynamics with a specific configuration.

        Args:
            params (Dict[str, Any]): Configuration dictionary for the pattern.
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

        print(query_time)
        if query_time not in self.consumption_data.index:
            # Interpolate the missing value
            extended_index = self.consumption_data.index.union([query_time]).sort_values()
            interpolated_data = self.consumption_data.reindex(extended_index).interpolate(method='index')
            return interpolated_data.loc[query_time, 'Consumption']
            # Return the interpolated value
        return self.consumption_data.loc[query_time, 'Consumption']
if __name__ == "__main__":
    # Load the processed data
    data_file = 'processed_30min.csv'
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Ensure the data is not empty
    if df.empty:
        raise ValueError(f"The data file {data_file} is empty or could not be loaded.")

    # Configuration for the ConsumptionUnit
    config = {
        'data_file': data_file,
        'consumption_capacity': 12000.0  # Example capacity in MWh
    }

    # Initialize the dynamics with the data file
    dynamics = DTDataConsumptionDynamics(params={'data_file': config['data_file']})
    # Initialize the ConsumptionUnit with data-driven dynamics
    unit = ConsumptionUnit(dynamics, config)


    # Define the start time for the simulation
    start_time = pd.Timestamp('2025-11-02')

    # Simulate for 10 days (48 intervals per day, 10 days = 480 intervals)
    total_intervals = 48 * 10

    for interval in range(total_intervals):
        # Calculate the time fraction (interval / 48 gives the fraction of the day)
        time_fraction = (interval // 48) + (interval % 48) / 48
        # Update the unit with the time fraction
        unit.update(time=time_fraction)

        # Calculate the current time for display purposes
        current_time = start_time + timedelta(minutes=30 * interval)

        # Print the current state
        print(f"Time: {current_time}, Consumption: {unit.get_state()} MWh")