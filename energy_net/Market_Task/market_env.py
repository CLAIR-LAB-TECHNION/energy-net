# Have a simple PCS (only battery) with consumption with non-fixed price, predict how much battery to store).
# Have an ISO run with a simple PCS environment wtih a set policy.
# And then try multi-agent
import pandas as pd
from datetime import timedelta
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.Market_Task.csv_consumption_dynamic import DTDataConsumptionDynamics

def main():
    # Configuration for the ConsumptionUnit
    data_file = 'processed_30min.csv'  # Ensure this file exists in the correct path
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

if __name__ == "__main__":
    main()