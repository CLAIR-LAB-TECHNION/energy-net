import unittest

from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.production.production_dynamics import GMMProductionDynamics
from energy_net.grid_entities.consumption.consumption_dynamics import GMMConsumptionDynamics
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.foundation.model import State, Action


class TestPCSUnitSimulation(unittest.TestCase):
    # Do not change this to set_up or another naming convention - it is implementing an abstract
    # method from unittest.TestCase and thus cannot be changed.
    def setUp(self):
        # --- Initialize battery ---
        model_parameters = {
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
            "lifetime_constant": 1e6
        }
        battery_dynamics = DeterministicBattery(model_parameters=model_parameters)
        battery_config = {
            "min": 0.0,
            "max": 1e6,
            "charge_rate_max": 1e5,
            "discharge_rate_max": 1e5,
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
            "init": 5e5
        }
        self.battery = Battery(dynamics=battery_dynamics, config=battery_config)

        # --- Initialize consumption unit ---
        consumption_dynamics_config = {
            "peak_production1": 1200.0,
            "peak_time1": 0.25,
            "width1": 0.05,
            "peak_production2": 1500.0,
            "peak_time2": 0.75,
            "width2": 0.1
        }
        consumption_dynamics = GMMConsumptionDynamics(params=consumption_dynamics_config)
        consumption_unit_config = {"consumption_capacity": 5000.0}
        self.consumption_unit = ConsumptionUnit(
            dynamics=consumption_dynamics,
            config=consumption_unit_config
        )

        # --- Initialize production unit ---
        production_dynamics_config = {
            "peak_production1": 1200.0,
            "peak_time1": 0.25,
            "width1": 0.05,
            "peak_production2": 1500.0,
            "peak_time2": 0.75,
            "width2": 0.1
        }
        production_dynamics = GMMProductionDynamics(params=production_dynamics_config)
        production_unit_config = {"production_capacity": 5000.0}
        self.production_unit = ProductionUnit(
            dynamics=production_dynamics,
            config=production_unit_config
        )

        # --- Create PCSUnit ---
        self.pcs_unit = PCSUnit(
            storage_units=[self.battery],
            production_units=[self.production_unit],
            consumption_units=[self.consumption_unit]
        )

    def test_run_24_hours_without_crash(self):
        """Run the PCSUnit simulation for 24 hours and ensure no exceptions occur."""
        time_step = 1.0 / 24  # 1 hour as fraction of a day

        # Initialize production/consumption at timestep 0 without applying actions
        initial_state = State({"time": 0.0})
        
        # Reset energy_change after initialization since this was just setup
        # Energy change should only track changes from actual timestep actions
        self.pcs_unit._state.set_attribute('energy_change', 0.0)

        for hour in range(24):
            current_time = hour * time_step

            with self.subTest(hour=hour + 1):
                # ---- Build State Object ----
                state = State({
                    "time": current_time
                })

                # ---- ONLY Battery Action ----
                battery_action = Action({
                    "value": (hour+1) * 10  # example charging
                })

                actions = {
                    "Battery_0": battery_action
                }

                # ---- Log the state BEFORE applying action ----
                total_battery = self.pcs_unit.get_total_storage()
                total_production = self.pcs_unit.get_production()
                total_consumption = self.pcs_unit.get_consumption()
                energy_change = self.pcs_unit.get_energy_change()

                print(f"Hour {hour + 1}:") # just so it's 1-24, not 0-23
                print(f"  Total Storage: {total_battery} MWh")
                print(f"  Total Production: {total_production} MW")
                print(f"  Total Consumption: {total_consumption} MW")
                print(f"  Action: {battery_action.get_action('value')} MWh change")
                print(f"  Energy Change (from previous timestep): {energy_change:.3f} MWh")
                print("-" * 40)

                # ---- Minimal Assertions ----
                self.assertTrue(total_battery >= 0)
                self.assertTrue(total_production >= 0)
                self.assertTrue(total_consumption >= 0)
                self.assertIsInstance(energy_change, float)

                # ---- Apply action via update (effects will show in next timestep) ----
                try:
                    self.pcs_unit.update(state=state, actions=actions)
                except Exception as e:
                    self.fail(f"PCS unit simulation crashed at hour {hour + 1}: {e}")


if __name__ == "__main__":
    unittest.main(buffer=False)
