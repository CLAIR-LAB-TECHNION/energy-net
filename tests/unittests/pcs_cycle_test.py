import unittest
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.storage.battery_dynamics import Deterministicbattery
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.production.production_dynamics import GMMProductionDynamics
from energy_net.grid_entities.consumption.consumption_dynamics import GMMConsumptionDynamics
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit


class TestPCSUnitSimulation(unittest.TestCase):
    def setUp(self):
        # --- Initialize battery ---
        model_parameters = {
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
            "lifetime_constant": 1e6
        }
        battery_dynamics = Deterministicbattery(model_parameters=model_parameters)
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
            "peak_consumption1": 1000.0,
            "peak_time1": 0.3,
            "width1": 0.05,
            "peak_consumption2": 1500.0,
            "peak_time2": 0.8,
            "width2": 0.1
        }
        consumption_dynamics = GMMConsumptionDynamics(params=consumption_dynamics_config)
        consumption_unit_config = {"consumption_capacity": 5000.0}
        self.consumption_unit = ConsumptionUnit(dynamics=consumption_dynamics, config=consumption_unit_config)

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
        self.production_unit = ProductionUnit(dynamics=production_dynamics, config=production_unit_config)

        # --- Create PCSUnit ---
        self.pcs_unit = PCSUnit(
            batteries=[self.battery],
            production_units=[self.production_unit],
            consumption_units=[self.consumption_unit]
        )

    def test_run_24_hours_without_crash(self):
        """Run the PCSUnit simulation for 24 hours and ensure no exceptions occur."""
        time_step = 1.0 / 24  # 1 hour as fraction of a day
        for hour in range(24):
            with self.subTest(hour=hour + 1):
                current_time = hour * time_step

                battery_action = 20.0  # Example charging action
                consumption_action = self.consumption_unit.get_state()
                production_action = self.production_unit.get_state()

                try:
                    self.pcs_unit.perform_collective_actions(
                        time=current_time,
                        battery_action=battery_action,
                        consumption_action=consumption_action,
                        production_action=production_action
                    )
                except Exception as e:
                    self.fail(f"PCS unit simulation crashed at hour {hour + 1}: {e}")

                # Log the state for inspection
                total_battery = self.pcs_unit.get_total_battery_capacity()
                total_production = self.pcs_unit.get_production()
                total_consumption = self.pcs_unit.get_consumption()
                energy_change = self.pcs_unit.get_energy_change()

                # Print/log output per hour
                print(f"Hour {hour + 1}:")
                print(f"  Battery Capacity: {total_battery} MWh")
                print(f"  Total Production: {total_production} MW")
                print(f"  Total Consumption: {total_consumption} MW")
                print(f"  Energy Change: {energy_change} MWh")
                print("-" * 40)

                # Minimal assertions to satisfy unittest
                self.assertTrue(total_battery >= 0)
                self.assertTrue(total_production >= 0)
                self.assertTrue(total_consumption >= 0)
                self.assertIsInstance(energy_change, float)


if __name__ == "__main__":
    unittest.main()
