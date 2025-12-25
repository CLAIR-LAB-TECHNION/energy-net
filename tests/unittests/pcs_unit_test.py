import unittest
from unittest.mock import MagicMock, PropertyMock
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.foundation.model import State, Action


# ------------------------------
# Tests from first file
# ------------------------------
class TestPCSUnitEnergyStorage(unittest.TestCase):
    # Do not change this name: unittest expects setUp
    def setUp(self):
        # Mock Battery
        self.battery = MagicMock(spec=Battery)
        self.battery.get_state.return_value = 50.0
        # If energy_change is a property on the real Battery, mock it properly:
        type(self.battery).energy_change = PropertyMock(return_value=0.0)

        # Add the new capacity methods
        self.battery.get_available_charge_capacity.return_value = 50.0  # Can charge up to 50 MWh
        self.battery.get_available_discharge_capacity.return_value = 30.0  # Can discharge up to 30 MWh

        # Mock Production Unit
        self.production_unit = MagicMock(spec=ProductionUnit)
        self.production_unit.get_state.return_value = 100.0

        # Mock Consumption Unit
        self.consumption_unit = MagicMock(spec=ConsumptionUnit)
        self.consumption_unit.get_state.return_value = 80.0

        # Create PCSUnit
        self.pcs_unit = PCSUnit(
            storage_units=[self.battery],
            production_units=[self.production_unit],
            consumption_units=[self.consumption_unit]
        )

    def test_battery_energy_storage(self):
        # New-style State object
        state = State({'time': 0.5})

        # Wrap action values in Action objects (per-entity)
        actions = {
            "Battery_0": Action({"energy_delta": 0.5}),
            "ConsumptionUnit_0": Action({"consumption_setpoint": 80.0}),
            "ProductionUnit_0": Action({"production_setpoint": 100.0})
        }

        for cycle in range(5):
            self.pcs_unit.update(state=state, actions=actions)
            self.pcs_unit.get_total_storage()

        # Expect get_state to be called once per cycle when get_total_storage runs
        self.assertEqual(self.battery.get_state.call_count, 5)

    def test_perform_actions(self):
        state = State({'time': 0.5})
        actions = {
            "Battery_0": Action({"energy_delta": 10.0}),
            "ConsumptionUnit_0": Action({"consumption_setpoint": 80.0}),
            "ProductionUnit_0": Action({"production_setpoint": 100.0})
        }
        self.pcs_unit.update(state=state, actions=actions)
        # Print mock calls for debugging if needed
        print(self.consumption_unit.mock_calls)


# ------------------------------
# Tests from second file
# ------------------------------
class TestPCSUnit(unittest.TestCase):
    # Do not change this name: unittest expects setUp
    def setUp(self):
        # Mock Batteries
        self.battery1 = MagicMock(spec=Battery)
        self.battery2 = MagicMock(spec=Battery)

        self.battery1.get_state.return_value = 50.0
        self.battery2.get_state.return_value = 30.0

        # Directly assign numeric energy_change
        self.battery1.energy_change = 5.0
        self.battery2.energy_change = -3.0

        # Add the new capacity methods for both batteries
        self.battery1.get_available_charge_capacity.return_value = 40.0  # Can charge up to 40 MWh
        self.battery1.get_available_discharge_capacity.return_value = 35.0  # Can discharge up to 35 MWh
        self.battery2.get_available_charge_capacity.return_value = 60.0  # Can charge up to 60 MWh
        self.battery2.get_available_discharge_capacity.return_value = 25.0  # Can discharge up to 25 MWh

        # Mock Production Units
        self.production_unit1 = MagicMock(spec=ProductionUnit)
        self.production_unit2 = MagicMock(spec=ProductionUnit)
        self.production_unit1.get_state.return_value = 100.0
        self.production_unit2.get_state.return_value = 150.0

        # Mock Consumption Units
        self.consumption_unit1 = MagicMock(spec=ConsumptionUnit)
        self.consumption_unit2 = MagicMock(spec=ConsumptionUnit)
        self.consumption_unit1.get_state.return_value = 80.0
        self.consumption_unit2.get_state.return_value = 120.0

        # Create PCSUnit
        self.pcs_unit = PCSUnit(
            storage_units=[self.battery1, self.battery2],
            production_units=[self.production_unit1, self.production_unit2],
            consumption_units=[self.consumption_unit1, self.consumption_unit2],
        )

    def test_get_total_storage(self):
        total_capacity = self.pcs_unit.get_total_storage()
        self.assertEqual(total_capacity, 80.0)

    def test_get_production(self):
        total_production = self.pcs_unit.get_production()
        self.assertEqual(total_production, 250.0)

    def test_get_consumption(self):
        total_consumption = self.pcs_unit.get_consumption()
        self.assertEqual(total_consumption, 200.0)

    def test_perform_collective_actions(self):
        state = State({'time': 0.5})
        actions = {
            "Battery_0": Action({"energy_delta": 10.0}),
            "Battery_1": Action({"energy_delta": 10.0}),
            "ConsumptionUnit_0": Action({"consumption_setpoint": 80.0}),
            "ConsumptionUnit_1": Action({"consumption_setpoint": 80.0}),
            "ProductionUnit_0": Action({"production_setpoint": 100.0}),
            "ProductionUnit_1": Action({"production_setpoint": 100.0})
        }
        self.pcs_unit.update(state=state, actions=actions)

    def test_reset(self):
        self.pcs_unit.reset(initial_storage_unit_level=40.0)
        self.battery1.reset.assert_called_with(40.0)
        self.battery2.reset.assert_called_with(40.0)
        self.production_unit1.reset.assert_called()
        self.production_unit2.reset.assert_called()
        self.consumption_unit1.reset.assert_called()
        self.consumption_unit2.reset.assert_called()


if __name__ == '__main__':
    unittest.main(buffer=False)