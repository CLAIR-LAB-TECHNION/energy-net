import unittest
from unittest.mock import MagicMock
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit

# ------------------------------
# Tests from first file
# ------------------------------
class TestPCSUnitEnergyStorage(unittest.TestCase):
   def setUp(self):
       # Mock Battery
       self.battery = MagicMock(spec=Battery)
       self.battery.get_state.return_value = 50.0
       self.battery.energy_change = 0.0


       # Mock Production Unit
       self.production_unit = MagicMock(spec=ProductionUnit)
       self.production_unit.get_state.return_value = 100.0


       # Mock Consumption Unit
       self.consumption_unit = MagicMock(spec=ConsumptionUnit)
       self.consumption_unit.get_state.return_value = 80.0


       # Create PCSUnit
       self.pcs_unit = PCSUnit(
           batteries=[self.battery],
           production_units=[self.production_unit],
           consumption_units=[self.consumption_unit]
       )


   def test_battery_energy_storage(self):
       # Simulate multiple cycles
       for cycle in range(5):
           self.pcs_unit.perform_collective_actions(
               time=0.5,
               battery_action=10.0,
               consumption_action=80.0,
               production_action=100.0
           )
           self.battery.update.assert_called_with(time=0.5, action=10.0)


       self.assertEqual(self.battery.get_state.call_count, 5)




# ------------------------------
# Tests from second file
# ------------------------------
class TestPCSUnit(unittest.TestCase):
   def setUp(self):
       # Mock Batteries
       self.battery1 = MagicMock(spec=Battery)
       self.battery2 = MagicMock(spec=Battery)
       self.battery1.get_state.return_value = 50.0
       self.battery2.get_state.return_value = 30.0
       self.battery1.energy_change = 5.0
       self.battery2.energy_change = -3.0


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
           batteries=[self.battery1, self.battery2],
           production_units=[self.production_unit1, self.production_unit2],
           consumption_units=[self.consumption_unit1, self.consumption_unit2],
       )


   def test_get_total_battery_capacity(self):
       total_capacity = self.pcs_unit.get_total_battery_capacity()
       self.assertEqual(total_capacity, 80.0)


   def test_get_production(self):
       total_production = self.pcs_unit.get_production()
       self.assertEqual(total_production, 250.0)


   def test_get_consumption(self):
       total_consumption = self.pcs_unit.get_consumption()
       self.assertEqual(total_consumption, 200.0)


   def test_get_energy_change(self):
       total_energy_change = self.pcs_unit.get_energy_change()
       self.assertEqual(total_energy_change, 2.0)


   def test_perform_collective_actions(self):
       self.pcs_unit.perform_collective_actions(
           time=0.5,
           battery_action=10.0,
           consumption_action=50.0,
           production_action=100.0
       )
       self.battery1.update.assert_called_with(time=0.5, action=10.0)
       self.battery2.update.assert_called_with(time=0.5, action=10.0)
       self.production_unit1.update.assert_called_with(time=0.5, action=100.0)
       self.production_unit2.update.assert_called_with(time=0.5, action=100.0)
       self.consumption_unit1.update.assert_called_with(time=0.5, action=50.0)
       self.consumption_unit2.update.assert_called_with(time=0.5, action=50.0)


   def test_reset(self):
       self.pcs_unit.reset(initial_battery_level=40.0)
       self.battery1.reset.assert_called_with(40.0)
       self.battery2.reset.assert_called_with(40.0)
       self.production_unit1.reset.assert_called()
       self.production_unit2.reset.assert_called()
       self.consumption_unit1.reset.assert_called()
       self.consumption_unit2.reset.assert_called()


if __name__ == '__main__':
    unittest.main()
