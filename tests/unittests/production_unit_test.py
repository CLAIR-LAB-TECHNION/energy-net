# test_production_unit.py

import unittest
from unittest.mock import MagicMock
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.dynamics import EnergyDynamics


class TestProductionUnit(unittest.TestCase):
    def setUp(self):
        # Mock the EnergyDynamics class
        self.mock_dynamics = MagicMock(spec=EnergyDynamics)
        self.mock_dynamics.get_value.return_value = 50.0

        # Configuration for the ProductionUnit
        self.config = {
            'production_capacity': 100.0
        }

        # Create an instance of ProductionUnit
        self.production_unit = ProductionUnit(dynamics=self.mock_dynamics, config=self.config)

    def test_initialization(self):
        # Test if the ProductionUnit is initialized correctly
        self.assertEqual(self.production_unit.production_capacity, 100.0)
        self.assertEqual(self.production_unit.current_production, 0.0)
        self.assertEqual(self.production_unit.initial_production, 0.0)

    def test_get_state(self):
        # Test the get_state method
        self.production_unit.current_production = 75.0
        self.assertEqual(self.production_unit.get_state(), 75.0)

    def test_update(self):
        # Test the update method
        self.production_unit.update(time=0.5)
        self.mock_dynamics.get_value.assert_called_once_with(time=0.5, action=0.0)
        self.assertEqual(self.production_unit.current_production, 50.0)

    def test_reset(self):
        # Test the reset method
        self.production_unit.current_production = 80.0
        self.production_unit.reset()
        self.assertEqual(self.production_unit.current_production, 0.0)

    def test_perform_action(self):
        # Test the perform_action method (no effect expected)
        self.production_unit.perform_action(action=10.0)
        self.assertEqual(self.production_unit.current_production, 0.0)


if __name__ == '__main__':
    unittest.main()