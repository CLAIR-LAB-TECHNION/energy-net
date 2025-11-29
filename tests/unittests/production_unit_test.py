# test_production_unit.py

import unittest
from unittest.mock import MagicMock
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.foundation.model import State, Action


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
        self.assertIsInstance(self.production_unit.get_state(), float)

    def test_update_legacy_interface(self):
        # Legacy: passing float as state (interpreted as time)
        self.production_unit.update(0.5, action=0.0)
        self.mock_dynamics.get_value.assert_called_once_with(time=0.5, action=0.0)
        self.assertEqual(self.production_unit.current_production, 50.0)

    def test_update_with_state_object(self):
        # New: passing State object
        state = State({'time': 0.5})
        self.production_unit.update(state, action=0.0)
        self.mock_dynamics.get_value.assert_called_with(time=0.5, action=0.0)
        self.assertEqual(self.production_unit.current_production, 50.0)

    def test_update_with_action_object(self):
        # New: passing Action object
        state = State({'time': 0.5})
        action = Action({'value': 0.0})
        self.production_unit.update(state, action)
        self.mock_dynamics.get_value.assert_called_with(time=0.5, action=0.0)
        self.assertEqual(self.production_unit.current_production, 50.0)

    def test_update_without_time_in_state(self):
        # Test handling of State without 'time' attribute
        state = State({'other_attribute': 123})
        self.production_unit.update(state)
        # Should default to 0.0
        self.mock_dynamics.get_value.assert_called_with(time=0.0, action=0.0)

    def test_reset(self):
        # Test the reset method
        self.production_unit.current_production = 80.0
        self.production_unit.reset()
        self.assertEqual(self.production_unit.current_production, 0.0)

    def test_perform_action_is_noop(self):
        # perform_action is pass (no-op) for ProductionUnit
        # Just verify it can be called without errors
        self.production_unit.perform_action(10.0)
        self.assertEqual(self.production_unit.current_production, 0.0)

        action = Action({'value': 10.0})
        self.production_unit.perform_action(action)
        self.assertEqual(self.production_unit.current_production, 0.0)


if __name__ == '__main__':
    unittest.main()