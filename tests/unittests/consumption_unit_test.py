import unittest
from unittest.mock import MagicMock
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.foundation.model import State, Action


class TestConsumptionUnit(unittest.TestCase):
    def setUp(self):
        # Mock EnergyDynamics
        self.mock_dynamics = MagicMock(spec=EnergyDynamics)
        self.mock_dynamics.get_value.return_value = 50.0  # Mocked consumption value

        # Configuration for ConsumptionUnit
        self.config = {
            'consumption_capacity': 100.0
        }

        # Initialize ConsumptionUnit
        self.consumption_unit = ConsumptionUnit(dynamics=self.mock_dynamics, config=self.config)

    def test_initialization(self):
        self.assertEqual(self.consumption_unit.consumption_capacity, 100.0)
        self.assertEqual(self.consumption_unit.current_consumption, 0.0)
        # Check internal state was initialized
        self.assertIsNotNone(self.consumption_unit._state)
        self.assertEqual(self.consumption_unit._state.get_attribute('consumption'), 0.0)
        self.assertEqual(self.consumption_unit._state.get_attribute('time'), 0.0)

    def test_get_state_legacy(self):
        # get_state() returns float for backward compatibility
        self.assertEqual(self.consumption_unit.get_state(), 0.0)
        self.assertIsInstance(self.consumption_unit.get_state(), float)

    def test_update_legacy_interface(self):
        # Legacy: passing float as state (interpreted as time)
        self.consumption_unit.update(0.5, action=0.0)
        self.assertEqual(self.consumption_unit.current_consumption, 50.0)
        self.mock_dynamics.get_value.assert_called_with(time=0.5, action=0.0)

    def test_update_new_interface_with_state(self):
        # New: passing State object
        state = State({'time': 0.5})
        action = Action({'value': 0.0})
        self.consumption_unit.update(state, action)
        self.assertEqual(self.consumption_unit.current_consumption, 50.0)
        self.mock_dynamics.get_value.assert_called_with(time=0.5, action=0.0)

    def test_update_new_interface_with_float_action(self):
        # Mixed: State object with float action
        state = State({'time': 0.5})
        self.consumption_unit.update(state, 0.0)
        self.assertEqual(self.consumption_unit.current_consumption, 50.0)

    def test_update_updates_internal_state(self):
        # Verify internal state is updated
        state = State({'time': 0.5})
        self.consumption_unit.update(state)
        internal_state = self.consumption_unit._state
        self.assertEqual(internal_state.get_attribute('time'), 0.5)
        self.assertEqual(internal_state.get_attribute('consumption'), 50.0)

    def test_perform_action_does_nothing_for_consumption(self):
        # ConsumptionUnit's perform_action is a no-op (consumption is autonomous)
        # Just verify it can be called without errors
        action = Action({'value': 5.0})
        self.consumption_unit.perform_action(action)

        # Verify it doesn't change consumption
        self.assertEqual(self.consumption_unit.current_consumption, 0.0)

    def test_perform_action_legacy_does_nothing(self):
        # Test perform_action with float (also a no-op)
        self.consumption_unit.perform_action(5.0)

        # Verify it doesn't change consumption
        self.assertEqual(self.consumption_unit.current_consumption, 0.0)

    def test_reset(self):
        # Update to change state
        self.consumption_unit.update(0.5)
        self.assertEqual(self.consumption_unit.current_consumption, 50.0)

        # Reset
        self.consumption_unit.reset()

        # Verify reset to initial values
        self.assertEqual(self.consumption_unit.current_consumption, 0.0)
        self.assertEqual(self.consumption_unit._state.get_attribute('consumption'), 0.0)
        self.assertEqual(self.consumption_unit._state.get_attribute('time'), 0.0)

    def test_state_without_time_attribute(self):
        # Test handling of State without 'time' attribute
        state = State({'other_attribute': 123})
        self.consumption_unit.update(state)
        # Should default to 0.0 and log warning
        self.mock_dynamics.get_value.assert_called_with(time=0.0, action=0.0)

    def test_multiple_updates_track_state(self):
        # Test that multiple updates correctly track state
        state1 = State({'time': 0.3})
        self.mock_dynamics.get_value.return_value = 30.0
        self.consumption_unit.update(state1)

        self.assertEqual(self.consumption_unit.current_consumption, 30.0)
        self.assertEqual(self.consumption_unit._state.get_attribute('consumption'), 30.0)
        self.assertEqual(self.consumption_unit._state.get_attribute('time'), 0.3)

        state2 = State({'time': 0.7})
        self.mock_dynamics.get_value.return_value = 70.0
        self.consumption_unit.update(state2)

        self.assertEqual(self.consumption_unit.current_consumption, 70.0)
        self.assertEqual(self.consumption_unit._state.get_attribute('consumption'), 70.0)
        self.assertEqual(self.consumption_unit._state.get_attribute('time'), 0.7)


if __name__ == '__main__':
    unittest.main()