import unittest
from unittest.mock import MagicMock
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.dynamics import EnergyDynamics

class TestConsumptionUnit(unittest.TestCase):
    def setUp(self):
        # Set up a mock for EnergyDynamics
        self.mock_dynamics = MagicMock(spec=EnergyDynamics)
        self.mock_dynamics.get_value.return_value = 50.0  # Mocked consumption value

        # Configuration for the ConsumptionUnit
        self.config = {
            'consumption_capacity': 100.0  # Maximum consumption capacity
        }

        # Initialize the ConsumptionUnit with mocked dynamics and configuration
        self.consumption_unit = ConsumptionUnit(dynamics=self.mock_dynamics, config=self.config)

    def test_initialization(self):
        # Test if the ConsumptionUnit initializes with correct values
        self.assertEqual(self.consumption_unit.consumption_capacity, 100.0)  # Check capacity
        self.assertEqual(self.consumption_unit.current_consumption, 0.0)  # Initial consumption should be 0

    def test_get_state(self):
        # Test if the initial state of the ConsumptionUnit is 0
        self.assertEqual(self.consumption_unit.get_state(), 0.0)

    def test_update(self):
        # Test the update method to ensure it updates the current consumption correctly
        self.consumption_unit.update(time=0.5)  # Update the unit at time 0.5
        self.assertEqual(self.consumption_unit.current_consumption, 50.0)  # Check updated consumption
        self.mock_dynamics.get_value.assert_called_with(time=0.5, action=0.0)  # Verify dynamics interaction

    def test_reset(self):
        # Test the reset method to ensure it resets the current consumption to 0
        self.consumption_unit.update(time=0.5)  # Update the unit first
        self.consumption_unit.reset()  # Reset the unit
        self.assertEqual(self.consumption_unit.current_consumption, 0.0)  # Check if reset worked


if __name__ == '__main__':
    unittest.main()