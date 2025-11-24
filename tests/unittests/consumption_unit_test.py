import unittest
from unittest.mock import MagicMock
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.core.dynamics import EnergyDynamics

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

    def test_get_state(self):
        self.assertEqual(self.consumption_unit.get_state(), 0.0)

    def test_update(self):
        self.consumption_unit.update(time=0.5)
        self.assertEqual(self.consumption_unit.current_consumption, 50.0)
        self.mock_dynamics.get_value.assert_called_with(time=0.5, action=0.0)

    def test_reset(self):
        self.consumption_unit.update(time=0.5)
        self.consumption_unit.reset()
        self.assertEqual(self.consumption_unit.current_consumption, 0.0)


if __name__ == '__main__':
    unittest.main()