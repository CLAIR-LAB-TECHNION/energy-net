import unittest
from unittest.mock import Mock
from unittest.mock import MagicMock

from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery
from energy_net.foundation.model import State, Action
from energy_net.common.utils import setup_logger
import os


class TestDeterministicBattery(unittest.TestCase):
    def setUp(self):
        # ---------------------------
        # Mock Dynamics for Battery
        # ---------------------------
        self.mock_dynamics = MagicMock()
        # Simulate get_value returning current_energy + 5 for testing
        self.mock_dynamics.get_value.side_effect = lambda **kwargs: kwargs['current_energy'] + 5

        # ---------------------------
        # Configuration for Battery
        # ---------------------------
        self.battery_config = {
            'min': 0,
            'max': 100,
            'charge_rate_max': 10,
            'discharge_rate_max': 10,
            'charge_efficiency': 0.9,
            'discharge_efficiency': 0.8,
            'init': 50
        }

        # Ensure log directory exists
        log_dir = 'tests/logs'
        os.makedirs(log_dir, exist_ok=True)

        # ---------------------------
        # Battery instance
        # ---------------------------
        self.battery = Battery(
            dynamics=self.mock_dynamics,
            config=self.battery_config,
            log_file=os.path.join(log_dir, 'battery_test.log')  # real log file
        )

        # ---------------------------
        # DeterministicBattery instance
        # ---------------------------
        self.dynamics = DeterministicBattery({
            'charge_efficiency': 0.9,
            'discharge_efficiency': 0.8,
            'lifetime_constant': 100
        })

    def test_get_value_charge(self):
        energy = self.dynamics.get_value(
            time=0.1,
            action=10,
            current_energy=50,
            min_energy=0,
            max_energy=100,
            charge_rate_max=15,
            discharge_rate_max=20
        )
        expected = 50 + 10 * self.dynamics.charge_efficiency
        self.assertAlmostEqual(energy, expected)

    def test_get_value_discharge(self):
        energy = self.dynamics.get_value(
            time=0.1,
            action=-10,
            current_energy=50,
            min_energy=0,
            max_energy=100,
            charge_rate_max=15,
            discharge_rate_max=20
        )
        expected = 50 - 10 * self.dynamics.discharge_efficiency
        self.assertAlmostEqual(energy, expected)

    def test_get_value_no_action(self):
        energy = self.dynamics.get_value(
            time=0.1,
            action=0,
            current_energy=50,
            min_energy=0,
            max_energy=100,
            charge_rate_max=15,
            discharge_rate_max=20
        )
        self.assertEqual(energy, 50)

    def test_exp_mult_clamping(self):
        # Exponent > 100 should clamp
        val = self.dynamics.exp_mult(10, 1, 2000)
        expected = 10 * 3.720075976020836e-44  # updated expected value
        self.assertAlmostEqual(val, expected, delta=1e-44)

    def test_invalid_lifetime_constant(self):
        with self.assertRaises(ValueError):
            self.dynamics.exp_mult(10, 0, 1)


class TestBattery(unittest.TestCase):
    def setUp(self):
        # Mock Dynamics
        self.mock_dynamics = Mock()
        self.mock_dynamics.get_value.side_effect = lambda **kwargs: kwargs['current_energy'] + 5

        # Configuration
        self.config = {
            'min': 0,
            'max': 100,
            'charge_rate_max': 10,
            'discharge_rate_max': 10,
            'charge_efficiency': 0.9,
            'discharge_efficiency': 0.8,
            'init': 50
        }

        # Mock logger to prevent actual logging
        self.battery = Battery(dynamics=self.mock_dynamics, config=self.config, log_file=None)
        self.battery.logger = Mock()

    def test_initial_state(self):
        self.assertEqual(self.battery.energy_level, 50)

    def test_perform_action_updates_energy(self):
        self.battery.current_time = 0.1
        action = Action({'value': 10})
        self.battery.perform_action(action)
        self.assertEqual(self.battery.energy_level, 55)
        self.assertEqual(self.battery.energy_change, 5)

    def test_get_state(self):
        state_value = self.battery.get_state()
        self.assertIsInstance(state_value, (int, float))  # allow int or float
        self.assertEqual(state_value, 50)

    def test_update_with_action(self):
        state = State({'time': 0.2})
        action = Action({'value': 5})
        self.battery.update(state, action)
        self.assertEqual(self.battery.energy_level, 55)

    def test_update_without_action(self):
        state = State({'time': 0.2})
        # No action means no energy change
        self.battery.update(state)
        self.assertEqual(self.battery.energy_level, 50)

    def test_reset_default(self):
        self.battery.energy_level = 10
        self.battery.reset()
        self.assertEqual(self.battery.energy_level, 50)

    def test_reset_custom(self):
        self.battery.reset(initial_level=20)
        self.assertEqual(self.battery.energy_level, 20)


if __name__ == "__main__":
    unittest.main()