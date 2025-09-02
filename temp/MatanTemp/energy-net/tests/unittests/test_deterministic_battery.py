# tests/test_deterministic_battery.py

import unittest
from energy_net.dynamics.storage_dynamics.battery_dynamics_det import BatteryDynamicsDet
from typing import Dict, Any


class TestDeterministicBattery(unittest.TestCase):
    def setUp(self):
        # Define valid model parameters
        self.valid_model_params: Dict[str, Any] = {
            'charge_efficiency': 0.4,
            'discharge_efficiency': 0.6,
            'lifetime_constant': 100.0
        }
        self.battery_dynamics = BatteryDynamicsDet(model_parameters=self.valid_model_params)

    def test_initialization_with_valid_parameters(self):
        # Initialization should succeed with valid parameters
        self.assertIsInstance(self.battery_dynamics, BatteryDynamicsDet)
        self.assertEqual(self.battery_dynamics.charge_efficiency, 0.4)
        self.assertEqual(self.battery_dynamics.discharge_efficiency, 0.6)
        self.assertEqual(self.battery_dynamics.lifetime_constant, 100.0)

    def test_initialization_missing_parameters(self):
        # Initialization should fail if parameters are missing
        incomplete_params = {
            'charge_efficiency': 0.4,
            'discharge_efficiency': 0.6
            # 'lifetime_constant' is missing
        }
        with self.assertRaises(AssertionError):
            BatteryDynamicsDet(model_parameters=incomplete_params)

    def test_initialization_invalid_efficiency_values(self):
        # Initialization should fail if efficiencies are out of bounds
        invalid_params_low = {
            'charge_efficiency': 0.0,  # Invalid
            'discharge_efficiency': 0.6,
            'lifetime_constant': 100.0
        }
        with self.assertRaises(AssertionError):
            BatteryDynamicsDet(model_parameters=invalid_params_low)

        invalid_params_high = {
            'charge_efficiency': 1.1,  # Invalid
            'discharge_efficiency': 0.6,
            'lifetime_constant': 100.0
        }
        with self.assertRaises(AssertionError):
            BatteryDynamicsDet(model_parameters=invalid_params_high)

    def test_initialization_invalid_lifetime_constant(self):
        # Initialization should fail if lifetime_constant is non-positive
        invalid_params = {
            'charge_efficiency': 0.4,
            'discharge_efficiency': 0.6,
            'lifetime_constant': 0.0  # Invalid
        }
        with self.assertRaises(AssertionError):
            BatteryDynamicsDet(model_parameters=invalid_params)

    def test_get_value_charging_within_limits(self):
        # Test charging within max limits
        kwargs = {
            'time': 0.5,
            'action': 2.0,  # MW
            'current_energy': 30.0,
            'min_energy': 0.0,
            'max_energy': 60.0,
            'charge_rate_max': 4.0,
            'discharge_rate_max': 2.0
        }
        updated_energy = self.battery_dynamics.get_value(**kwargs)
        expected_energy = 30.0 + 2.0 * 0.4  # 30 + 0.8 = 30.8
        self.assertAlmostEqual(updated_energy, expected_energy)

    def test_get_value_discharging_within_limits(self):
        # Test discharging within min limits
        kwargs = {
            'time': 0.5,
            'action': -1.5,  # MW
            'current_energy': 30.0,
            'min_energy': 0.0,
            'max_energy': 60.0,
            'charge_rate_max': 4.0,
            'discharge_rate_max': 2.0
        }
        updated_energy = self.battery_dynamics.get_value(**kwargs)
        expected_energy = 30.0 - 1.5 * 0.6  # 30 - 0.9 = 29.1
        self.assertAlmostEqual(updated_energy, expected_energy)

    def test_get_value_no_action(self):
        # Test no action (action = 0)
        kwargs = {
            'time': 0.5,
            'action': 0.0,  # No action
            'current_energy': 30.0,
            'min_energy': 0.0,
            'max_energy': 60.0,
            'charge_rate_max': 4.0,
            'discharge_rate_max': 2.0
        }
        updated_energy = self.battery_dynamics.get_value(**kwargs)
        expected_energy = 30.0  # No change
        self.assertAlmostEqual(updated_energy, expected_energy)


if __name__ == '__main__':
    unittest.main()
