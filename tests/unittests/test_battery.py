import unittest
import math
from unittest.mock import Mock, patch
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.storage.battery_dynamics import BatteryDynamics
from energy_net.model.action import Action, ActionType


# Shared test configurations and utilities
class TestConfig:
    """Shared configuration constants for battery tests."""
    BATTERY_CONFIG = {
        "min": 0.0,
        "max": 100.0,
        "charge_rate_max": 10.0,
        "discharge_rate_max": 15.0,
        "charge_efficiency": 0.9,
        "discharge_efficiency": 0.85,
        "init": 50.0
    }
    
    DYNAMICS_CONFIG = {
        "charge_efficiency": 0.9,
        "discharge_efficiency": 0.85
    }
    
    @staticmethod
    def get_standard_kwargs(action, current_energy=50.0):
        """Get standard kwargs for dynamics calculations."""
        return {
            'time': 0.5,
            'action': action,
            'current_energy': current_energy,
            'min_energy': 0.0,
            'max_energy': 100.0,
            'charge_rate_max': 10.0,
            'discharge_rate_max': 15.0
        }


class TestBattery(unittest.TestCase):
    """Comprehensive unit tests for Battery class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = TestConfig.BATTERY_CONFIG.copy()
        self.dynamics = BatteryDynamics(TestConfig.DYNAMICS_CONFIG)
        self.battery = Battery(self.dynamics, self.config)

    def test_battery_initialization(self):
        """Test battery initialization with valid configuration."""
        self.assertEqual(self.battery.energy_min, 0.0)
        self.assertEqual(self.battery.energy_max, 100.0)
        self.assertEqual(self.battery.charge_rate_max, 10.0)
        self.assertEqual(self.battery.discharge_rate_max, 15.0)
        self.assertEqual(self.battery.charge_efficiency, 0.9)
        self.assertEqual(self.battery.discharge_efficiency, 0.85)
        self.assertEqual(self.battery.initial_energy, 50.0)
        self.assertEqual(self.battery.energy_level, 50.0)

    def test_battery_initialization_missing_config(self):
        """Test battery initialization with missing configuration parameters."""
        incomplete_config = {"min": 0.0, "max": 100.0}
        with self.assertRaises(ValueError) as context:
            Battery(self.dynamics, incomplete_config)
        self.assertIn("Missing required Battery config keys", str(context.exception))

    def test_battery_initialization_invalid_values(self):
        """Test battery initialization with various invalid values."""
        test_cases = [
            # (config_override, expected_error_text)
            ({"min": 100.0, "max": 50.0}, "'min' must be < 'max'"),
            ({"charge_rate_max": -5.0}, "Rate limits must be positive"),
            ({"discharge_rate_max": 0.0}, "Rate limits must be positive"),
            ({"charge_efficiency": 1.5}, "charge_efficiency must be in (0, 1]"),
            ({"charge_efficiency": 0.0}, "charge_efficiency must be in (0, 1]"),
            ({"discharge_efficiency": 1.1}, "discharge_efficiency must be in (0, 1]"),
        ]
        
        for config_override, expected_error in test_cases:
            with self.subTest(config_override=config_override):
                invalid_config = self.config.copy()
                invalid_config.update(config_override)
                with self.assertRaises(ValueError) as context:
                    Battery(self.dynamics, invalid_config)
                self.assertIn(expected_error, str(context.exception))

    def test_battery_initialization_clamp_initial_energy(self):
        """Test battery initialization clamps initial energy to bounds."""
        config_with_out_of_bounds_init = self.config.copy()
        config_with_out_of_bounds_init["init"] = 150.0  # Above max
        battery = Battery(self.dynamics, config_with_out_of_bounds_init)
        self.assertEqual(battery.energy_level, 100.0)  # Clamped to max

    def test_perform_action_basic_operations(self):
        """Test battery charging, discharging, and zero actions."""
        test_cases = [
            # (amount, expected_energy_change, description)
            (5.0, 5.0 * 0.9, "charging"),
            (-8.0, -(8.0 * 0.85), "discharging"),
            (0.0, 0.0, "zero amount"),
        ]
        
        for amount, expected_change, description in test_cases:
            with self.subTest(description=description):
                # Reset to known state
                self.battery.energy_level = 50.0
                initial_energy = self.battery.energy_level
                
                action = Action(id=ActionType.STORAGE, amount=amount)
                self.battery.perform_action(action)
                
                expected_energy = initial_energy + expected_change
                self.assertAlmostEqual(self.battery.energy_level, expected_energy, places=6)
                self.assertAlmostEqual(self.battery.energy_change, expected_change, places=6)

    def test_perform_action_invalid_inputs(self):
        """Test battery action with various invalid inputs."""
        test_cases = [
            # (action, expected_error_text)
            (Action(id=ActionType.PRODUCTION, amount=5.0), "Battery only accepts ActionType.STORAGE actions"),
            ("not_an_action", "action must be an Action dataclass instance"),
            (Action(id=ActionType.STORAGE, amount=float('inf')), "Action amount must be a finite float"),
        ]
        
        for action, expected_error in test_cases:
            with self.subTest(action=action):
                with self.assertRaises(ValueError) as context:
                    self.battery.perform_action(action)
                self.assertIn(expected_error, str(context.exception))

    def test_energy_boundary_clamping(self):
        """Test energy level clamping to min/max bounds."""
        test_cases = [
            # (initial_energy, action_amount, expected_final_energy, description)
            (98.0, 10.0, 100.0, "clamping to max"),
            (2.0, -10.0, 0.0, "clamping to min"),
        ]
        
        for initial_energy, action_amount, expected_final, description in test_cases:
            with self.subTest(description=description):
                self.battery.energy_level = initial_energy
                action = Action(id=ActionType.STORAGE, amount=action_amount)
                self.battery.perform_action(action)
                self.assertEqual(self.battery.energy_level, expected_final)

    def test_get_state(self):
        """Test getting battery state."""
        self.assertEqual(self.battery.get_state(), 50.0)
        
        # Change state and test again
        self.battery.energy_level = 75.0
        self.assertEqual(self.battery.get_state(), 75.0)

    def test_update_with_action(self):
        """Test battery update with action."""
        action = Action(id=ActionType.STORAGE, amount=3.0)
        initial_energy = self.battery.energy_level
        
        self.battery.update(time=0.5, action=action)
        
        self.assertEqual(self.battery.current_time, 0.5)
        expected_energy = initial_energy + (3.0 * 0.9)
        self.assertAlmostEqual(self.battery.energy_level, expected_energy, places=6)

    def test_update_without_action(self):
        """Test battery update without action (defaults to zero)."""
        initial_energy = self.battery.energy_level
        
        self.battery.update(time=0.3)
        
        self.assertEqual(self.battery.current_time, 0.3)
        self.assertEqual(self.battery.energy_level, initial_energy)  # No change

    def test_update_invalid_time(self):
        """Test battery update with invalid time value."""
        with self.assertRaises(ValueError) as context:
            self.battery.update(time=float('nan'))
        self.assertIn("time must be a finite float", str(context.exception))

    def test_reset_functionality(self):
        """Test battery reset with different scenarios."""
        # Change the battery state
        self.battery.energy_level = 75.0
        self.battery.energy_change = 25.0
        
        test_cases = [
            # (reset_level, expected_final_level, description)
            (None, 50.0, "default reset"),
            (25.0, 25.0, "custom level reset"),
            (150.0, 100.0, "clamping to max on reset"),
            (-10.0, 0.0, "clamping to min on reset"),
        ]
        
        for reset_level, expected_level, description in test_cases:
            with self.subTest(description=description):
                if reset_level is None:
                    self.battery.reset()
                else:
                    self.battery.reset(initial_level=reset_level)
                self.assertEqual(self.battery.energy_level, expected_level)
                self.assertEqual(self.battery.energy_change, 0.0)

    def test_reset_invalid_level(self):
        """Test battery reset with invalid level."""
        with self.assertRaises(ValueError) as context:
            self.battery.reset(initial_level=float('inf'))
        self.assertIn("initial_level must be a finite float", str(context.exception))

    @patch('energy_net.grid_entities.storage.battery.setup_logger')
    def test_logging_initialization(self, mock_logger):
        """Test that logger is properly initialized."""
        Battery(self.dynamics, self.config)
        mock_logger.assert_called_with("Battery", "logs/storage.log")

    def test_dynamics_error_handling(self):
        """Test error handling when dynamics fails."""
        # Mock dynamics that raises TypeError on first call, succeeds on second
        mock_dynamics = Mock()
        mock_dynamics.calculate.side_effect = [TypeError("Unsupported parameter"), 50.0]
        
        battery = Battery(mock_dynamics, self.config)
        action = Action(id=ActionType.STORAGE, amount=5.0)
        
        # Should not raise exception, should handle gracefully
        battery.perform_action(action)
        
        # Verify that calculate was called twice (first with all params, then without efficiency params)
        self.assertEqual(mock_dynamics.calculate.call_count, 2)

    def test_dynamics_non_finite_return(self):
        """Test handling of non-finite values from dynamics."""
        # Mock dynamics that returns infinity
        mock_dynamics = Mock()
        mock_dynamics.calculate.return_value = float('inf')
        
        battery = Battery(mock_dynamics, self.config)
        action = Action(id=ActionType.STORAGE, amount=5.0)
        
        with self.assertRaises(ValueError) as context:
            battery.perform_action(action)
        self.assertIn("Dynamics returned a non-finite energy value", str(context.exception))


class TestBatteryDynamics(unittest.TestCase):
    """Unit tests for BatteryDynamics class."""

    def test_battery_dynamics_initialization_and_validation(self):
        """Test BatteryDynamics initialization with various configurations."""
        # Valid config
        valid_config = TestConfig.DYNAMICS_CONFIG
        dynamics = BatteryDynamics(valid_config)
        self.assertEqual(dynamics.model_parameters, valid_config)
        self.assertEqual(dynamics.charge_efficiency, 0.9)
        self.assertEqual(dynamics.discharge_efficiency, 0.85)
        
        # Invalid configs
        invalid_configs = [
            ({'charge_efficiency': 0.9}, "Missing required parameter 'discharge_efficiency'"),
            ({'charge_efficiency': 1.5, 'discharge_efficiency': 0.85}, "charge_efficiency must be in range (0, 1]"),
            ({'charge_efficiency': 0.9, 'discharge_efficiency': 0.0}, "discharge_efficiency must be in range (0, 1]"),
        ]
        
        for config, expected_error in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(AssertionError) as context:
                    BatteryDynamics(config)
                self.assertIn(expected_error, str(context.exception))

    def test_battery_dynamics_reset(self):
        """Test that reset method works correctly."""
        dynamics = BatteryDynamics(TestConfig.DYNAMICS_CONFIG)
        # Should not raise any exception
        dynamics.reset()


class TestBatteryDynamicsImplementation(unittest.TestCase):
    """Unit tests for BatteryDynamics implementation class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dynamics = BatteryDynamics(TestConfig.DYNAMICS_CONFIG)

    def test_battery_dynamics_implementation_initialization(self):
        """Test BatteryDynamics initialization."""
        self.assertEqual(self.dynamics.charge_efficiency, 0.9)
        self.assertEqual(self.dynamics.discharge_efficiency, 0.85)

    def test_calculate_basic_operations(self):
        """Test calculation with different action types."""
        test_cases = [
            # (amount, expected_result, description)
            (5.0, 50.0 + (5.0 * 0.9), "charging action"),
            (-8.0, 50.0 - (8.0 * 0.85), "discharging action"),
            (0.0, 50.0, "zero action"),
        ]
        
        for amount, expected, description in test_cases:
            with self.subTest(description=description):
                action = Action(id=ActionType.STORAGE, amount=amount)
                kwargs = TestConfig.get_standard_kwargs(action)
                result = self.dynamics.calculate(**kwargs)
                self.assertAlmostEqual(result, expected, places=6)

    def test_calculate_rate_limits(self):
        """Test calculation respects rate limits."""
        test_cases = [
            # (amount, rate_type, expected_error)
            (15.0, "charge", "Charging action exceeds maximum charge rate"),
            (-20.0, "discharge", "Discharging action exceeds maximum discharge rate"),
        ]
        
        for amount, rate_type, expected_error in test_cases:
            with self.subTest(rate_type=rate_type):
                action = Action(id=ActionType.STORAGE, amount=amount)
                kwargs = TestConfig.get_standard_kwargs(action)
                with self.assertRaises(AssertionError) as context:
                    self.dynamics.calculate(**kwargs)
                self.assertIn(expected_error, str(context.exception))

    def test_calculate_validation_errors(self):
        """Test calculation with various validation errors."""
        # Missing required kwargs
        action = Action(id=ActionType.STORAGE, amount=5.0)
        incomplete_kwargs = {'time': 0.5, 'action': action, 'current_energy': 50.0}
        with self.assertRaises(AssertionError) as context:
            self.dynamics.calculate(**incomplete_kwargs)
        self.assertIn("Missing required argument", str(context.exception))
        
        # Invalid action type
        invalid_kwargs = TestConfig.get_standard_kwargs("not_an_action")
        with self.assertRaises(ValueError) as context:
            self.dynamics.calculate(**invalid_kwargs)
        self.assertIn("action must be an Action dataclass instance", str(context.exception))

    def test_exp_mult_function(self):
        """Test the exp_mult static method."""
        # Test normal decay
        result = BatteryDynamics.exp_mult(100.0, 10.0, 5)
        expected = 100.0 * math.exp(-5.0 / 10.0)
        self.assertAlmostEqual(result, expected, places=6)

        # Test with zero time step
        result = BatteryDynamics.exp_mult(100.0, 10.0, 0)
        self.assertAlmostEqual(result, 100.0, places=6)

        # Test extreme clamping (high time step)
        result = BatteryDynamics.exp_mult(100.0, 0.1, 1000)
        self.assertGreater(result, 0)  # Should not be zero due to clamping

    def test_exp_mult_invalid_lifetime_constant(self):
        """Test exp_mult with invalid lifetime constant."""
        with self.assertRaises(ValueError) as context:
            BatteryDynamics.exp_mult(100.0, 0.0, 5)
        self.assertIn("Lifetime constant must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            BatteryDynamics.exp_mult(100.0, -5.0, 5)
        self.assertIn("Lifetime constant must be positive", str(context.exception))

    def test_edge_cases_and_limits(self):
        """Test edge cases including very small values and rate limits."""
        # Very small values
        small_test_cases = [
            (1e-10, 50.0 + (1e-10 * 0.9), "very small charge"),
            (-1e-10, 50.0 - (1e-10 * 0.85), "very small discharge"),
        ]
        
        for amount, expected, description in small_test_cases:
            with self.subTest(description=description):
                action = Action(id=ActionType.STORAGE, amount=amount)
                kwargs = TestConfig.get_standard_kwargs(action)
                result = self.dynamics.calculate(**kwargs)
                self.assertAlmostEqual(result, expected, places=15)
        
        # Rate limits
        rate_limit_cases = [
            (10.0, 50.0 + (10.0 * 0.9), "charging at max rate"),
            (-15.0, 50.0 - (15.0 * 0.85), "discharging at max rate"),
        ]
        
        for amount, expected, description in rate_limit_cases:
            with self.subTest(description=description):
                action = Action(id=ActionType.STORAGE, amount=amount)
                kwargs = TestConfig.get_standard_kwargs(action)
                result = self.dynamics.calculate(**kwargs)
                self.assertAlmostEqual(result, expected, places=6)


if __name__ == '__main__':
    unittest.main()



