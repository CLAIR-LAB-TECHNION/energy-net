import unittest
from unittest.mock import MagicMock, Mock
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.grid_entities.consumption.consumption_dynamics import ModelDrivenConsumptionDynamics
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.foundation.model import State, Action
from energy_net.consumption_prediction.predicting_consumption_model import EnergyPredictor
import os


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

    def test_get_state(self):
        # get_state() returns numeric consumption value
        state_value = self.consumption_unit.get_state()
        self.assertIsInstance(state_value, (int, float))  # allow int or float
        self.assertEqual(state_value, 0.0)

    def test_update_with_state(self):
        # Passing State object
        state = State({'time': 0.5})
        action = Action({'value': 0.0})
        self.consumption_unit.update(state, action)
        self.assertEqual(self.consumption_unit.current_consumption, 50.0)
        self.mock_dynamics.get_value.assert_called_with(time=0.5, action=0.0)

    def test_update_with_state_no_action(self):
        # State object without action
        state = State({'time': 0.5})
        self.consumption_unit.update(state)
        self.assertEqual(self.consumption_unit.current_consumption, 50.0)
        self.mock_dynamics.get_value.assert_called_with(time=0.5, action=0.0)

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

    def test_reset(self):
        # Update to change state
        state = State({'time': 0.5})
        self.consumption_unit.update(state)
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
        # Should default to 0.0 
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


class TestModelDrivenConsumptionDynamics(unittest.TestCase):
    """Test suite for ModelDrivenConsumptionDynamics class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and train a predictor model once for all tests."""
        # Use the existing test data
        cls.test_data_path = os.path.join(
            os.path.dirname(__file__), 
            '../gym/data_for_tests/synthetic_household_consumption.csv'
        )
        
        # Train a simple predictor on the test data
        print("\nTraining test predictor model...")
        cls.predictor = EnergyPredictor(cls.test_data_path)
        print("Test predictor ready!")
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.reference_date = '2025-01-01'
        self.params = {
            'model': self.predictor,
            'reference_date': self.reference_date
        }
    
    def test_initialization_with_valid_params(self):
        """Test that ModelDrivenConsumptionDynamics initializes correctly with valid parameters."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        
        self.assertIsNotNone(dynamics.model)
        self.assertEqual(dynamics.model, self.predictor)
        self.assertIsNotNone(dynamics.reference_date)
    
    def test_initialization_missing_model(self):
        """Test that initialization fails when model is missing."""
        invalid_params = {'reference_date': self.reference_date}
        
        with self.assertRaises(ValueError) as context:
            ModelDrivenConsumptionDynamics(invalid_params)
        
        self.assertIn('model', str(context.exception).lower())
    
    def test_initialization_missing_reference_date(self):
        """Test that initialization fails when reference_date is missing."""
        invalid_params = {'model': self.predictor}
        
        with self.assertRaises(ValueError) as context:
            ModelDrivenConsumptionDynamics(invalid_params)
        
        self.assertIn('reference_date', str(context.exception).lower())
    
    def test_initialization_invalid_model(self):
        """Test that initialization fails when model doesn't have predict method."""
        invalid_params = {
            'model': "not a model",
            'reference_date': self.reference_date
        }
        
        with self.assertRaises(ValueError) as context:
            ModelDrivenConsumptionDynamics(invalid_params)
        
        self.assertIn('predict', str(context.exception).lower())
    
    def test_initialization_invalid_date_format(self):
        """Test that initialization fails with invalid date format."""
        invalid_params = {
            'model': self.predictor,
            'reference_date': 'invalid-date'
        }
        
        with self.assertRaises(ValueError) as context:
            ModelDrivenConsumptionDynamics(invalid_params)
        
        self.assertIn('parse', str(context.exception).lower())
    
    def test_time_conversion_day_zero_midnight(self):
        """Test time conversion for day 0 at midnight (time=0.0)."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        date_str, time_str = dynamics._convert_time_to_datetime(0.0)
        
        self.assertEqual(date_str, '2025-01-01')
        self.assertEqual(time_str, '00:00')
    
    def test_time_conversion_day_zero_noon(self):
        """Test time conversion for day 0 at noon (time=0.5)."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        date_str, time_str = dynamics._convert_time_to_datetime(0.5)
        
        self.assertEqual(date_str, '2025-01-01')
        self.assertEqual(time_str, '12:00')
    
    def test_time_conversion_day_one_midnight(self):
        """Test time conversion for day 1 at midnight (time=1.0)."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        date_str, time_str = dynamics._convert_time_to_datetime(1.0)
        
        self.assertEqual(date_str, '2025-01-02')
        self.assertEqual(time_str, '00:00')
    
    def test_time_conversion_day_one_afternoon(self):
        """Test time conversion for day 1 at 14:00 (time=1.583...)."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        # 14:00 = 14/24 ≈ 0.583
        time_value = 1.0 + (14.0 / 24.0)
        date_str, time_str = dynamics._convert_time_to_datetime(time_value)
        
        self.assertEqual(date_str, '2025-01-02')
        self.assertEqual(time_str, '14:00')
    
    def test_time_conversion_day_five_morning(self):
        """Test time conversion for day 5 at 06:00 (time=5.25)."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        date_str, time_str = dynamics._convert_time_to_datetime(5.25)
        
        self.assertEqual(date_str, '2025-01-06')
        self.assertEqual(time_str, '06:00')
    
    def test_time_conversion_with_minutes(self):
        """Test time conversion with non-zero minutes."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        # 14:30 = (14*60 + 30) / (24*60) = 870/1440
        time_value = 2.0 + (870.0 / 1440.0)
        date_str, time_str = dynamics._convert_time_to_datetime(time_value)
        
        self.assertEqual(date_str, '2025-01-03')
        self.assertEqual(time_str, '14:30')
    
    def test_get_value_day_zero(self):
        """Test get_value returns valid consumption for day 0."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        consumption = dynamics.get_value(time=0.0)
        
        self.assertIsInstance(consumption, float)
        self.assertGreater(consumption, 0)
    
    def test_get_value_day_zero_noon(self):
        """Test get_value returns valid consumption for day 0 at noon."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        consumption = dynamics.get_value(time=0.5)
        
        self.assertIsInstance(consumption, float)
        self.assertGreater(consumption, 0)
    
    def test_get_value_multiple_days(self):
        """Test get_value returns valid consumption across multiple days."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        
        # Test various time points
        test_times = [0.0, 0.5, 1.0, 1.583, 5.25, 10.0]
        
        for time_val in test_times:
            with self.subTest(time=time_val):
                consumption = dynamics.get_value(time=time_val)
                self.assertIsInstance(consumption, float)
                self.assertGreater(consumption, 0)
    
    def test_get_value_calls_model_predict(self):
        """Test that get_value correctly calls model.predict with proper arguments."""
        # Create a mock model
        mock_model = Mock()
        mock_model.predict.return_value = 42.5
        
        params = {
            'model': mock_model,
            'reference_date': '2025-01-01'
        }
        
        dynamics = ModelDrivenConsumptionDynamics(params)
        consumption = dynamics.get_value(time=1.5)
        
        # Verify model.predict was called
        mock_model.predict.assert_called_once()
        
        # Verify the arguments (day 1 at noon = Jan 2, 12:00)
        call_args = mock_model.predict.call_args[0]
        self.assertEqual(call_args[0], '2025-01-02')
        self.assertEqual(call_args[1], '12:00')
        
        # Verify return value
        self.assertEqual(consumption, 42.5)
    
    def test_consumption_unit_with_model_dynamics(self):
        """Test ConsumptionUnit integration with ModelDrivenConsumptionDynamics."""
        # Create dynamics with trained model
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        
        # Create ConsumptionUnit
        config = {'consumption_capacity': 100.0}
        consumption_unit = ConsumptionUnit(dynamics=dynamics, config=config)
        
        # Initial state check
        self.assertEqual(consumption_unit.current_consumption, 0.0)
        
        # Update with time=0.5 (day 0, noon)
        state = State({'time': 0.5})
        consumption_unit.update(state)
        
        # Verify consumption was updated
        self.assertGreater(consumption_unit.current_consumption, 0.0)
        self.assertIsInstance(consumption_unit.current_consumption, float)
        
        # Update with different time
        state2 = State({'time': 2.0})
        consumption_unit.update(state2)
        
        # Verify consumption changed
        new_consumption = consumption_unit.current_consumption
        self.assertGreater(new_consumption, 0.0)
    
    def test_consumption_unit_tracks_state_with_model(self):
        """Test that ConsumptionUnit correctly tracks state when using model dynamics."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        config = {'consumption_capacity': 100.0}
        consumption_unit = ConsumptionUnit(dynamics=dynamics, config=config)
        
        # Update multiple times
        times = [0.25, 0.5, 1.0, 1.5, 2.75]
        consumption_values = []
        
        for time_val in times:
            state = State({'time': time_val})
            consumption_unit.update(state)
            consumption_values.append(consumption_unit.current_consumption)
            
            # Verify internal state is updated
            self.assertEqual(consumption_unit._state.get_attribute('time'), time_val)
            self.assertEqual(
                consumption_unit._state.get_attribute('consumption'), 
                consumption_unit.current_consumption
            )
        
        # Verify we got different consumption values (model is working)
        self.assertEqual(len(set(consumption_values)), len(times))
    
    def test_reset_functionality(self):
        """Test that dynamics reset works correctly."""
        dynamics = ModelDrivenConsumptionDynamics(self.params)
        
        # Get a value to ensure dynamics is initialized
        consumption = dynamics.get_value(time=1.0)
        self.assertGreater(consumption, 0)
        
        # Reset should not raise an error
        dynamics.reset()
    
    def test_model_prediction_with_different_reference_dates(self):
        """Test that different reference dates produce correct date conversions."""
        params_2024 = {
            'model': self.predictor,
            'reference_date': '2024-06-15'
        }
        
        dynamics = ModelDrivenConsumptionDynamics(params_2024)
        date_str, time_str = dynamics._convert_time_to_datetime(0.0)
        
        self.assertEqual(date_str, '2024-06-15')
        self.assertEqual(time_str, '00:00')
        
        # Test a different day
        date_str, time_str = dynamics._convert_time_to_datetime(10.0)
        self.assertEqual(date_str, '2024-06-25')

if __name__ == '__main__':
    unittest.main()
