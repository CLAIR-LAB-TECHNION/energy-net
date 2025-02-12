# tests/test_isos.py

import unittest
from typing import  List
from energy_net.dynamics.iso.hourly_pricing_iso import HourlyPricingISO
from energy_net.dynamics.iso.random_pricing_iso import RandomPricingISO
from energy_net.dynamics.iso.quadratic_pricing_iso import QuadraticPricingISO
from energy_net.dynamics.iso.time_of_use_pricing_iso import TimeOfUsePricingISO
from energy_net.dynamics.iso.dynamic_pricing_iso import DynamicPricingISO
from energy_net.dynamics.iso.fixed_pricing_iso import FixedPricingISO

class TestHourlyPricingISO(unittest.TestCase):
    def setUp(self):
        self.hourly_rates = {hour: 50.0 for hour in range(24)}
        self.iso = HourlyPricingISO(hourly_rates=self.hourly_rates)

    def test_pricing_function_peak_hour(self):
        observation = {'time': 0.5}  # 12:00 PM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.hourly_rates[12]
        reward = pricing_func(10)
        expected_reward = (10 * expected_price_buy) 
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_default_price(self):
        observation = {'time': 1.0}  # 24:00 or 0:00 AM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.hourly_rates[0]
        reward = pricing_func(0)
        expected_reward = 0.0
        self.assertEqual(reward, expected_reward)


class TestRandomPricingISO(unittest.TestCase):
    def setUp(self):
        self.min_price = 40.0
        self.max_price = 60.0
        self.iso = RandomPricingISO(min_price=self.min_price, max_price=self.max_price)

    def test_pricing_function_range(self):
        observation = {}
        pricing_func = self.iso.get_pricing_function(observation)
        # Since prices are random, we check if they fall within the expected range
        # For unit tests, it's better to mock randomness to have deterministic tests
        # However, for simplicity, we'll perform a basic range check here
        buy = 10
        reward = pricing_func(buy)
        # Assuming price_buy is between min_price and max_price
        # And price_sell is between 80% and 95% of price_buy
        # So reward should be between:
        min_reward = (buy * self.min_price) 
        max_reward = (buy * self.max_price)   # Since sell_price could be as low as 80% of buy
        self.assertTrue(min_reward <= reward <= (buy * self.max_price))


class TestQuadraticPricingISO(unittest.TestCase):
    def setUp(self):
        self.a = 1.0
        self.b = 0.0
        self.c = 50.0
        self.iso = QuadraticPricingISO(a=self.a, b=self.b, c=self.c)

    def test_pricing_function_zero_demand(self):
        observation = {'demand': 0.0}
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.c  # 50.0
        reward = pricing_func(10)
        expected_reward = (10 * expected_price_buy) 
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_positive_demand(self):
        observation = {'demand': 2.0}
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.a * (2.0 ** 2) + self.b * 2.0 + self.c  # 1*4 + 0*2 + 50 = 54
        reward = pricing_func(10)
        expected_reward = (10 * 54) 
        self.assertAlmostEqual(reward, expected_reward, places=2)


class TestTimeOfUsePricingISO(unittest.TestCase):
    def setUp(self):
        self.peak_hours = [17, 18, 19]
        self.off_peak_hours = [0, 1, 2, 3, 4, 5]
        self.peak_price = 60.0
        self.off_peak_price = 30.0
        self.iso = TimeOfUsePricingISO(
            peak_hours=self.peak_hours,
            off_peak_hours=self.off_peak_hours,
            peak_price=self.peak_price,
            off_peak_price=self.off_peak_price
        )

    def test_pricing_function_peak_hour(self):
        observation = {'time': 17 / 24}  # 17:00 or 5:00 PM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.peak_price
        reward = pricing_func(10)
        expected_reward = (10 * self.peak_price)
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_off_peak_hour(self):
        observation = {'time': 4 / 24}  # 4:00 AM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = self.off_peak_price
        reward = pricing_func(10)
        expected_reward = (10 * self.off_peak_price) 
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_default_hour(self):
        observation = {'time': 12 / 24}  # 12:00 PM
        pricing_func = self.iso.get_pricing_function(observation)
        expected_price_buy = (self.peak_price + self.off_peak_price) / 2  # 45.0
        reward = pricing_func(10)
        expected_reward = (10 * 45.0) 
        self.assertEqual(reward, expected_reward)


class TestDynamicPricingISO(unittest.TestCase):
    def setUp(self):
        self.base_price = 50.0
        self.demand_factor = 1.0
        self.supply_factor = 1.0
        self.elasticity = 0.5
        self.iso = DynamicPricingISO(
            base_price=self.base_price,
            demand_factor=self.demand_factor,
            supply_factor=self.supply_factor,
            elasticity=self.elasticity
        )

    def test_pricing_function_equal_demand_supply(self):
        observation = {'demand': 1.0, 'supply': 1.0}
        pricing_func = self.iso.get_pricing_function(observation)
        # price_buy = 50 * (1 + 0.5*(1 - 1)) = 50
        expected_price_buy = self.base_price
        reward = pricing_func(10)
        expected_reward = (10 * 50.0) 
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_high_demand(self):
        observation = {'demand': 2.0, 'supply': 1.0}
        pricing_func = self.iso.get_pricing_function(observation)
        # price_buy = 50 * (1 + 0.5*(2 - 1)) = 50 * 1.5 = 75
        expected_price_buy = 75.0
        
        reward = pricing_func(10)
        expected_reward = (10 * 75.0)
        self.assertEqual(reward, expected_reward)

    def test_pricing_function_low_demand(self):
        observation = {'demand': 0.5, 'supply': 1.0}
        pricing_func = self.iso.get_pricing_function(observation)
        # price_buy = 50 * (1 + 0.5*(0.5 - 1)) = 50 * 0.75 = 37.5
        expected_price_buy = 37.5
        reward = pricing_func(10)
        expected_reward = (10 * 37.5) 
        self.assertEqual(reward, expected_reward)
        
class TestFixedPricingISO(unittest.TestCase):
    def setUp(self):
        # Define a fixed pricing schedule for testing
        self.pricing_schedule: List[float] = [50.0, 52.0, 51.5, 53.0, 54.0]
        self.iso = FixedPricingISO(pricing_schedule=self.pricing_schedule)
    
    def test_initialization_with_valid_parameters(self):
        # Test that ISO initializes correctly with a valid pricing schedule
        self.assertEqual(self.iso.pricing_schedule, self.pricing_schedule)
        self.assertEqual(self.iso.current_step, 0)
        self.assertEqual(self.iso.episode_length, 5)
    
    def test_initialization_missing_pricing_schedule(self):
        # Test that ISO raises ValueError if pricing_schedule is missing
        with self.assertRaises(TypeError):
            # Missing 'pricing_schedule' argument
            FixedPricingISO()
    
    def test_initialization_invalid_pricing_schedule_type(self):
        # Test that ISO handles invalid pricing_schedule types gracefully
        with self.assertRaises(TypeError):
            FixedPricingISO(pricing_schedule="invalid_schedule")  # Should be a list
            
    def test_initialization_invalid_pricing_schedule_elements(self):
        # Test that ISO raises ValueError if elements in pricing_schedule are not numbers
        invalid_pricing_schedule = [50.0, "52.0", 51.5, None, 54.0]
        with self.assertRaises(ValueError):
            FixedPricingISO(pricing_schedule=invalid_pricing_schedule)
    
    def test_get_pricing_function_returns_correct_prices(self):
        # Test that the pricing function returns the correct prices step by step
        for expected_price in self.pricing_schedule:
            pricing_function = self.iso.get_pricing_function(observation={})
            price = pricing_function(action_amount=1.0)  # action_amount is arbitrary
            self.assertEqual(price, expected_price)
    
    def test_get_pricing_function_after_episode_length(self):
        # Test that after the episode length, the last price is returned
        for _ in range(len(self.pricing_schedule)):
            pricing_function = self.iso.get_pricing_function(observation={})
            pricing_function(action_amount=1.0)
        
        # Now, any further calls should return the last price
        pricing_function = self.iso.get_pricing_function(observation={})
        price = pricing_function(action_amount=1.0)
        self.assertEqual(price, self.pricing_schedule[-1])
    
    def test_reset_method(self):
        # Test that reset restores the ISO to the initial state
        for _ in range(3):
            pricing_function = self.iso.get_pricing_function(observation={})
            pricing_function(action_amount=1.0)
        
        self.iso.reset()
        self.assertEqual(self.iso.current_step, 0)
        
        # After reset, pricing should start from the beginning again
        pricing_function = self.iso.get_pricing_function(observation={})
        price = pricing_function(action_amount=1.0)
        self.assertEqual(price, self.pricing_schedule[0])
    
    def test_update_pricing_schedule(self):
        # Test updating the pricing schedule
        new_pricing_schedule: List[float] = [55.0, 56.0, 57.0]
        self.iso.pricing_schedule = new_pricing_schedule
        self.iso.episode_length = len(new_pricing_schedule)
        self.iso.current_step = 0
        
        for expected_price in new_pricing_schedule:
            pricing_function = self.iso.get_pricing_function(observation={})
            price = pricing_function(action_amount=1.0)
            self.assertEqual(price, expected_price)
        
        # Beyond the new schedule length
        pricing_function = self.iso.get_pricing_function(observation={})
        price = pricing_function(action_amount=1.0)
        self.assertEqual(price, new_pricing_schedule[-1])


if __name__ == '__main__':
    unittest.main()
