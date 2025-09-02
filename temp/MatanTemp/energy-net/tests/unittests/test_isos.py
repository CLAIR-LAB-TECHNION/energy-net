# tests/test_isos.py

import unittest
import numpy as np
from energy_net.controllers.iso.pricing_strategy import (
    PricingStrategyFactory,
    PricingStrategy,
    QuadraticPricingStrategy,
    ConstantPricingStrategy,
    OnlinePricingStrategy
)
from energy_net.market.pricing.pricing_policy import PricingPolicy


class TestPricingStrategies(unittest.TestCase):
    def setUp(self):
        self.min_price = 40.0
        self.max_price = 60.0
        self.max_steps = 48
        self.config = {
            'quadratic': {
                'dispatch': {'min': 0.0, 'max': 300.0},
                'polynomial': {'min': -100.0, 'max': 100.0}
            },
            'constant': {
                'dispatch': {'min': 0.0, 'max': 300.0}
            },
            'online': {
                'dispatch': {'min': 0.0, 'max': 300.0}
            }
        }

    def test_quadratic_pricing_strategy(self):
        strategy = QuadraticPricingStrategy(
            min_price=self.min_price,
            max_price=self.max_price,
            max_steps_per_episode=self.max_steps,
            config=self.config
        )
        
        # Test action space creation
        action_space = strategy.create_action_space(use_dispatch_action=False)
        self.assertEqual(action_space.shape, (6,))  # 6 coefficients for quadratic functions
        
        # Test action processing
        action = np.array([1.0, 0.0, 50.0, 0.9, 0.0, 45.0], dtype=np.float32)  # [a, b, c] for buy and sell
        buy_price, sell_price, dispatch, first_action = strategy.process_action(
            action=action,
            step_count=1,
            first_action_taken=False,
            predicted_demand=2.0
        )
        
        # Expected prices for demand=2.0:
        # buy_price = 1.0 * (2.0)^2 + 0.0 * 2.0 + 50.0 = 54.0
        # sell_price = 0.9 * buy_price = 48.6
        self.assertAlmostEqual(buy_price, 54.0, places=2)
        self.assertAlmostEqual(sell_price, 48.6, places=2)

    def test_constant_pricing_strategy(self):
        strategy = ConstantPricingStrategy(
            min_price=self.min_price,
            max_price=self.max_price,
            max_steps_per_episode=self.max_steps,
            config=self.config
        )
        
        # Test action space creation
        action_space = strategy.create_action_space(use_dispatch_action=False)
        self.assertEqual(action_space.shape, (2,))  # 2 values for buy and sell prices
        
        # Test action processing
        action = np.array([55.0, 49.5], dtype=np.float32)  # buy and sell prices
        
        # First step to set the prices
        buy_price, sell_price, dispatch, first_action = strategy.process_action(
            action=action,
            step_count=1,
            first_action_taken=False,
            predicted_demand=2.0
        )
        
        # Verify the prices are set correctly
        self.assertEqual(buy_price, 55.0)
        self.assertEqual(sell_price, 49.5)
        
        # Test that prices remain constant in subsequent steps
        buy_price2, sell_price2, dispatch2, first_action2 = strategy.process_action(
            action=action,
            step_count=2,
            first_action_taken=True,
            predicted_demand=3.0
        )
        
        self.assertEqual(buy_price2, 55.0)
        self.assertEqual(sell_price2, 49.5)

    def test_online_pricing_strategy(self):
        strategy = OnlinePricingStrategy(
            min_price=self.min_price,
            max_price=self.max_price,
            max_steps_per_episode=self.max_steps,
            config=self.config
        )
        
        # Test action space creation
        action_space = strategy.create_action_space(use_dispatch_action=False)
        self.assertEqual(action_space.shape, (2,))  # 2 values for buy and sell prices
        
        # Test action processing
        action = np.array([52.0, 46.8], dtype=np.float32)  # buy and sell prices
        buy_price, sell_price, dispatch, first_action = strategy.process_action(
            action=action,
            step_count=1,
            first_action_taken=False,
            predicted_demand=2.0
        )
        
        self.assertAlmostEqual(buy_price, 52.0, places=2)
        self.assertAlmostEqual(sell_price, 46.8, places=2)

    def test_pricing_strategy_factory(self):
        action_spaces_config = {
            'quadratic': self.config['quadratic'],
            'constant': self.config['constant'],
            'online': self.config['online']
        }
        
        # Test quadratic strategy creation
        quadratic_strategy = PricingStrategyFactory.create_strategy(
            pricing_policy=PricingPolicy.QUADRATIC,
            min_price=self.min_price,
            max_price=self.max_price,
            max_steps_per_episode=self.max_steps,
            action_spaces_config=action_spaces_config
        )
        self.assertIsInstance(quadratic_strategy, QuadraticPricingStrategy)
        
        # Test constant strategy creation
        constant_strategy = PricingStrategyFactory.create_strategy(
            pricing_policy=PricingPolicy.CONSTANT,
            min_price=self.min_price,
            max_price=self.max_price,
            max_steps_per_episode=self.max_steps,
            action_spaces_config=action_spaces_config
        )
        self.assertIsInstance(constant_strategy, ConstantPricingStrategy)
        
        # Test online strategy creation
        online_strategy = PricingStrategyFactory.create_strategy(
            pricing_policy=PricingPolicy.ONLINE,
            min_price=self.min_price,
            max_price=self.max_price,
            max_steps_per_episode=self.max_steps,
            action_spaces_config=action_spaces_config
        )
        self.assertIsInstance(online_strategy, OnlinePricingStrategy)


if __name__ == '__main__':
    unittest.main()
