"""
Unit test to verify asymmetric buy/sell pricing functionality.

This test validates that when asymmetric pricing is enabled:
- ISO learns separate buy and sell prices independently
- Buy prices come from action[0:48]
- Sell prices come from action[48:96]
"""

import unittest
import numpy as np
from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.grid_entities.management.price_curve import RLPriceCurveStrategy
from unittest.mock import Mock

class TestAsymmetricPricing(unittest.TestCase):
    """Test suite to verify asymmetric buy/sell pricing."""
    
    def test_asymmetric_separate_prices(self):
        """
        Test that asymmetric pricing uses separate learned buy and sell prices.
        
        With use_asymmetric_pricing=True:
        - Buy prices come from action[0:48]
        - Sell prices come from action[48:96]
        - These are independent values learned by the agent
        """
        # Create a mock ISO model that returns different buy/sell prices
        mock_iso_model = Mock()
        # First 48: buy prices (all 0.0 = minimum)
        # Second 48: sell prices (all 1.0 = maximum)
        # Last 48: dispatch values
        mock_action = np.concatenate([
            np.full(48, 0.0, dtype=np.float32),  # Buy prices at minimum
            np.full(48, 1.0, dtype=np.float32),  # Sell prices at maximum
            np.full(48, 2.0, dtype=np.float32)   # Dispatch (not used for pricing)
        ])
        mock_iso_model.predict.return_value = (mock_action, None)
        
        # Create asymmetric pricing strategy
        asymmetric_strategy = RLPriceCurveStrategy(
            iso_model=mock_iso_model,
            price_min=0.0,
            price_max=0.20,
            use_asymmetric_pricing=True
        )
        
        # Create dummy input (384 dimensions: 48 predictions + 336 features)
        iso_input = np.random.rand(384).astype(np.float32)
        
        # Calculate buy and sell prices
        buy_prices = asymmetric_strategy.calculate_buy_price(iso_input)
        sell_prices = asymmetric_strategy.calculate_sell_price(iso_input)
        
        print(f"\n{'='*70}")
        print(f"Asymmetric Pricing Test Results")
        print(f"{'='*70}")
        print(f"\nBuy Prices (from action[0:48]):")
        print(f"  First 3: {buy_prices[:3]}")
        print(f"  Expected: all $0.00 (minimum)")
        print(f"\nSell Prices (from action[48:96]):")
        print(f"  First 3: {sell_prices[:3]}")
        print(f"  Expected: all $0.20 (maximum)")
        print(f"{'='*70}")
        
        # Verify buy prices are at minimum (action[0:48] = 0.0)
        expected_buy = np.full(48, 0.0, dtype=np.float32)
        np.testing.assert_array_almost_equal(
            buy_prices,
            expected_buy,
            decimal=4,
            err_msg="Buy prices should be at minimum ($0.00)"
        )
        
        # Verify sell prices are at maximum (action[48:96] = 1.0)
        expected_sell = np.full(48, 0.20, dtype=np.float32)
        np.testing.assert_array_almost_equal(
            sell_prices,
            expected_sell,
            decimal=4,
            err_msg="Sell prices should be at maximum ($0.20)"
        )
        
        print(f"✓  Buy and sell prices correctly use separate action outputs!")
        print(f"{'='*70}\n")
    
    def test_symmetric_pricing_default(self):
        """
        Test that symmetric pricing is the default (backward compatibility).
        
        When use_asymmetric_pricing=False (default), buy and sell prices
        should be identical and come from action[0:48].
        """
        # Create a mock ISO model
        mock_iso_model = Mock()
        mock_action = np.concatenate([
            np.full(48, 0.5, dtype=np.float32),  # Prices
            np.full(48, 2.0, dtype=np.float32)   # Dispatch
        ])
        mock_iso_model.predict.return_value = (mock_action, None)
        
        # Create symmetric pricing strategy (default)
        symmetric_strategy = RLPriceCurveStrategy(
            iso_model=mock_iso_model,
            price_min=0.0,
            price_max=0.20,
            use_asymmetric_pricing=False
        )
        
        # Create dummy input
        iso_input = np.random.rand(384).astype(np.float32)
        
        # Calculate prices
        base_prices = symmetric_strategy.calculate_price(iso_input)
        buy_prices = symmetric_strategy.calculate_buy_price(iso_input)
        sell_prices = symmetric_strategy.calculate_sell_price(iso_input)
        
        print(f"\n{'='*70}")
        print(f"Symmetric Pricing Test Results (Default Behavior)")
        print(f"{'='*70}")
        print(f"Asymmetric Pricing Enabled: {symmetric_strategy.use_asymmetric_pricing}")
        print(f"\nSample Prices (first timestep):")
        print(f"  Base Price:  ${base_prices[0]:.4f}")
        print(f"  Buy Price:   ${buy_prices[0]:.4f}")
        print(f"  Sell Price:  ${sell_prices[0]:.4f}")
        print(f"{'='*70}")
        
        # Verify all prices are identical
        np.testing.assert_array_almost_equal(
            base_prices,
            buy_prices,
            decimal=6,
            err_msg="Buy prices should equal base prices in symmetric mode"
        )
        
        np.testing.assert_array_almost_equal(
            base_prices,
            sell_prices,
            decimal=6,
            err_msg="Sell prices should equal base prices in symmetric mode"
        )
        
        # Expected value: 0.5 * 0.20 = 0.10
        expected = np.full(48, 0.10, dtype=np.float32)
        np.testing.assert_array_almost_equal(
            base_prices,
            expected,
            decimal=4,
            err_msg="Prices should be $0.10 (0.5 * 0.20)"
        )
        
        print(f"✓  Buy and sell prices are identical (symmetric pricing)!")
        print(f"{'='*70}\n")
    
    def test_asymmetric_mixed_values(self):
        """
        Test asymmetric pricing with mixed buy/sell values.
        """
        mock_iso_model = Mock()
        # Alternating pattern: buy prices vary, sell prices vary differently
        buy_pattern = np.array([0.0, 1.0] * 24, dtype=np.float32)  # 0, 1, 0, 1, ...
        sell_pattern = np.array([1.0, 0.0] * 24, dtype=np.float32)  # 1, 0, 1, 0, ...
        
        mock_action = np.concatenate([
            buy_pattern,
            sell_pattern,
            np.full(48, 2.0, dtype=np.float32)  # Dispatch
        ])
        mock_iso_model.predict.return_value = (mock_action, None)
        
        asymmetric_strategy = RLPriceCurveStrategy(
            iso_model=mock_iso_model,
            price_min=0.0,
            price_max=0.20,
            use_asymmetric_pricing=True
        )
        
        iso_input = np.random.rand(384).astype(np.float32)
        buy_prices = asymmetric_strategy.calculate_buy_price(iso_input)
        sell_prices = asymmetric_strategy.calculate_sell_price(iso_input)
        
        print(f"\n{'='*70}")
        print(f"Asymmetric Mixed Values Test")
        print(f"{'='*70}")
        print(f"Buy pattern:  [0, 1, 0, 1, ...] → [${buy_prices[0]:.2f}, ${buy_prices[1]:.2f}, ${buy_prices[2]:.2f}, ${buy_prices[3]:.2f}, ...]")
        print(f"Sell pattern: [1, 0, 1, 0, ...] → [${sell_prices[0]:.2f}, ${sell_prices[1]:.2f}, ${sell_prices[2]:.2f}, ${sell_prices[3]:.2f}, ...]")
        print(f"{'='*70}")
        
        # Verify buy prices follow the pattern
        expected_buy = buy_pattern * 0.20  # Scale to [0, 0.20]
        np.testing.assert_array_almost_equal(
            buy_prices,
            expected_buy,
            decimal=4
        )
        
        # Verify sell prices follow the opposite pattern
        expected_sell = sell_pattern * 0.20
        np.testing.assert_array_almost_equal(
            sell_prices,
            expected_sell,
            decimal=4
        )
        
        print(f"✓  Buy and sell prices independently follow their patterns!")
        print(f"{'='*70}\n")

if __name__ == '__main__':
    unittest.main(verbosity=2)