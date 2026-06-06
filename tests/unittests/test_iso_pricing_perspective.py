"""
Unit test to verify ISO pricing perspective in alternating environment.

This test validates whether step_money represents the PCS's or ISO's perspective
when energy is bought/sold between the PCS and ISO.

Test Scenario:
- Fixed high price: $1.00 per unit
- PCS always tries to charge battery with +1 unit per timestep
- Run for 1 day (48 timesteps)

Expected Behavior (if correctly implemented):
- When PCS buys energy, ISO should earn money (positive reward)
- When PCS sells energy, ISO should pay money (negative reward)
"""

import unittest
import numpy as np
from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.gym_envs.alternating_env import AlternatingISOEnv
from energy_net.grid_entities.management.price_curve import PriceCurveStrategy
from stable_baselines3 import PPO


class FixedHighPriceStrategy(PriceCurveStrategy):
    """Price strategy that always returns a fixed high price for all timesteps."""
    
    def __init__(self, fixed_price=1.00, steps_per_day=48):
        self.fixed_price = fixed_price
        self.steps_per_day = steps_per_day
    
    def calculate_price(self, observation=None):
        """Return fixed price for all timesteps."""
        return np.full(self.steps_per_day, self.fixed_price, dtype=np.float32)


class TestISOPricingPerspective(unittest.TestCase):
    """Test suite to verify the perspective of money flow in ISO-PCS interactions."""
    
    def setUp(self):
        """Set up test environment with fixed high pricing."""
        self.fixed_price = 1.00
        self.price_strategy = FixedHighPriceStrategy(fixed_price=self.fixed_price)
        
        # Create PCS environment with fixed pricing
        self.pcs_env = PCSEnv(
            price_strategy=self.price_strategy,
            verbosity=0  # Silent mode for testing
        )
    
    def test_alternating_iso_perspective_when_pcs_buys(self):
        """
        Test that AlternatingISOEnv correctly converts PCS money to ISO perspective.
        
        When PCS buys energy (charges battery):
        - PCS step_money is negative (PCS pays)
        - ISO reward should be positive (ISO receives payment)
        """
        import os
        
        # Get paths to test data files
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        test_data = os.path.join(project_root, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
        predictions = os.path.join(project_root, 'tests/gym/data_for_tests/consumption_predictions.csv')
        
        # Create a simple random PCS policy
        pcs_model = PPO("MlpPolicy", self.pcs_env, verbose=0, n_steps=48, batch_size=48)
        
        # Create alternating ISO environment
        iso_env = AlternatingISOEnv(
            actual_csv=test_data,
            predicted_csv=predictions,
            pcs_env=self.pcs_env,
            pcs_model=pcs_model,
            iso_verbosity=0,
            pcs_verbosity=0
        )
        
        # Take one step in ISO environment
        obs, _ = iso_env.reset()
        
        # Create a dummy action (prices + dispatch)
        # The prices don't matter much since we set the strategy
        # But dispatch needs to be reasonable
        action = np.concatenate([
            np.ones(48) * 0.1,  # 48 prices
            np.ones(48) * 2.0   # 48 dispatch values
        ])
        
        # Override with our fixed price strategy
        iso_env.pcs_env.set_price_strategy(self.price_strategy)
        
        # Step the environment
        next_obs, iso_reward, done, truncated, info = iso_env.step(action)
        
        pcs_money = info.get('pcs_money', 0)
        iso_money = info.get('money_earned', 0)
        
        print(f"\n{'='*70}")
        print(f"Alternating ISO Environment Perspective Test")
        print(f"{'='*70}")
        print(f"PCS Money (step_money perspective): ${pcs_money:.2f}")
        print(f"ISO Money (corrected perspective):  ${iso_money:.2f}")
        print(f"ISO Reward:                         ${iso_reward:.2f}")
        print(f"{'='*70}")
        
        # The key assertion: ISO money should be the negative of PCS money
        self.assertAlmostEqual(
            iso_money,
            -pcs_money,
            places=2,
            msg=f"ISO money should be negative of PCS money. "
                f"PCS: ${pcs_money:.2f}, ISO: ${iso_money:.2f}"
        )
        
        # And ISO reward should match ISO money
        self.assertAlmostEqual(
            iso_reward,
            iso_money,
            places=2,
            msg=f"ISO reward should match ISO money. "
                f"Reward: ${iso_reward:.2f}, Money: ${iso_money:.2f}"
        )
        
        print(f"✓  Perspective conversion is correct!")
        print(f"   PCS money and ISO money have opposite signs as expected")
        print(f"{'='*70}\n")
    
    def test_pcs_environment_money_sign(self):
        """
        Test money flow when PCS consistently buys energy (charges battery).
        
        Scenario: PCS charges battery every timestep at $1.00/unit
        - Energy flow: ISO → PCS (PCS is buying)
        - Money flow: PCS → ISO (PCS is paying)
        
        From ISO's perspective:
        - ISO is SELLING energy to PCS
        - ISO should RECEIVE payment
        - ISO reward should be POSITIVE
        
        From PCS's perspective:
        - PCS is BUYING energy from ISO  
        - PCS is PAYING money
        - PCS money should be NEGATIVE
        """
        obs, _ = self.pcs_env.reset()
        
        total_pcs_money = 0.0
        total_energy_bought = 0.0
        steps_completed = 0
        
        # Run for one full day (48 steps)
        for step in range(48):
            # Action: +1.0 means charge battery (buy energy)
            action = np.array([1.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = self.pcs_env.step(action)
            
            # Accumulate PCS money (what would become ISO reward in alternating_env)
            step_money = info.get('step_money', 0.0)
            total_pcs_money += step_money
            
            # Track energy flow
            energy_change = info.get('storage_after_units', 0) - info.get('storage_before_units', 0)
            if energy_change > 0:  # Battery charged = energy bought
                total_energy_bought += energy_change
            
            steps_completed += 1
            
            if terminated or truncated:
                break
        
        print(f"\n{'='*70}")
        print(f"ISO Pricing Perspective Test Results")
        print(f"{'='*70}")
        print(f"Fixed Price: ${self.fixed_price:.2f} per unit")
        print(f"Steps Completed: {steps_completed}")
        print(f"Total Energy Bought by PCS: {total_energy_bought:.2f} units")
        print(f"Total PCS Money (step_money sum): ${total_pcs_money:.2f}")
        print(f"{'='*70}")
        
        if total_pcs_money < 0:
            print(f"✓  Perspective is correct")
            print(f"   PCS bought energy and step_money is NEGATIVE (${total_pcs_money:.2f})")
            print(f"   This correctly represents PCS's perspective:")
            print(f"   - PCS pays money → negative step_money")
        elif total_pcs_money > 0:
            print(f"⚠️  PERSPECTIVE ISSUE DETECTED!")
            print(f"   PCS bought energy but total money is POSITIVE (${total_pcs_money:.2f})")
            print(f"   This would mean PCS earns money when buying, which is incorrect")
        else:
            print(f"⚠️  No money was exchanged (total = $0.00)")
            print(f"   Test may be inconclusive - check if battery accepted charges")
        
        print(f"{'='*70}\n")
        
        # step_money should represent PCS's perspective
        # When PCS buys energy, PCS pays money, so step_money should be negative
        self.assertLess(
            total_pcs_money, 
            0,
            msg=f"PCS should have negative money when buying energy. "
                f"Got ${total_pcs_money:.2f}. step_money should represent "
                f"PCS's perspective (negative when paying for energy)."
        )
    
    def test_pcs_selling_energy_perspective(self):
        """
        Test money flow when PCS sells energy (discharges battery).
        
        Scenario: PCS discharges battery every timestep at $1.00/unit
        - Energy flow: PCS → ISO (PCS is selling)
        - Money flow: ISO → PCS (ISO is paying)
        
        From ISO's perspective:
        - ISO is BUYING energy from PCS
        - ISO should PAY money
        - ISO reward should be NEGATIVE
        
        From PCS's perspective:
        - PCS is SELLING energy to ISO
        - PCS is RECEIVING payment
        - PCS money should be POSITIVE
        """
        obs, _ = self.pcs_env.reset()
        
        # First, charge the battery so we can discharge it
        for _ in range(20):
            action = np.array([5.0], dtype=np.float32)  # Charge quickly
            obs, _, terminated, truncated, _ = self.pcs_env.step(action)
            if terminated or truncated:
                obs, _ = self.pcs_env.reset()
                break
        
        # Reset to start fresh day with charged battery
        obs, _ = self.pcs_env.reset()
        
        total_pcs_money = 0.0
        total_energy_sold = 0.0
        steps_completed = 0
        
        # Run discharge test
        for step in range(48):
            # Action: -1.0 means discharge battery (sell energy)
            action = np.array([-1.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = self.pcs_env.step(action)
            
            step_money = info.get('step_money', 0.0)
            total_pcs_money += step_money
            
            # Track energy flow
            energy_change = info.get('storage_after_units', 0) - info.get('storage_before_units', 0)
            if energy_change < 0:  # Battery discharged = energy sold
                total_energy_sold += abs(energy_change)
            
            steps_completed += 1
            
            if terminated or truncated:
                break
        
        print(f"\n{'='*70}")
        print(f"ISO Pricing Perspective Test Results (PCS Selling)")
        print(f"{'='*70}")
        print(f"Fixed Price: ${self.fixed_price:.2f} per unit")
        print(f"Steps Completed: {steps_completed}")
        print(f"Total Energy Sold by PCS: {total_energy_sold:.2f} units")
        print(f"Total PCS Money (step_money sum): ${total_pcs_money:.2f}")
        print(f"{'='*70}")
        
        if total_energy_sold > 0:  # Only assess if energy was actually sold
            if total_pcs_money > 0:
                print(f"⚠️  PERSPECTIVE ISSUE DETECTED!")
                print(f"   PCS sold energy but step_money is POSITIVE (${total_pcs_money:.2f})")
                print(f"   This means step_money represents PCS's perspective:")
                print(f"   - PCS receives payment → positive step_money")
                print(f"   But in alternating_env, this becomes ISO's reward!")
                print(f"   - ISO pays money → should be NEGATIVE reward")
                print(f"   - Currently: ISO gets POSITIVE reward when it should pay!")
                print(f"\n   CONCLUSION: Reward perspective is inverted (PCS vs ISO)")
            elif total_pcs_money < 0:
                print(f"✓  Perspective appears correct")
                print(f"   PCS sold energy and total money is NEGATIVE (${total_pcs_money:.2f})")
                print(f"   This represents ISO's perspective:")
                print(f"   - ISO buys energy → negative money (payment)")
        
        print(f"{'='*70}\n")
        
        # When PCS sells, ISO should pay (negative reward for ISO)
        if total_energy_sold > 0:
            self.assertLess(
                total_pcs_money,
                0,
                msg=f"ISO should have negative reward when PCS sells energy. "
                    f"Got ${total_pcs_money:.2f}. This indicates step_money is from "
                    f"PCS's perspective (positive when selling), not ISO's perspective "
                    f"(negative when paying)."
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)