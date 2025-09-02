# tests/test_environment_registration.py

import unittest
import gymnasium as gym
import energy_net
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

class TestEnvironmentRegistration(unittest.TestCase):
    def test_environment_registration(self):
        """
        Test that EnergyNetEnv is correctly registered and can be instantiated.
        """
        try:
            env = gym.make('EnergyNetEnv-v0', 
                          pricing_policy=PricingPolicy.ONLINE,
                          cost_type=CostType.CONSTANT,
                          demand_pattern=DemandPattern.SINUSOIDAL)
            self.assertIsNotNone(env, "Failed to instantiate EnergyNetEnv-v0.")
            self.assertEqual(env.spec.id, 'EnergyNetEnv-v0', "Environment ID does not match.")
            env.close()
        except Exception as e:
            self.fail(f"Failed to make EnergyNetEnv-v0: {e}")

if __name__ == '__main__':
    unittest.main()
