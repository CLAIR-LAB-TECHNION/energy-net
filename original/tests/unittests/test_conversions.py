# tests/test_components.py

import unittest

import numpy as np

from energy_net import Bounds
from original.dynamics import BatteryDynamicsDet
from original.dynamics import ProductionDynamicsDet
from original.dynamics.consumption_dynamics.consumption_dynamics_det import ConsumptionDynamicsDet
from original.envs import bounds_to_gym, gym_to_bounds


class TestConversion(unittest.TestCase):
    def setUp(self):
      pass

    def test_gym_energy_conversions(self):
        # single values
        energy_net_bounds = Bounds(low=0.0,high=5.0)
        gym_bounds = bounds_to_gym(energy_net_bounds)
        energy_net_conv_bounds = gym_to_bounds(gym_bounds)

        # verify values have not changed
        self.assertEqual(energy_net_bounds.low.all(), energy_net_conv_bounds.low.all())
        self.assertEqual(energy_net_bounds.high.all(), energy_net_conv_bounds.high.all())

        energy_net_bounds = Bounds(low=np.array([0.0, 0.0, 0.0], dtype=np.float32),high=np.array([5, 6, 5], dtype=np.float32))
        gym_bounds = bounds_to_gym(energy_net_bounds)
        energy_net_conv_bounds = gym_to_bounds(gym_bounds)

        # verify values have not changed
        self.assertEqual(energy_net_bounds.low.all(), energy_net_conv_bounds.low.all())
        self.assertEqual(energy_net_bounds.high.all(), energy_net_conv_bounds.high.all())

        energy_net_bounds = Bounds(low=np.array([0.0,0.0,-np.inf],dtype=np.float32),high=np.array([0.0,0.0,-np.inf],dtype=np.float32))
        gym_bounds = bounds_to_gym(energy_net_bounds)
        energy_net_conv_bounds = gym_to_bounds(gym_bounds)

        # verify values have not changed
        self.assertEqual(energy_net_bounds.low.all(), energy_net_conv_bounds.low.all())
        self.assertEqual(energy_net_bounds.high.all(), energy_net_conv_bounds.high.all())


if __name__ == '__main__':
    unittest.main()
