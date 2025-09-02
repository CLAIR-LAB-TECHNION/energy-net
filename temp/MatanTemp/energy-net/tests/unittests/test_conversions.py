# tests/test_components.py

import unittest
from typing import Dict, Any
import math

import numpy as np

from energy_net.components.storage_devices.battery import Battery
from energy_net.components.production_devices.production_unit import ProductionUnit
from energy_net.components.consumption_devices.consumption_unit import ConsumptionUnit
from energy_net.defs import Bounds
from energy_net.dynamics.storage_dynamics.battery_dynamics_det import BatteryDynamicsDet
from energy_net.dynamics.production_dynamics.production_dynmaics_det import ProductionDynamicsDet
from energy_net.dynamics.consumption_dynamics.consumption_dynamics_det import ConsumptionDynamicsDet
from energy_net.components.pcsunit import PCSUnit
from energy_net.envs.env_utils import bounds_to_gym, gym_to_bounds


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
