# tests/test_components.py

import unittest
from typing import Dict, Any
import math



class TestControllers(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialization_with_valid_parameters(self):
        # Test that Battery initializes correctly with valid parameters
        self.assertEqual(self.battery.energy_level, 30.0)
        self.assertEqual(self.battery.energy_min, 0.0)
        self.assertEqual(self.battery.energy_max, 60.0)
        self.assertEqual(self.battery.charge_rate_max, 4.0)
        self.assertEqual(self.battery.discharge_rate_max, 2.0)
        self.assertEqual(self.battery.charge_efficiency, 0.4)
        self.assertEqual(self.battery.discharge_efficiency, 0.6)



if __name__ == '__main__':
    unittest.main()
