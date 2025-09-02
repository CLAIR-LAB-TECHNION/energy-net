# tests/test_components.py

import unittest
from typing import Dict, Any
import math

from energy_net.components.electrolyzer import Electrolyzer
from energy_net.components.production_devices.production_unit import ProductionUnit
from energy_net.components.consumption_devices.consumption_unit import ConsumptionUnit
from energy_net.dynamics.production_dynamics.production_dynmaics_det import ProductionDynamicsDet
from energy_net.dynamics.consumption_dynamics.consumption_dynamics_det import ConsumptionDynamicsDet
from energy_net.components.electrolyzer import Electrolyzer



class TestElectrolyzer(unittest.TestCase):
    def setUp(self):
        # Define configuration for Electrolyzer
        self.electrolyzer_config: Dict[str, Any] = {
            'production_unit': {
                'dynamic_type': 'model_based',
                'model_type': 'deterministic_production',
                'model_parameters': {
                    'production_capacity': 50.0,
                    'peak_production': 50.0,
                    'peak_time': 0.5,
                    'width': 0.1
                }
            },
            'consumption_unit': {
                'dynamic_type': 'model_based',
                'model_type': 'deterministic_consumption',
                'model_parameters': {
                    'consumption_capacity': 45.0,
                    'peak_consumption1': 20.0,
                    'peak_time1': 0.4,
                    'width1': 0.05,
                    'peak_consumption2': 25.0,
                    'peak_time2': 0.7,
                    'width2': 0.05
                }
            }
        }

        # Initialize Electrolyzer component
        self.electrolyzer = Electrolyzer(config=self.electrolyzer_config)

    def test_initialization_with_valid_parameters(self):
        # Test that Electrolyzer initializes correctly with valid parameters
        self.assertEqual(self.electrolyzer.production_unit.get_state(), 0.0)
        self.assertEqual(self.electrolyzer.consumption_unit.get_state(), 0.0)

    def test_initialization_missing_parameters(self):
        # Test that Electrolyzer raises AssertionError if configuration parameters are missing
        incomplete_config: Dict[str, Any] = {            
            'production_unit': {
                'dynamic_type': 'model_based',
                'model_type': 'deterministic_production',
                'model_parameters': {
                    'production_capacity': 50.0,
                    'peak_production': 50.0,
                    'peak_time': 0.5,
                    'width': 0.1
                }
            }
            # 'consumption_unit' is missing
        }
        with self.assertRaises(AssertionError):
            Electrolyzer(config=incomplete_config)

    def test_reset(self):
        # Test that reset restores all components to their initial states
        self.electrolyzer.reset()
        self.assertEqual(self.electrolyzer.production_unit.get_state(), 0.0)
        self.assertEqual(self.electrolyzer.consumption_unit.get_state(), 0.0)


if __name__ == '__main__':
    unittest.main()
