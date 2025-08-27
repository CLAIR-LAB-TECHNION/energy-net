# tests/test_components.py

import unittest
from typing import Dict, Any
import math

from original.components.storage_devices.battery import Battery
from original.components.production_devices import ProductionUnit
from original.components.consumption_devices.consumption_unit import ConsumptionUnit
from original.dynamics import DeterministicBattery
from original.dynamics import DeterministicProduction
from original.dynamics.consumption_dynamics.consumption_dynamics_det import DeterministicConsumption
from original.components.pcsunit import PCSUnit


class TestBattery(unittest.TestCase):
    def setUp(self):
        # Define valid model parameters for BatteryDynamicsDet
        battery_model_params: Dict[str, Any] = {
            'charge_efficiency': 0.4,
            'discharge_efficiency': 0.6,
            'lifetime_constant': 100.0
        }
        self.battery_dynamics = DeterministicBattery(model_parameters=battery_model_params)

        # Define configuration for Battery
        self.battery_config: Dict[str, Any] = {
            'min': 0.0,
            'max': 60.0,
            'charge_rate_max': 4.0,
            'discharge_rate_max': 2.0,
            'charge_efficiency': 0.4,
            'discharge_efficiency': 0.6,
            'init': 30.0
        }

        # Initialize Battery component
        self.battery = Battery(dynamics=self.battery_dynamics, config=self.battery_config)

    def test_initialization_with_valid_parameters(self):
        # Test that Battery initializes correctly with valid parameters
        self.assertEqual(self.battery.energy_level, 30.0)
        self.assertEqual(self.battery.energy_min, 0.0)
        self.assertEqual(self.battery.energy_max, 60.0)
        self.assertEqual(self.battery.charge_rate_max, 4.0)
        self.assertEqual(self.battery.discharge_rate_max, 2.0)
        self.assertEqual(self.battery.charge_efficiency, 0.4)
        self.assertEqual(self.battery.discharge_efficiency, 0.6)

    def test_initialization_missing_parameters(self):
        # Test that Battery raises AssertionError if configuration parameters are missing
        incomplete_config: Dict[str, Any] = {
            'min': 0.0,
            'max': 60.0,
            'charge_rate_max': 4.0,
            # 'discharge_rate_max' is missing
            'charge_efficiency': 0.4,
            'discharge_efficiency': 0.6,
            'init': 30.0
        }
        with self.assertRaises(AssertionError):
            Battery(dynamics=self.battery_dynamics, config=incomplete_config)

    def test_perform_action_charge_within_limits(self):
        # Test charging within charge_rate_max and energy_max
        self.battery.update(time=0.5, action=2.0)  # Charge 2.0 MW
        expected_energy = 30.0 + 2.0 * 0.4  # 30 + 0.8 = 30.8
        self.assertAlmostEqual(self.battery.get_state(), expected_energy, places=5)

    def test_perform_action_discharge_within_limits(self):
        # Test discharging within discharge_rate_max and energy_min
        self.battery.update(time=0.5, action=-1.5)  # Discharge 1.5 MW
        expected_energy = 30.0 - 1.5 * 0.6  # 30 - 0.9 = 29.1
        self.assertAlmostEqual(self.battery.get_state(), expected_energy, places=5)

    def test_charge_exceeds_max_energy(self):
        # Test charging that would exceed max_energy
        self.battery_config['init'] = 59.0
        self.battery.energy_level = 59.0
        self.battery.update(time=0.5, action=4.0)  # Attempt to charge 4.0 MW
        # Charge power limited to 4.0 MW, energy_change = 4.0 * 0.4 = 1.6
        # New energy = min(59.0 + 1.6, 60.0) = 60.0
        expected_energy = 60.0
        self.assertAlmostEqual(self.battery.get_state(), expected_energy, places=5)

    def test_discharge_exceeds_min_energy(self):
        # Test discharging that would exceed min_energy
        self.battery_config['init'] = 0.5
        self.battery.energy_level = 0.5
        self.battery.update(time=0.5, action=-2.0)  # Attempt to discharge 2.0 MW
        # Discharge power limited to 2.0 MW, energy_change = 2.0 * 0.6 = 1.2
        # New energy = max(0.5 - 1.2, 0.0) = 0.0
        expected_energy = 0.0
        self.assertAlmostEqual(self.battery.get_state(), expected_energy, places=5)

    def test_no_action(self):
        # Test that energy level remains unchanged when no action is taken
        initial_energy = self.battery.get_state()
        self.battery.update(time=0.5, action=0.0)  # No action
        self.assertAlmostEqual(self.battery.get_state(), initial_energy, places=5)

    def test_reset(self):
        # Test that reset restores the energy level to initial value
        self.battery.update(time=0.5, action=2.0)  # Charge 2.0 MW
        self.battery.reset()
        self.assertEqual(self.battery.get_state(), 30.0)


class TestProductionUnit(unittest.TestCase):
    def setUp(self):
        # Define model parameters for ProductionDynamicsDet
        production_model_params: Dict[str, Any] = {
            'peak_production': 50.0,
            'peak_time': 0.5,  # Midday
            'width': 0.1
        }
        self.production_dynamics = DeterministicProduction(model_parameters=production_model_params)

        # Define configuration for ProductionUnit
        self.production_config: Dict[str, Any] = {
            'production_capacity': 50.0
        }

        # Initialize ProductionUnit component
        self.production_unit = ProductionUnit(dynamics=self.production_dynamics, config=self.production_config)

    def test_initialization_with_valid_parameters(self):
        # Test that ProductionUnit initializes correctly with valid parameters
        self.assertEqual(self.production_unit.production_capacity, 50.0)
        self.assertEqual(self.production_unit.get_state(), 0.0)

    def test_initialization_missing_parameters(self):
        # Test that ProductionUnit raises AssertionError if configuration parameters are missing
        incomplete_config: Dict[str, Any] = {
            # 'production_capacity' is missing
        }
        with self.assertRaises(AssertionError):
            ProductionUnit(dynamics=self.production_dynamics, config=incomplete_config)

    def test_update_production_at_peak_time(self):
        # Test production at peak time
        self.production_unit.update(time=0.5, action=0.0)  # At peak
        expected_production = min(
            50.0 * math.exp(-((0.5 - 0.5) ** 2) / (2 * (0.1 ** 2))),
            50.0
        )  # 50 * 1 = 50
        self.assertAlmostEqual(self.production_unit.get_state(), expected_production, places=5)

    def test_update_production_before_peak_time(self):
        # Test production before peak time
        self.production_unit.update(time=0.3, action=0.0)  # Before peak
        expected_production = min(
            50.0 * math.exp(-((0.3 - 0.5) ** 2) / (2 * (0.1 ** 2))),
            50.0
        )
        self.assertAlmostEqual(self.production_unit.get_state(), expected_production, places=5)

    def test_update_production_after_peak_time(self):
        # Test production after peak time
        self.production_unit.update(time=0.7, action=0.0)  # After peak
        expected_production = min(
            50.0 * math.exp(-((0.7 - 0.5) ** 2) / (2 * (0.1 ** 2))),
            50.0
        )
        self.assertAlmostEqual(self.production_unit.get_state(), expected_production, places=5)

    def test_reset(self):
        # Test that reset restores the production level to initial value
        self.production_unit.update(time=0.5, action=0.0)  # At peak
        self.production_unit.reset()
        self.assertEqual(self.production_unit.get_state(), 0.0)


class TestConsumptionUnit(unittest.TestCase):
    def setUp(self):
        # Define model parameters for ConsumptionDynamicsDet
        consumption_model_params: Dict[str, Any] = {
            'peak_consumption1': 20.0,
            'peak_time1': 0.4,  # Midday
            'width1': 0.05,
            'peak_consumption2': 25.0,
            'peak_time2': 0.7,  # Evening
            'width2': 0.05
        }
        self.consumption_dynamics = DeterministicConsumption(model_parameters=consumption_model_params)

        # Define configuration for ConsumptionUnit
        self.consumption_config: Dict[str, Any] = {
            'consumption_capacity': 45.0
        }

        # Initialize ConsumptionUnit component
        self.consumption_unit = ConsumptionUnit(dynamics=self.consumption_dynamics, config=self.consumption_config)

    def test_initialization_with_valid_parameters(self):
        # Test that ConsumptionUnit initializes correctly with valid parameters
        self.assertEqual(self.consumption_unit.consumption_capacity, 45.0)
        self.assertEqual(self.consumption_unit.get_state(), 0.0)

    def test_initialization_missing_parameters(self):
        # Test that ConsumptionUnit raises AssertionError if configuration parameters are missing
        incomplete_config: Dict[str, Any] = {
            # 'consumption_capacity' is missing
        }
        with self.assertRaises(AssertionError):
            ConsumptionUnit(dynamics=self.consumption_dynamics, config=incomplete_config)

    def test_update_consumption_at_peak1(self):
        # Test consumption at first peak time
        self.consumption_unit.update(time=0.4, action=0.0)  # At peak1
        expected_consumption = min(
            20.0 * math.exp(-((0.4 - 0.4) ** 2) / (2 * (0.05 ** 2))) +
            25.0 * math.exp(-((0.4 - 0.7) ** 2) / (2 * (0.05 ** 2))),
            45.0
        )  
        self.assertAlmostEqual(self.consumption_unit.get_state(), expected_consumption, places=5)

    def test_update_consumption_at_peak2(self):
        # Test consumption at second peak time
        self.consumption_unit.update(time=0.7, action=0.0)  # At peak2
        expected_consumption = min(
            20.0 * math.exp(-((0.7 - 0.4) ** 2) / (2 * (0.05 ** 2))) +
            25.0 * math.exp(-((0.7 - 0.7) ** 2) / (2 * (0.05 ** 2))),
            45.0
        )  
        self.assertAlmostEqual(self.consumption_unit.get_state(), expected_consumption, places=5)

    def test_update_consumption_between_peaks(self):
        # Test consumption between peak1 and peak2
        self.consumption_unit.update(time=0.6, action=0.0)  # Between peaks
        expected_consumption = min(
            20.0 * math.exp(-((0.6 - 0.4) ** 2) / (2 * (0.05 ** 2))) +
            25.0 * math.exp(-((0.6 - 0.7) ** 2) / (2 * (0.05 ** 2))),
            45.0
        )
        self.assertAlmostEqual(self.consumption_unit.get_state(), expected_consumption, places=5)

    def test_update_consumption_exceeds_capacity(self):
        # Test consumption that would exceed consumption_capacity
        self.consumption_config['consumption_capacity'] = 40.0
        self.consumption_unit.consumption_capacity = 40.0
        self.consumption_unit.update(time=0.7, action=0.0)  # At peak2
        expected_consumption = min(
            20.0 * math.exp(-((0.7 - 0.4) ** 2) / (2 * (0.05 ** 2))) +
            25.0 * math.exp(-((0.7 - 0.7) ** 2) / (2 * (0.05 ** 2))),
            40.0
        )  # ≈ 16.7054 + 25 = 41.7054 → min(41.7054, 40.0) = 40.0
        self.assertAlmostEqual(self.consumption_unit.get_state(), expected_consumption, places=5)

    def test_no_action(self):
        # Test that consumption level updates based on dynamics even when no action is taken
        initial_consumption = self.consumption_unit.get_state()
        self.consumption_unit.update(time=0.5, action=0.0)  # No action
        # According to ConsumptionDynamicsDet, action=0.0 still updates consumption based on time
        expected_consumption = min(
            20.0 * math.exp(-((0.5 - 0.4) ** 2) / (2 * (0.05 ** 2))) +
            25.0 * math.exp(-((0.5 - 0.7) ** 2) / (2 * (0.05 ** 2))),
            45.0
        )
        self.assertAlmostEqual(self.consumption_unit.get_state(), expected_consumption, places=5)

    def test_reset(self):
        # Test that reset restores the consumption level to initial value
        self.consumption_unit.update(time=0.4, action=0.0)  # At peak1
        self.consumption_unit.reset()
        self.assertEqual(self.consumption_unit.get_state(), 0.0)


class TestPCSUnit(unittest.TestCase):
    def setUp(self):
        # Define configuration for PCSUnit
        self.pcs_config: Dict[str, Any] = {
            'storage': {
                'dynamic_type': 'model_based',
                'model_type': 'deterministic_battery',
                'model_parameters': {
                    'min': 0.0,
                    'max': 60.0,
                    'charge_rate_max': 4.0,
                    'discharge_rate_max': 2.0,
                    'charge_efficiency': 0.4,
                    'discharge_efficiency': 0.6,
                    'init': 30.0,
                    'lifetime_constant': 100.0
                }
            },
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

        # Initialize PCSUnit component
        self.pcs_unit = PCSUnit(config=self.pcs_config)

    def test_initialization_with_valid_parameters(self):
        # Test that PCSUnit initializes correctly with valid parameters
        self.assertEqual(self.pcs_unit.battery.get_state(), 30.0)
        self.assertEqual(self.pcs_unit.production_unit.get_state(), 0.0)
        self.assertEqual(self.pcs_unit.consumption_unit.get_state(), 0.0)

    def test_initialization_missing_parameters(self):
        # Test that PCSUnit raises AssertionError if configuration parameters are missing
        incomplete_config: Dict[str, Any] = {
            'storage': {
                'dynamic_type': 'model_based',
                'model_type': 'deterministic_battery',
                'model_parameters': {
                    'min': 0.0,
                    'max': 60.0,
                    'charge_rate_max': 4.0,
                    'discharge_rate_max': 2.0,
                    'charge_efficiency': 0.4,
                    'discharge_efficiency': 0.6,
                    'init': 30.0,
                    'lifetime_constant': 100.0
                }
            },
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
            PCSUnit(config=incomplete_config)

    def test_update_with_charge_action(self):
        # Test updating PCSUnit with a charge action
        self.pcs_unit.update(time=0.5, battery_action=2.0)  # Charge 2.0 MW
        expected_energy = 30.0 + 2.0 * 0.4  # 30 + 0.8 = 30.8
        self.assertAlmostEqual(self.pcs_unit.battery.get_state(), expected_energy, places=5)

    def test_update_with_discharge_action(self):
        # Test updating PCSUnit with a discharge action
        self.pcs_unit.update(time=0.5, battery_action=-1.0)  # Discharge 1.0 MW
        expected_energy = 30.0 - 1.0 * 0.6  # 30 - 0.6 = 29.4
        self.assertAlmostEqual(self.pcs_unit.battery.get_state(), expected_energy, places=5)

    def test_reset(self):
        # Test that reset restores all components to their initial states
        self.pcs_unit.update(time=0.5, battery_action=2.0)  # Charge 2.0 MW
        self.pcs_unit.update(time=0.5, battery_action=-1.0)  # Discharge 1.0 MW
        self.pcs_unit.reset()
        self.assertEqual(self.pcs_unit.battery.get_state(), 30.0)
        self.assertEqual(self.pcs_unit.production_unit.get_state(), 0.0)
        self.assertEqual(self.pcs_unit.consumption_unit.get_state(), 0.0)


# class TestPCSUnitIntegration(unittest.TestCase):
#     def setUp(self):
#         # Define model parameters for DeterministicBattery
#         battery_model_params: Dict[str, Any] = {
#             'charge_efficiency': 0.4,
#             'discharge_efficiency': 0.6,
#             'lifetime_constant': 100.0
#         }
#         self.battery_dynamics = DeterministicBattery(model_parameters=battery_model_params)

#         # Define model parameters for DeterministicProduction
#         production_model_params: Dict[str, Any] = {
#             'peak_production': 50.0,
#             'peak_time': 0.5,
#             'width': 0.1
#         }
#         self.production_dynamics = DeterministicProduction(model_parameters=production_model_params)

#         # Define model parameters for DeterministicConsumption
#         consumption_model_params: Dict[str, Any] = {
#             'peak_consumption1': 20.0,
#             'peak_time1': 0.4,
#             'width1': 0.05,
#             'peak_consumption2': 25.0,
#             'peak_time2': 0.7,
#             'width2': 0.05
#         }
#         self.consumption_dynamics = DeterministicConsumption(model_parameters=consumption_model_params)

#         # Define configuration for PCSUnit
#         self.pcs_config: Dict[str, Any] = {
#             'storage': {
#                 'min': 0.0,
#                 'max': 60.0,
#                 'charge_rate_max': 4.0,
#                 'discharge_rate_max': 2.0,
#                 'charge_efficiency': 0.4,
#                 'discharge_efficiency': 0.6,
#                 'init': 30.0
#             },
#             'production_unit': {
#                 'production_capacity': 50.0
#             },
#             'consumption_unit': {
#                 'consumption_capacity': 45.0
#             }
#         }

#         # Initialize PCSUnit component
#         self.pcs_unit = PCSUnit(
#             battery_dynamics=self.battery_dynamics,
#             production_dynamics=self.production_dynamics,
#             consumption_dynamics=self.consumption_dynamics,
#             config=self.pcs_config
#         )

#     def test_integration_charge_and_discharge(self):
#         # Charge the storage
#         self.pcs_unit.update(time=0.5, battery_action=3.0)  # Charge 3.0 MW
#         expected_energy_after_charge = 30.0 + 3.0 * 0.4  # 30 + 1.2 = 31.2
#         self.assertAlmostEqual(self.pcs_unit.storage.get_state(), expected_energy_after_charge, places=5)

#         # Discharge the storage
#         self.pcs_unit.update(time=0.5, battery_action=-1.5)  # Discharge 1.5 MW
#         expected_energy_after_discharge = 31.2 - 1.5 * 0.6  # 31.2 - 0.9 = 30.3
#         self.assertAlmostEqual(self.pcs_unit.storage.get_state(), expected_energy_after_discharge, places=5)

#     def test_integration_reset(self):
#         # Perform some actions
#         self.pcs_unit.update(time=0.5, battery_action=2.0)  # Charge 2.0 MW
#         self.pcs_unit.update(time=0.6, battery_action=-1.0)  # Discharge 1.0 MW

#         # Reset PCSUnit
#         self.pcs_unit.reset()
#         self.assertEqual(self.pcs_unit.storage.get_state(), 30.0)
#         self.assertEqual(self.pcs_unit.production_unit.get_state(), 0.0)
#         self.assertEqual(self.pcs_unit.consumption_unit.get_state(), 0.0)


if __name__ == '__main__':
    unittest.main()
