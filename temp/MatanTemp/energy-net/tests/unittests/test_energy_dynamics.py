# tests/test_energy_dynamics.py

import unittest
from energy_net.dynamics.production_dynamics.production_dynmaics_det import ProductionDynamicsDet
from energy_net.dynamics.consumption_dynamics.consumption_dynamics_det import ConsumptionDynamicsDet

from typing import Any, Dict
import math


class TestDeterministicProduction(unittest.TestCase):
    def setUp(self):
        model_params: Dict[str, Any] = {
            'peak_production': 50.0,
            'peak_time': 0.5,  # Midday
            'width': 0.1
        }
        self.dynamics: ProductionDynamicsDet = ProductionDynamicsDet(model_parameters=model_params)

    def test_get_value_peak_midday(self):
        time: float = 0.5  # Midday
        expected_production: float = 50.0 * math.exp(-((0.5 - 0.5) ** 2) / (2 * (0.1 ** 2)))  # 50 * 1 = 50
        production: float = self.dynamics.get_value(time=time)
        self.assertAlmostEqual(production, expected_production, places=5)

    def test_get_value_morning(self):
        time: float = 0.3  # Morning
        expected_production: float = 50.0 * math.exp(-((0.3 - 0.5) ** 2) / (2 * (0.1 ** 2)))
        production: float = self.dynamics.get_value(time=time)
        self.assertAlmostEqual(production, expected_production, places=5)

    def test_get_value_evening(self):
        time: float = 0.7  # Evening
        expected_production: float = 50.0 * math.exp(-((0.7 - 0.5) ** 2) / (2 * (0.1 ** 2)))
        production: float = self.dynamics.get_value(time=time)
        self.assertAlmostEqual(production, expected_production, places=5)

    def test_missing_parameters(self):
        with self.assertRaises(AssertionError):
            incomplete_params: Dict[str, Any] = {
                'peak_production': 50.0,
                # Missing 'peak_time' and 'width'
            }
            ProductionDynamicsDet(model_parameters=incomplete_params)


class TestDeterministicConsumption(unittest.TestCase):
    def setUp(self):
        model_params: Dict[str, Any] = {
            'peak_consumption1': 20.0,
            'peak_time1': 0.4,  # Midday
            'width1': 0.05,
            'peak_consumption2': 25.0,
            'peak_time2': 0.7,  # Evening
            'width2': 0.05
        }
        self.dynamics: ConsumptionDynamicsDet = ConsumptionDynamicsDet(model_parameters=model_params)

    def test_get_value_midday_peak1(self):
        time: float = 0.4  # Midday peak 1
        expected_consumption1: float = 20.0 * math.exp(-((0.4 - 0.4) ** 2) / (2 * (0.05 ** 2)))  # 20 * 1 = 20
        expected_consumption2: float = 25.0 * math.exp(-((0.4 - 0.7) ** 2) / (2 * (0.05 ** 2)))  # Near zero
        expected_total: float = expected_consumption1 + expected_consumption2
        consumption: float = self.dynamics.get_value(time=time)
        self.assertAlmostEqual(consumption, expected_total, places=5)

    def test_get_value_evening_peak2(self):
        time: float = 0.7  # Evening peak 2
        expected_consumption1: float = 20.0 * math.exp(-((0.7 - 0.4) ** 2) / (2 * (0.05 ** 2)))  # Near zero
        expected_consumption2: float = 25.0 * math.exp(-((0.7 - 0.7) ** 2) / (2 * (0.05 ** 2)))  # 25 * 1 = 25
        expected_total: float = expected_consumption1 + expected_consumption2
        consumption: float = self.dynamics.get_value(time=time)
        self.assertAlmostEqual(consumption, expected_total, places=5)

    def test_get_value_between_peaks(self):
        time: float = 0.6  # Between midday and evening
        expected_consumption1: float = 20.0 * math.exp(-((0.6 - 0.4) ** 2) / (2 * (0.05 ** 2)))
        expected_consumption2: float = 25.0 * math.exp(-((0.6 - 0.7) ** 2) / (2 * (0.05 ** 2)))
        expected_total: float = expected_consumption1 + expected_consumption2
        consumption: float = self.dynamics.get_value(time=time)
        self.assertAlmostEqual(consumption, expected_total, places=5)

    def test_missing_parameters(self):
        with self.assertRaises(AssertionError):
            incomplete_params: Dict[str, Any] = {
                'peak_consumption1': 20.0,
                # Missing 'peak_time1', 'width1', 'peak_consumption2', 'peak_time2', 'width2'
            }
            ConsumptionDynamicsDet(model_parameters=incomplete_params)


if __name__ == '__main__':
    unittest.main()