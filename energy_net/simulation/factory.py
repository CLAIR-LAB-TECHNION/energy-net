"""
Wiring: build a Prosumer and a ProsumerSimulator from files or components.

This is the only place that chooses concrete dynamics and forecasters.
Every argument can be replaced by an injected component.
"""

import math
import os
from typing import Any, Dict, Optional

import pandas as pd

from energy_net.dynamics import ConstantDynamics, CSVDynamics
from energy_net.forecasting import NoisyGroundTruthForecaster, TableForecaster
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.grid_entities.management.price_curve import (
    PriceCurveStrategy,
    SineWavePriceStrategy,
)
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.prosumer.prosumer import Prosumer
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery
from energy_net.simulation.calendar import CALENDAR_FEATURE_COLUMNS
from energy_net.simulation.prosumer_simulator import ProsumerSimulator
from energy_net.simulation.tariffs import ForecastDailyObservation, TariffSchedule

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEST_DATA = os.path.join(_PROJECT_ROOT, "tests", "gym", "data_for_tests")

DEFAULT_LOAD_DATA_FILE = os.path.join(_TEST_DATA, "synthetic_household_consumption_test.csv")
DEFAULT_LOAD_FORECAST_FILE = os.path.join(_TEST_DATA, "consumption_predictions.csv")
DEFAULT_BATTERY_CONFIG = {"min": 0.0, "max": 100.0, "init": 0.0}


def steps_per_episode(episode_length_days: float, dt_days: float) -> int:
    """Number of intervals; rejects lengths that are not whole intervals."""
    steps = episode_length_days / dt_days
    if steps < 1 or not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"episode_length_days={episode_length_days} is not a whole "
            f"number of dt={dt_days}-day intervals"
        )
    return int(round(steps))


def make_prosumer(load_data_file: Optional[str] = None,
                  load_forecast_file: Optional[str] = None,
                  generation_data_file: Optional[str] = None,
                  generation_forecast_file: Optional[str] = None,
                  *,
                  load_dynamics=None,
                  generation_dynamics=None,
                  load_forecaster=None,
                  generation_forecaster=None,
                  battery: Optional[Battery] = None,
                  battery_config: Optional[Dict[str, Any]] = None,
                  load_value_column: str = "Consumption",
                  load_forecast_column: str = "predicted_consumption",
                  generation_value_column: str = "Production",
                  generation_forecast_column: str = "predicted_production") -> Prosumer:
    """
    Build a Prosumer. Defaults reproduce the legacy PCS environment: the
    test-data household load with its precomputed forecasts, a 0-100 battery
    starting empty, and a zero-output generator forecast perfectly.
    """
    if load_dynamics is None:
        load_dynamics = CSVDynamics.from_csv(
            load_data_file or DEFAULT_LOAD_DATA_FILE, load_value_column,
            quantity="consumption",
        )
    if load_forecaster is None:
        load_forecaster = TableForecaster.from_csv(
            load_forecast_file or DEFAULT_LOAD_FORECAST_FILE, load_forecast_column,
        )

    if generation_dynamics is None:
        generation_dynamics = (
            CSVDynamics.from_csv(generation_data_file, generation_value_column,
                                 quantity="generation")
            if generation_data_file else ConstantDynamics(0.0, quantity="generation")
        )
    if generation_forecaster is None:
        generation_forecaster = (
            TableForecaster.from_csv(generation_forecast_file, generation_forecast_column)
            if generation_forecast_file else NoisyGroundTruthForecaster(generation_dynamics)
        )

    if battery is None:
        battery = Battery(
            dynamics=DeterministicBattery(model_parameters={}),
            config=dict(DEFAULT_BATTERY_CONFIG if battery_config is None else battery_config),
        )

    return Prosumer(
        storage=battery,
        load=ConsumptionUnit(load_dynamics, {"consumption_capacity": 6.0},
                             forecaster=load_forecaster),
        generator=ProductionUnit(generation_dynamics, {"production_capacity": 0.0},
                                 forecaster=generation_forecaster),
    )


def make_prosumer_simulator(load_data_file: Optional[str] = None,
                            load_forecast_file: Optional[str] = None,
                            generation_data_file: Optional[str] = None,
                            generation_forecast_file: Optional[str] = None,
                            *,
                            prosumer: Optional[Prosumer] = None,
                            dt: float = 0.5 / 24,
                            episode_length_days: float = 1,
                            forecast_horizon: Optional[int] = None,
                            price_strategy: Optional[PriceCurveStrategy] = None,
                            daily_observation=None,
                            verbosity: int = 0,
                            **prosumer_kwargs) -> ProsumerSimulator:
    """
    Build a ProsumerSimulator.

    ``dt`` and ``episode_length_days`` are in days, as in the legacy env.
    ``prosumer_kwargs`` are forwarded to :func:`make_prosumer` when no
    ``prosumer`` is given.
    """
    if prosumer is None:
        prosumer = make_prosumer(
            load_data_file, load_forecast_file,
            generation_data_file, generation_forecast_file,
            **prosumer_kwargs,
        )
    elif prosumer_kwargs or any((load_data_file, load_forecast_file,
                                 generation_data_file, generation_forecast_file)):
        raise ValueError("Pass either a prosumer or the arguments to build one, not both")

    if daily_observation is None:
        daily_observation = ForecastDailyObservation(prosumer.load)
    tariffs = TariffSchedule(
        price_strategy if price_strategy is not None else SineWavePriceStrategy(),
        daily_observation,
    )
    simulator = ProsumerSimulator(
        prosumer,
        tariffs,
        episode_steps=steps_per_episode(episode_length_days, dt),
        forecast_horizon=forecast_horizon,
        dt=pd.Timedelta(days=dt),
    )

    if verbosity > 0:
        forecaster = prosumer.load.forecaster
        if hasattr(forecaster, "__len__"):
            print(f"Loaded {len(forecaster)} aligned predictions")
        print(f"Calendar feature channels: {list(CALENDAR_FEATURE_COLUMNS)}")
    return simulator
