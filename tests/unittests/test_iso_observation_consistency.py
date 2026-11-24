"""
An ISO pricing policy must see the same observation when it prices the
prosumer's day as when it was trained in the ISO env.

Regression: the legacy PCSEnv fed data-driven strategies demand / 10 while
ISOEnv trains on raw demand (and with a feature-free predictions CSV the
dimensions differed as well: 384 vs 48).
"""

import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.alternating_env import ISODailyObservation, get_pred_window
from energy_net.gym_envs.iso_env import ISOEnv
from energy_net.gym_envs.prosumer_env import ProsumerEnv
from energy_net.simulation import make_prosumer_simulator
from tests.golden.scenarios import (
    DATA_DIR,
    DEFAULT_FORECAST,
    DEFAULT_LOAD,
    RecordingDataDrivenStrategy,
)

DAY = pd.Timestamp("2025-07-10")


def _iso_env(predictions=DEFAULT_FORECAST):
    return ISOEnv(DEFAULT_LOAD, predictions, verbosity=0)


def _iso_index(iso_env, day):
    return int((day - iso_env.base_timestamp) / pd.Timedelta(minutes=30))


def test_default_daily_observation_uses_raw_forecasts():
    strategy = RecordingDataDrivenStrategy()
    simulator = make_prosumer_simulator(price_strategy=strategy)
    simulator.reset(DAY)

    (received,) = strategy.received
    expected_demand = simulator.prosumer.load.predict(
        DAY, pd.date_range(DAY, periods=48, freq="30min")
    )
    np.testing.assert_array_equal(received[:48], expected_demand)
    assert received.shape == (48 + 48 * 7,)


def test_default_matches_the_iso_observation_for_a_standard_predictions_csv():
    strategy = RecordingDataDrivenStrategy()
    make_prosumer_simulator(price_strategy=strategy).reset(DAY)
    iso_env = _iso_env()
    iso_obs = get_pred_window(iso_env, _iso_index(iso_env, DAY), iso_env.T)
    # ISOEnv stores float32; the calendar features are the CSV's own.
    np.testing.assert_allclose(strategy.received[0], iso_obs, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("predictions", [
    DEFAULT_FORECAST,
    f"{DATA_DIR}/consumption_predictions_without_features.csv",
])
def test_injected_iso_observation_is_identical_to_the_iso_env(predictions):
    iso_env = _iso_env(predictions)
    strategy = RecordingDataDrivenStrategy()
    env = ProsumerEnv(load_data_file=DEFAULT_LOAD, load_forecast_file=predictions,
                      price_strategy=strategy, verbosity=0)
    env.simulator.tariffs.daily_observation = ISODailyObservation(iso_env)

    env.reset(options={"start_date": DAY + pd.Timedelta(hours=12)})  # two days
    expected = [
        get_pred_window(iso_env, _iso_index(iso_env, day), iso_env.T)
        for day in (DAY, DAY + pd.Timedelta(days=1))
    ]
    assert len(strategy.received) == 2
    for got, want in zip(strategy.received, expected):
        np.testing.assert_array_equal(got, want)


def test_iso_observation_refuses_days_outside_the_iso_data(tmp_path):
    # The ISO only knows the first 10 days of the prosumer's data.
    actual, predicted = tmp_path / "iso_actual.csv", tmp_path / "iso_predicted.csv"
    pd.read_csv(DEFAULT_LOAD).head(10 * 48).to_csv(actual, index=False)
    pd.read_csv(DEFAULT_FORECAST).head(10 * 48).to_csv(predicted, index=False)
    iso_env = ISOEnv(str(actual), str(predicted), verbosity=0)
    source = ISODailyObservation(iso_env)

    before_data = iso_env.base_timestamp - pd.Timedelta(days=1)
    assert source.availability_error(before_data, before_data) is not None

    simulator = make_prosumer_simulator(price_strategy=RecordingDataDrivenStrategy())
    simulator.tariffs.daily_observation = source
    inside = iso_env.base_timestamp + pd.Timedelta(days=3)  # day 5 has invalid demand
    beyond = iso_env.base_timestamp + pd.Timedelta(days=20)
    assert simulator.episode_error(inside) is None
    # The prosumer has data for this day; only the ISO observation is missing.
    assert simulator.prosumer.load.dynamics.availability_error(
        simulator.episode_times(beyond)) is None
    assert simulator.episode_error(beyond) == f"No ISO observation for {beyond.date()}"
