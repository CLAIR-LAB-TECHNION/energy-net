"""
Regression: loops that start prosumer days must skip days the prosumer
cannot run (invalid data, no preceding interval) and wrap at the end of the
data, exactly like AlternatingISOEnv.reset.

The default test data has invalid demand on 2025-07-14, and its first row
(2025-07-09 00:00) has no preceding interval.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from energy_net.gym_envs.alternating_env import (  # noqa: E402
    next_eligible_start,
    run_alternating_training,
)
from energy_net.gym_envs.evaluate_alternating import AlternatingEvaluator  # noqa: E402
from energy_net.gym_envs.iso_env import ISOEnv  # noqa: E402
from energy_net.gym_envs.prosumer_env import ProsumerEnv  # noqa: E402
from tests.golden.scenarios import DEFAULT_FORECAST, DEFAULT_LOAD  # noqa: E402

DAY = 48  # half-hour rows per day
JULY_14 = 5 * DAY


@pytest.fixture(scope="module")
def envs():
    iso_env = ISOEnv(DEFAULT_LOAD, DEFAULT_FORECAST, verbosity=0)
    prosumer_env = ProsumerEnv(load_data_file=DEFAULT_LOAD,
                               load_forecast_file=DEFAULT_FORECAST, verbosity=0)
    return iso_env, prosumer_env


def _date(iso_env, index):
    return (iso_env.base_timestamp + pd.Timedelta(minutes=30 * index)).date()


def test_next_eligible_start_skips_days_the_prosumer_cannot_run(envs):
    iso_env, prosumer_env = envs
    assert prosumer_env.simulator.episode_error(
        iso_env.base_timestamp + pd.Timedelta(minutes=30 * JULY_14)
    ) is not None
    assert next_eligible_start(iso_env, prosumer_env, JULY_14) == JULY_14 + DAY
    # The first row has no preceding interval.
    assert next_eligible_start(iso_env, prosumer_env, 0) == DAY


def test_next_eligible_start_wraps_at_the_end_of_the_data(envs):
    iso_env, prosumer_env = envs
    past_the_end = iso_env.total_rows
    assert next_eligible_start(iso_env, prosumer_env, past_the_end) == DAY


def test_alternating_training_survives_an_invalid_day():
    # Iteration 2 used to start the prosumer on 2025-07-14 and raise
    # "Cannot start episode at 2025-07-14 00:00:00".
    history = run_alternating_training(cycle_days=2, total_iterations=2)
    assert history["iteration"] == [1, 2]
    assert all(np.isfinite(history["avg_money"]))


class _ConstantModel:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        return self.action, None


def test_evaluation_skips_days_the_prosumer_cannot_run():
    evaluator = AlternatingEvaluator(
        iso_model=_ConstantModel(np.full(2 * DAY, 0.5)),
        prosumer_model=_ConstantModel([0.0]),
        actual_csv=DEFAULT_LOAD,
        predicted_csv=DEFAULT_FORECAST,
    )
    # start_idx=0 (the first row) used to raise immediately.
    history = evaluator.run_evaluation(num_days=6, start_idx=0)

    days = sorted({timestamp.date() for timestamp in history["timestamp"]})
    assert len(days) == 6
    assert pd.Timestamp("2025-07-09").date() not in days  # no preceding interval
    assert pd.Timestamp("2025-07-14").date() not in days  # invalid demand
    assert len(history) == 6 * DAY
