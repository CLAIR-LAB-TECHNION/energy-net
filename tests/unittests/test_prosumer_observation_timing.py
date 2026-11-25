
"""Tests for POMDP observation timing and reset semantics."""

import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.prosumer_env import ProsumerEnv


def obs_value(env, observation, name):
    return float(env.encoder.field(observation, name)[0])


def _make_env(tmp_path, invalid_index=None):
    timestamps = pd.date_range(
        "2025-01-01", periods=4 * 48, freq="30min"
    )

    demand = 1.0 + np.arange(len(timestamps)) / 10.0
    forecasts = demand + 20.0
    recorded = demand.copy()
    if invalid_index is not None:
        recorded[invalid_index] = np.nan

    consumption_file = tmp_path / "consumption.csv"
    predictions_file = tmp_path / "predictions.csv"

    pd.DataFrame({
        "Datetime": timestamps,
        "Consumption": recorded,
    }).to_csv(consumption_file, index=False)

    predictions = pd.DataFrame({
        "timestamp": timestamps,
        "predicted_consumption": forecasts,
    })

    for feature in (
        "sin_time", "cos_time", "sin_week", "cos_week",
        "sin_year", "cos_year", "is_weekend"
    ):
        predictions[feature] = 0.0

    predictions.to_csv(predictions_file, index=False)

    env = ProsumerEnv(
        load_data_file=str(consumption_file),
        load_forecast_file=str(predictions_file),
        verbosity=0,
    )
    return env, timestamps, demand, forecasts


@pytest.fixture
def env_and_demand(tmp_path):
    env, timestamps, demand, forecasts = _make_env(tmp_path)
    try:
        yield env, timestamps, demand, forecasts
    finally:
        env.close()


def test_reset_uses_previous_load_without_processing_demand(
    env_and_demand,
):
    env, timestamps, demand, forecasts = env_and_demand
    start_idx = 48

    obs, _ = env.reset(options={"start_date": timestamps[start_idx]})

    assert env.current_step == 0
    assert env.current_datetime == timestamps[start_idx]

    # Observation contains the previous actual demand.
    assert obs_value(env, obs, "previous_load") == pytest.approx(
        demand[start_idx - 1] / 10.0
    )

    # Forecast begins with the imminent interval.
    assert obs_value(env, obs, "load_forecast") == pytest.approx(
        forecasts[start_idx] / 10.0
    )

    # Reset must not physically process current demand.
    assert env.simulator.prosumer.load.get_state() == pytest.approx(0.0)
    assert env.simulator.prosumer.storage_level() == pytest.approx(0.0)
    assert env.total_reward == pytest.approx(0.0)

    assert obs.shape == env.observation_space.shape


def test_step_updates_previous_load_for_next_observation(
    env_and_demand,
):
    env, timestamps, demand, forecasts = env_and_demand
    start_idx = 48

    env.reset(options={"start_date": timestamps[start_idx]})

    for offset in range(2):
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )

        realized = demand[start_idx + offset]

        # Current demand was settled in this transition.
        assert info["consumption_units"] == pytest.approx(
            realized
        )

        # The next observation reveals that realized demand.
        assert env.simulator.observe().previous_load == pytest.approx(realized)
        assert obs_value(env, obs, "previous_load") == pytest.approx(realized / 10.0)

        # The next forecast now refers to the next interval.
        assert obs_value(env, obs, "load_forecast") == pytest.approx(
            forecasts[start_idx + offset + 1] / 10.0
        )

        assert env.current_step == offset + 1
        assert env.current_datetime == (
            timestamps[start_idx + offset + 1]
        )
        assert not terminated
        assert not truncated
        assert reward == pytest.approx(info["step_money"])


def test_first_dataset_timestamp_requires_preceding_load(
    env_and_demand,
):
    env, timestamps, _, _ = env_and_demand

    with pytest.raises(ValueError, match="preceding"):
        env.reset(options={"start_date": timestamps[0]})


def test_default_reset_selects_first_eligible_timestamp(
    env_and_demand,
):
    env, timestamps, demand, _ = env_and_demand

    obs, _ = env.reset()

    assert env.current_datetime == timestamps[1]
    assert obs_value(env, obs, "previous_load") == pytest.approx(demand[0] / 10.0)
    assert env.simulator.prosumer.load.get_state() == pytest.approx(0.0)


def test_invalid_preceding_load_preserves_active_episode(tmp_path):
    # The reading preceding the episode at index 96 is invalid.
    env, timestamps, _, _ = _make_env(tmp_path, invalid_index=95)
    try:
        env.reset(options={"start_date": timestamps[1]})
        env.step(np.array([0.0], dtype=np.float32))

        def state():
            return (
                env.current_datetime,
                env.current_step,
                env.simulator.observe().previous_load,
                env.total_reward,
                env.simulator.prosumer.storage_level(),
            )

        old_state = state()
        with pytest.raises(ValueError, match="preceding"):
            env.reset(options={"start_date": timestamps[96]})
        assert state() == old_state
    finally:
        env.close()
