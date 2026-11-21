"""Calendar-alignment regression tests for the CSV-backed PCS environment."""

import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.pcs_env import PCSEnv


@pytest.fixture
def dated_pcs_env(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=3 * 48, freq="30min")
    consumption = np.repeat([1.0, 3.0, 5.0], 48) + np.tile(
        np.arange(48) / 64.0, 3
    )
    readings = pd.DataFrame({"Consumption": consumption}, index=timestamps)
    data_file = tmp_path / "consumption.csv"
    predictions_file = tmp_path / "predictions.csv"
    readings.to_csv(data_file, index_label="Datetime")
    # Perfect forecasts make a calendar mismatch unambiguous.
    pd.DataFrame({
        "timestamp": timestamps,
        "predicted_consumption": consumption,
    }).to_csv(predictions_file, index=False)

    env = PCSEnv(
        test_data_file=str(data_file),
        predictions_file=str(predictions_file),
        verbosity=0,
    )
    yield env, readings
    env.close()


@pytest.mark.parametrize(
    "start_index", [0, 48, 60],
    ids=["first-day-control", "second-day", "second-day-intraday"],
)
def test_reset_consumption_matches_selected_date(dated_pcs_env, start_index):
    env, readings = dated_pcs_env
    start_date = readings.index[start_index]

    obs, _ = env.reset(start_date=start_date)

    expected = readings.loc[start_date, "Consumption"]
    assert env.current_datetime == start_date
    assert obs[4] * 10.0 == pytest.approx(expected)
    assert obs[1] * 10.0 == pytest.approx(expected)
    assert env.pcs.get_consumption() == pytest.approx(expected)


@pytest.mark.parametrize(
    "start_index", [0, 48, 60],
    ids=["first-day-control", "second-day", "second-day-intraday"],
)
def test_step_consumption_matches_selected_day(dated_pcs_env, start_index):
    env, readings = dated_pcs_env
    start_date = readings.index[start_index]
    env.reset(start_date=start_date)

    actual = []
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )
        actual.append(info["consumption_units"])

    assert terminated
    assert not truncated
    assert env.current_datetime == start_date + pd.Timedelta(days=1)
    np.testing.assert_allclose(
        actual, readings.iloc[start_index:start_index + 48]["Consumption"],
        err_msg="Realized demand must use the selected calendar day's CSV rows",
    )


def test_reset_without_date_uses_next_days_consumption(dated_pcs_env):
    env, readings = dated_pcs_env
    env.reset(start_date=readings.index[0])
    for _ in range(env.max_steps):
        env.step(np.array([0.0], dtype=np.float32))

    obs, _ = env.reset()
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    expected = readings.iloc[48]["Consumption"]
    assert obs[4] * 10.0 == pytest.approx(expected)
    np.testing.assert_allclose(
        [obs[1] * 10.0, info["consumption_units"]],
        [expected, expected],
        err_msg="The next episode must not replay the first day's demand",
    )


def test_valid_episode_crosses_midnight(dated_pcs_env):
    env, readings = dated_pcs_env

    # January 2 at 18:00; 48 valid readings remain.
    start_index = 48 + 36
    start_date = readings.index[start_index]

    env.reset(start_date=start_date)

    actual = []
    for _ in range(env.max_steps):
        _, _, terminated, _, info = env.step(
            np.array([0.0], dtype=np.float32)
        )
        actual.append(info["consumption_units"])

    assert terminated
    np.testing.assert_allclose(
        actual,
        readings.iloc[
            start_index:start_index + env.max_steps
        ]["Consumption"].to_numpy(),
    )


def test_reject_episode_without_full_data(dated_pcs_env):
    env, readings = dated_pcs_env

    # January 3 at 18:00: only 12 readings remain.
    start_date = readings.index[-12]

    with pytest.raises(ValueError):
        env.reset(start_date=start_date)


def test_automatic_reset_wraps_between_episodes(dated_pcs_env):
    env, readings = dated_pcs_env

    # The last complete episode begins on January 3.
    env.reset(start_date=readings.index[-48])

    for _ in range(env.max_steps):
        _, _, terminated, _, _ = env.step(
            np.array([0.0], dtype=np.float32)
        )

    assert terminated

    # With no explicit date, wrap to the first valid episode.
    env.reset()

    assert env.current_datetime == readings.index[0]
    assert env.pcs.get_consumption() == pytest.approx(
        readings.iloc[0]["Consumption"]
    )


def test_rejected_reset_preserves_episode_state(dated_pcs_env):
    env, readings = dated_pcs_env

    env.reset(start_date=readings.index[0])
    env.step(np.array([0.0], dtype=np.float32))

    before = (
        env.current_step,
        env.current_datetime,
        env.total_reward,
        env.shortage_count,
        env.pcs.get_total_storage(),
    )

    with pytest.raises(ValueError):
        env.reset(start_date=readings.index[-12])

    after = (
        env.current_step,
        env.current_datetime,
        env.total_reward,
        env.shortage_count,
        env.pcs.get_total_storage(),
    )

    assert after == before


def test_non_midnight_dataset_start_preserves_consumption_alignment(tmp_path):
    timestamps = pd.date_range(
        "2025-01-01 00:30",
        periods=3 * 48,
        freq="30min",
    )
    demand = 1.0 + np.arange(len(timestamps)) * 0.01

    consumption_file = tmp_path / "consumption.csv"
    predictions_file = tmp_path / "predictions.csv"

    pd.DataFrame({
        "Datetime": timestamps,
        "Consumption": demand,
    }).to_csv(consumption_file, index=False)

    pd.DataFrame({
        "timestamp": timestamps,
        "predicted_consumption": demand,
    }).to_csv(predictions_file, index=False)

    env = PCSEnv(
        test_data_file=str(consumption_file),
        predictions_file=str(predictions_file),
        verbosity=0,
    )

    try:
        env.reset(start_date=timestamps[0])

        assert env.pcs.get_consumption() == pytest.approx(demand[0])

        for i in range(env.max_steps):
            _, _, terminated, _, info = env.step(
                np.array([0.0], dtype=np.float32)
            )

            assert info["consumption_units"] == pytest.approx(
                demand[i]
            ), f"Consumption mismatch at {timestamps[i]}"

        assert terminated

    finally:
        env.close()