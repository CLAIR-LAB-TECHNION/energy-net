"""Calendar-alignment regression tests for the CSV-backed prosumer environment."""


import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.prosumer_env import ProsumerEnv


def obs_value(env, observation, name):
    return float(env.encoder.field(observation, name)[0])


def previous_load(env):
    return env.simulator.observe().previous_load


@pytest.fixture
def dated_prosumer_env(tmp_path):
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

    env = ProsumerEnv(
        load_data_file=str(data_file),
        load_forecast_file=str(predictions_file),
        verbosity=0,
    )
    yield env, readings
    env.close()


@pytest.mark.parametrize(
    "start_index", [1, 48, 60],
    ids=["first-day-control", "second-day", "second-day-intraday"],
)
def test_reset_consumption_matches_selected_date(dated_prosumer_env, start_index):
    env, readings = dated_prosumer_env
    start_date = readings.index[start_index]

    obs, _ = env.reset(options={"start_date": start_date})

    previous_load = readings.loc[
    start_date - pd.Timedelta(minutes=30),
    "Consumption",]
    current_forecast = readings.loc[
        start_date, "Consumption"
    ]

    assert env.current_datetime == start_date
    assert obs_value(env, obs, "load_forecast") * 10.0 == pytest.approx(current_forecast)
    assert obs_value(env, obs, "previous_load") * 10.0 == pytest.approx(previous_load)
    assert env.simulator.prosumer.load.get_state() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "start_index", [1, 48, 60],
    ids=["first-day-control", "second-day", "second-day-intraday"],
)
def test_step_consumption_matches_selected_day(dated_prosumer_env, start_index):
    env, readings = dated_prosumer_env
    start_date = readings.index[start_index]
    env.reset(options={"start_date": start_date})

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


def test_reset_without_date_uses_next_days_consumption(
    dated_prosumer_env,
):
    env, readings = dated_prosumer_env

    env.reset(options={"start_date": readings.index[1]})
    for _ in range(env.max_steps):
        env.step(np.array([0.0], dtype=np.float32))

    obs, _ = env.reset()
    _, _, _, _, info = env.step(
        np.array([0.0], dtype=np.float32)
    )

    expected = readings.iloc[49]["Consumption"]
    previous = readings.iloc[48]["Consumption"]

    assert obs_value(env, obs, "load_forecast") * 10.0 == pytest.approx(expected)

    np.testing.assert_allclose(
        [obs_value(env, obs, "previous_load") * 10.0, info["consumption_units"]],
        [previous, expected],
        err_msg=(
            "Previous and imminent demand must belong "
            "to different intervals"
        ),
    )


def test_valid_episode_crosses_midnight(dated_prosumer_env):
    env, readings = dated_prosumer_env

    # January 2 at 18:00; 48 valid readings remain.
    start_index = 48 + 36
    start_date = readings.index[start_index]

    env.reset(options={"start_date": start_date})

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


def test_reject_episode_without_full_data(dated_prosumer_env):
    env, readings = dated_prosumer_env

    # January 3 at 18:00: only 12 readings remain.
    start_date = readings.index[-12]

    with pytest.raises(ValueError):
        env.reset(options={"start_date": start_date})


def test_automatic_reset_wraps_between_episodes(dated_prosumer_env):
    env, readings = dated_prosumer_env

    # The last complete episode begins on January 3.
    env.reset(options={"start_date": readings.index[-48]})

    for _ in range(env.max_steps):
        _, _, terminated, _, _ = env.step(
            np.array([0.0], dtype=np.float32)
        )

    assert terminated

    # With no explicit date, wrap to the first valid episode.
    env.reset()

    assert env.current_datetime == readings.index[1]
    assert previous_load(env) == pytest.approx(
        readings.iloc[0]["Consumption"]
    )
    assert env.simulator.prosumer.load.get_state() == pytest.approx(0.0)


def test_rejected_reset_preserves_episode_state(dated_prosumer_env):
    env, readings = dated_prosumer_env

    env.reset(options={"start_date": readings.index[1]})
    env.step(np.array([0.0], dtype=np.float32))

    before = (
        env.current_step,
        env.current_datetime,
        env.total_reward,
        env.simulator.metrics.grid_supplied_count,
        env.simulator.prosumer.storage_level(),
    )

    with pytest.raises(ValueError):
        env.reset(options={"start_date": readings.index[-12]})

    after = (
        env.current_step,
        env.current_datetime,
        env.total_reward,
        env.simulator.metrics.grid_supplied_count,
        env.simulator.prosumer.storage_level(),
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

    env = ProsumerEnv(
        load_data_file=str(consumption_file),
        load_forecast_file=str(predictions_file),
        verbosity=0,
    )

    try:
        env.reset(options={"start_date": timestamps[1]})

        assert previous_load(env) == pytest.approx(demand[0])
        assert env.simulator.prosumer.load.get_state() == pytest.approx(0.0)

        for i in range(env.max_steps):
            obs, _, terminated, _, info = env.step(
                np.array([0.0], dtype=np.float32)
            )

            expected_demand = demand[i + 1]

            # The prosumer processes current-interval demand.
            assert info["consumption_units"] == pytest.approx(
                expected_demand
            )

            assert env.simulator.prosumer.load.get_state() == pytest.approx(
                expected_demand
            )

            # The next observation contains the demand
            # that was just realized.
            if not terminated:
                # During the episode, the next observation exposes
                # the demand just realized.
                assert obs_value(env, obs, "previous_load") * 10.0 == pytest.approx(
                    expected_demand
                )
            else:
                # Terminal observation is a placeholder, not
                # another valid decision observation.
                assert np.all(obs == 0.0)

        assert terminated

    finally:
        env.close()


def test_gymnasium_options_start_date_matches_direct_reset(
    dated_prosumer_env,
):
    env, readings = dated_prosumer_env
    start = readings.index[60]

    direct_obs, _ = env.reset(options={"start_date": start})

    options_obs, info = env.reset(
        seed=123,
        options={"start_date": start.isoformat()},
    )

    np.testing.assert_array_equal(options_obs, direct_obs)

    assert env.current_datetime == start
    assert previous_load(env) == pytest.approx(
        readings.iloc[59]["Consumption"]
    )
    assert info == {}


def test_invalid_reset_options_preserve_active_state(
    dated_prosumer_env,
):
    env, readings = dated_prosumer_env

    env.reset(options={"start_date": readings.index[1]})
    env.step(np.array([2.0], dtype=np.float32))

    def state():
        return (
            env.current_datetime,
            env.current_step,
            previous_load(env),
            env.simulator.prosumer.storage_level(),
            env.total_reward,
            env.encoder.encode(env.simulator.observe()).tolist(),
        )

    old_state = state()

    with pytest.raises(ValueError, match="Unknown reset options"):
        env.reset(options={"start_date": readings.index[48],
                           "start": readings.index[60]})

    assert state() == old_state


def test_render_action_directions_and_debug_average(
    dated_prosumer_env, capsys
):
    env, readings = dated_prosumer_env
    env.reset(options={"start_date": readings.index[1]})

    env.step(np.array([2.0], dtype=np.float32))
    env.render(verbosity=3)

    assert "(Charge)" in capsys.readouterr().out

    env.step(np.array([-1.0], dtype=np.float32))
    env.render(verbosity=4)

    output = capsys.readouterr().out

    assert "Action Type:           DISCHARGE" in output

    assert (
        f"Average Reward/Step:   ${env.total_reward / 2:.6f}"
        in output
    )
