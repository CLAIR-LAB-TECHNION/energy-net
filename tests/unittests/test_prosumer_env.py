"""Gym-level tests for ProsumerEnv (adapter behavior, not physics)."""

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from energy_net.dynamics import CSVDynamics
from energy_net.forecasting import TableForecaster
from energy_net.gym_envs.prosumer_env import ProsumerEnv
from energy_net.simulation import (
    CALENDAR_FEATURE_COLUMNS,
    NetMeteringReward,
    make_prosumer_simulator,
)


def test_passes_the_gymnasium_env_checker():
    check_env(ProsumerEnv(verbosity=0), skip_render_check=True)


def test_layout_is_named_and_covers_the_vector():
    env = ProsumerEnv(verbosity=0)
    horizon = env.simulator.forecast_horizon
    assert env.observation_space.shape == (14 + 3 * horizon,)
    covered = sorted(
        index for part in env.observation_layout.values()
        for index in range(part.start, part.stop)
    )
    assert covered == list(range(env.observation_space.shape[0]))
    assert env.observation_layout["calendar"].stop - env.observation_layout["calendar"].start == len(
        CALENDAR_FEATURE_COLUMNS
    )


def test_generation_is_observed_and_settled(tmp_path):
    times = pd.date_range("2025-01-01", periods=3 * 48, freq="30min")
    generation = pd.Series(np.full(len(times), 2.0), index=times)
    from energy_net.simulation import make_prosumer

    prosumer = make_prosumer(
        generation_dynamics=CSVDynamics(generation, "generation"),
        generation_forecaster=TableForecaster(
            pd.DataFrame({"predicted_production": 1.5}, index=times),
            "predicted_production",
        ),
        load_dynamics=CSVDynamics(pd.Series(np.full(len(times), 0.5), index=times), "consumption"),
        load_forecaster=TableForecaster(
            pd.DataFrame({"predicted_consumption": 0.5}, index=times),
            "predicted_consumption",
        ),
    )
    env = ProsumerEnv(make_prosumer_simulator(prosumer=prosumer), verbosity=0)
    obs, _ = env.reset(options={"start_date": times[1]})
    assert env.encoder.field(obs, "previous_generation")[0] == pytest.approx(0.2)
    assert env.encoder.field(obs, "generation_forecast")[0] == pytest.approx(0.15)
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["generation_units"] == 2.0
    assert info["battery_charged_from_surplus_units"] == pytest.approx(1.5)
    assert env.ground_truth("generator", times[2])[0] == 2.0


def test_reset_options_contract():
    env = ProsumerEnv(verbosity=0)
    obs, info = env.reset(options={"start_date": "2025-07-10", "initial_storage": 40.0})
    assert info == {}
    assert env.encoder.field(obs, "storage")[0] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="Unknown reset options"):
        env.reset(options={"start": "2025-07-10"})
    with pytest.raises(TypeError):
        env.reset(options=["2025-07-10"])


def test_injected_simulator_and_reward_are_used_as_is():
    simulator = make_prosumer_simulator()
    reward = NetMeteringReward()
    env = ProsumerEnv(simulator, reward, verbosity=0)
    assert env.simulator is simulator and env.reward_fn is reward
    with pytest.raises(ValueError, match="not both"):
        ProsumerEnv(simulator, load_data_file="x.csv")


def test_set_price_strategy_returns_the_refreshed_observation():
    from energy_net.grid_entities.management.price_curve import FixedPriceStrategy

    env = ProsumerEnv(verbosity=0)
    assert env.set_price_strategy(FixedPriceStrategy(0.1)) is None
    env.reset(options={"start_date": "2025-07-10"})
    refreshed = env.set_price_strategy(FixedPriceStrategy(0.3))
    assert env.encoder.field(refreshed, "buy_tariff")[0] == pytest.approx(0.3)
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["observation_at_decision"] == refreshed.tolist()


@pytest.mark.parametrize("verbosity", [0, 1, 2, 3, 4])
def test_render_does_not_change_state(verbosity, capsys):
    env = ProsumerEnv(verbosity=0)
    obs, _ = env.reset(options={"start_date": "2025-07-10"})
    env.step(np.array([-2.0], dtype=np.float32))
    before = (env.simulator.observe(), env.simulator.prosumer.storage_level(),
              env.simulator.tariffs.cached_days())
    data = env.render(verbosity=verbosity)
    assert {"step", "storage", "price"} <= set(data)
    assert (env.simulator.observe(), env.simulator.prosumer.storage_level(),
            env.simulator.tariffs.cached_days()) == before
    printed = capsys.readouterr().out
    assert (printed == "") == (verbosity in (0, 1))


def test_trains_with_stable_baselines3():
    sb3 = pytest.importorskip("stable_baselines3")
    env = ProsumerEnv(verbosity=0)
    model = sb3.PPO("MlpPolicy", env, n_steps=48, batch_size=48, verbose=0)
    model.learn(total_timesteps=96)
    action, _ = model.predict(env.reset()[0], deterministic=True)
    assert env.action_space.contains(action.astype(np.float32))


def test_seeded_reset_is_reproducible_and_unseeded_reset_advances():
    env = ProsumerEnv(verbosity=0)
    first, _ = env.reset(seed=3)
    start = env.current_datetime
    for _ in range(env.max_steps):
        env.step(np.array([1.0], dtype=np.float32))
    env.reset()
    assert env.current_datetime == start + env.simulator.dt * env.max_steps
    again, _ = env.reset(seed=3)
    assert env.current_datetime == start
    np.testing.assert_array_equal(first, again)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_actions_are_rejected_without_side_effects(bad):
    env = ProsumerEnv(verbosity=0)
    env.reset(options={"start_date": "2025-07-10", "initial_storage": 20.0})
    before = (env.current_step, env.simulator.prosumer.storage_level(), env.simulator.observe())
    with pytest.raises(ValueError, match="must be finite"):
        env.step(np.array([bad], dtype=np.float32))
    assert (env.current_step, env.simulator.prosumer.storage_level(),
            env.simulator.observe()) == before
