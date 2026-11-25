"""Specification tests for electricity settlement and experimental shaping."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from energy_net.grid_entities.management.price_curve import ActionBasedPriceStrategy
from energy_net.gym_envs.prosumer_env import ProsumerEnv
from energy_net.simulation import make_reward, net_meter_money

OBS_SIZE = 14 + 3 * 48  # named layout, horizon 48


def obs_value(env, observation, name):
    """First value of a named observation field."""
    return float(env.encoder.field(observation, name)[0])


@pytest.mark.parametrize("exchange, buy, sell, expected", [
    (5.0, 0.10, 0.10, -0.50),
    (-7.0, 0.20, 0.05, 0.35),
    (0.0, 0.20, 0.05, 0.0),
])
def test_net_meter_money(exchange, buy, sell, expected):
    assert net_meter_money(exchange, buy, sell) == pytest.approx(expected)


def test_optional_penalty_preserves_bill():
    supplied = SimpleNamespace(money=-0.5, grid_supplied=True)
    covered = SimpleNamespace(money=-0.5, grid_supplied=False)
    assert make_reward(True, 7.0)(supplied).total == pytest.approx(-0.5 - 7.0)
    assert make_reward(True, 7.0)(covered).total == -0.5
    assert make_reward(False, 7.0)(supplied).total == -0.5


def _env(tmp_path, penalized=False, demand=5.0):
    timestamps = pd.date_range("2025-01-01", periods=3 * 48, freq="30min")
    actual = tmp_path / "consumption.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"Datetime": timestamps, "Consumption": demand}).to_csv(actual, index=False)
    pd.DataFrame({
        "timestamp": timestamps,
        "predicted_consumption": demand,
        **{key: np.zeros(len(timestamps)) for key in (
            "sin_time", "cos_time", "sin_week", "cos_week",
            "sin_year", "cos_year", "is_weekend"
        )},
    }).to_csv(predictions, index=False)

    # ISO buys from the prosumer at 0.05, sells to it at 0.20.
    iso_buy_action = np.full(48, 0.25, dtype=np.float32)
    iso_sell_action = np.ones(48, dtype=np.float32)
    price_strategy = ActionBasedPriceStrategy(
        action=np.r_[iso_buy_action, iso_sell_action],
        use_asymmetric_pricing=True,
        price_min=0.0, price_max=0.20,
    )
    return ProsumerEnv(
        load_data_file=str(actual), load_forecast_file=str(predictions),
        price_strategy=price_strategy, include_shortage_penalty=penalized,
        shortage_penalty=7.0, verbosity=0,
    )


@pytest.mark.parametrize("penalized", [False, True])
def test_two_tariffs_and_reward_variants(tmp_path, penalized):
    env = _env(tmp_path, penalized=penalized)
    try:
        observation, _ = env.reset(options={"start_date": "2025-01-02 00:00"})
        assert observation.dtype == np.float32
        assert observation.shape == (OBS_SIZE,)
        assert env.observation_space.shape == (OBS_SIZE,)
        assert obs_value(env, observation, "buy_tariff") == pytest.approx(0.20)
        assert obs_value(env, observation, "sell_tariff") == pytest.approx(0.05)
        assert obs_value(env, observation, "load_forecast") == pytest.approx(5.0 / 10.0)

        _, reward, _, _, info = env.step(np.array([3.0], dtype=np.float32))
        assert info["energy_sold_or_bought_units"] == pytest.approx(3.0)
        assert info["uncovered_consumption_units"] == pytest.approx(2.0)
        assert info["grid_exchange_units"] == pytest.approx(5.0)
        assert info["step_money"] == pytest.approx(-1.0)
        assert info["shortage_penalty"] == pytest.approx(7.0 if penalized else 0.0)
        assert reward == pytest.approx(-8.0 if penalized else -1.0)
        assert env.simulator.metrics.total_money == pytest.approx(-1.0)
    finally:
        env.close()


def test_observed_tariffs_match_settlement_at_nonmidnight_start(tmp_path):
    env = _env(tmp_path)
    try:
        observation, _ = env.reset(options={"start_date": "2025-01-02 18:00"})
        assert obs_value(env, observation, "buy_tariff") == pytest.approx(.20)
        assert obs_value(env, observation, "sell_tariff") == pytest.approx(.05)
        for i in range(13):  # crosses the midnight boundary
            buy = obs_value(env, observation, "buy_tariff")
            sell = obs_value(env, observation, "sell_tariff")
            observation, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
            assert info["buy_tariff"] == pytest.approx(buy)
            assert info["sell_tariff"] == pytest.approx(sell)
        assert env.current_datetime == pd.Timestamp("2025-01-03 00:30")
    finally:
        env.close()


def test_export_is_valued_at_prosumer_sell_tariff(tmp_path):
    env = _env(tmp_path, demand=3.0)
    try:
        env.reset(options={"start_date": "2025-01-02 00:00", "initial_storage": 10.0})
        _, reward, _, _, info = env.step(np.array([-8.0], dtype=np.float32))
        assert info["storage_after_units"] == pytest.approx(0.0)
        assert info["uncovered_consumption_units"] == pytest.approx(1.0)
        assert info["grid_exchange_units"] == pytest.approx(-7.0)
        assert info["step_money"] == pytest.approx(0.35)
        assert reward == pytest.approx(0.35)
    finally:
        env.close()


def test_tariff_slot_uses_calendar_clock_across_midnight(tmp_path):
    env = _env(tmp_path)
    # ISO buy (prosumer sell) tariff varies by calendar slot; ISO sell
    # (prosumer buy) tariff is fixed at 0.20 to keep non-inverted tariffs.
    iso_buy = np.arange(48, dtype=np.float32) / 100.0
    iso_sell = np.ones(48, dtype=np.float32)
    env.set_price_strategy(ActionBasedPriceStrategy(
        action=np.r_[iso_buy, iso_sell],
        use_asymmetric_pricing=True, price_min=0.0, price_max=0.20,
    ))
    try:
        observation, _ = env.reset(options={"start_date": "2025-01-02 18:00"})
        assert obs_value(env, observation, "sell_tariff") == pytest.approx(0.20 * 0.36)
        for _ in range(11):
            observation, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert env.current_datetime == pd.Timestamp("2025-01-02 23:30")
        assert obs_value(env, observation, "sell_tariff") == pytest.approx(0.20 * 0.47)
        observation, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert info["sell_tariff"] == pytest.approx(0.20 * 0.47)
        assert env.current_datetime == pd.Timestamp("2025-01-03 00:00")
        assert obs_value(env, observation, "sell_tariff") == pytest.approx(0.0)
    finally:
        env.close()


def test_strategy_change_invalidates_cached_tariffs(tmp_path):
    env = _env(tmp_path)
    try:
        obs, _ = env.reset(options={"start_date": "2025-01-02 00:00"})
        assert obs_value(env, obs, "buy_tariff") == pytest.approx(0.20)
        changed = env.set_price_strategy(ActionBasedPriceStrategy(
            action=np.full(48, 0.5, dtype=np.float32),
            use_asymmetric_pricing=False, price_min=0.0, price_max=0.20,
        ))
        assert obs_value(env, changed, "buy_tariff") == pytest.approx(.10)
        assert obs_value(env, changed, "sell_tariff") == pytest.approx(.10)
    finally:
        env.close()


def test_rl_price_strategy_uses_one_prediction_for_both_tariffs(tmp_path):
    from unittest.mock import Mock
    from energy_net.grid_entities.management.price_curve import RLPriceCurveStrategy
    mock_iso = Mock()
    mock_iso.predict.return_value = (
        np.r_[np.full(48, .25), np.ones(48)].astype(np.float32), None
    )
    env = _env(tmp_path)
    env.set_price_strategy(RLPriceCurveStrategy(
        iso_model=mock_iso, use_asymmetric_pricing=True,
        price_min=0.0, price_max=0.20,
    ))
    try:
        observation, _ = env.reset(options={"start_date": "2025-01-02 00:00"})
        assert obs_value(env, observation, "buy_tariff") == pytest.approx(.20)
        assert obs_value(env, observation, "sell_tariff") == pytest.approx(.05)
        env.step(np.array([0.0], dtype=np.float32))
        assert mock_iso.predict.call_count == 1
    finally:
        env.close()
