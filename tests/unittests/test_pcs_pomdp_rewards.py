"""Specification tests for electricity settlement and experimental shaping."""

import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.pcs_rewards import (
    net_meter_reward,
    net_meter_reward_with_shortage_penalty,
)
from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.grid_entities.management.price_curve import ActionBasedPriceStrategy


@pytest.mark.parametrize("exchange, buy, sell, expected", [
    (5.0, 0.10, 0.10, -0.50),
    (-7.0, 0.20, 0.05, 0.35),
    (0.0, 0.20, 0.05, 0.0),
])
def test_net_meter_reward(exchange, buy, sell, expected):
    assert net_meter_reward(exchange, buy, sell) == pytest.approx(expected)


def test_optional_penalty_preserves_bill():
    baseline = net_meter_reward(5.0, 0.10, 0.05)
    shaped = net_meter_reward_with_shortage_penalty(5.0, 0.10, 0.05, 2.0, 7.0)
    assert shaped == pytest.approx(baseline - 7.0)
    assert net_meter_reward_with_shortage_penalty(5.0, 0.10, 0.05, 0.0, 7.0) == baseline


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

    # ISO buys from PCS at 0.05, sells to PCS at 0.20.
    iso_buy_action = np.full(48, 0.25, dtype=np.float32)
    iso_sell_action = np.ones(48, dtype=np.float32)
    price_strategy = ActionBasedPriceStrategy(
        action=np.r_[iso_buy_action, iso_sell_action],
        use_asymmetric_pricing=True,
        price_min=0.0, price_max=0.20,
    )
    return PCSEnv(
        test_data_file=str(actual), predictions_file=str(predictions),
        price_strategy=price_strategy, include_shortage_penalty=penalized,
        shortage_penalty=7.0, verbosity=0,
    )


@pytest.mark.parametrize("penalized", [False, True])
def test_two_tariffs_and_reward_variants(tmp_path, penalized):
    env = _env(tmp_path, penalized=penalized)
    try:
        observation, _ = env.reset(start_date="2025-01-02 00:00")
        assert observation.dtype == np.float32
        assert observation.shape == (109,)
        assert env.observation_space.shape == (109,)
        assert observation[2] == pytest.approx(0.20)  # PCS buy/import tariff
        assert observation[3] == pytest.approx(0.05)  # PCS sell/export tariff
        assert observation[4] == pytest.approx(5.0 / 10.0)  # forecast starts here

        _, reward, _, _, info = env.step(np.array([3.0], dtype=np.float32))
        assert info["energy_sold_or_bought_units"] == pytest.approx(3.0)
        assert info["uncovered_consumption_units"] == pytest.approx(2.0)
        assert info["grid_exchange_units"] == pytest.approx(5.0)
        assert info["step_money"] == pytest.approx(-1.0)
        assert info["shortage_penalty"] == pytest.approx(7.0 if penalized else 0.0)
        assert reward == pytest.approx(-8.0 if penalized else -1.0)
        assert env.get_money() == pytest.approx(-1.0)
    finally:
        env.close()


def test_observed_tariffs_match_settlement_at_nonmidnight_start(tmp_path):
    env = _env(tmp_path)
    try:
        observation, _ = env.reset(start_date="2025-01-02 18:00")
        assert observation[2] == pytest.approx(.20)
        assert observation[3] == pytest.approx(.05)
        for i in range(13):  # crosses the midnight boundary
            buy, sell = float(observation[2]), float(observation[3])
            observation, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
            assert info["buy_tariff"] == pytest.approx(buy)
            assert info["sell_tariff"] == pytest.approx(sell)
        assert env.current_datetime == pd.Timestamp("2025-01-03 00:30")
    finally:
        env.close()


def test_export_is_valued_at_pcs_sell_tariff(tmp_path):
    env = _env(tmp_path, demand=3.0)
    try:
        env.reset(start_date="2025-01-02 00:00")
        env.pcs.storage_units[0].reset(10.0)
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
    # ISO buy (PCS sell) tariff varies by calendar slot; ISO sell (PCS buy)
    # tariff is fixed at 0.20 to keep non-inverted tariffs.
    iso_buy = np.arange(48, dtype=np.float32) / 100.0
    iso_sell = np.ones(48, dtype=np.float32)
    env.set_price_strategy(ActionBasedPriceStrategy(
        action=np.r_[iso_buy, iso_sell],
        use_asymmetric_pricing=True, price_min=0.0, price_max=0.20,
    ))
    try:
        observation, _ = env.reset(start_date="2025-01-02 18:00")
        assert observation[3] == pytest.approx(0.20 * 0.36)
        for _ in range(11):
            observation, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert env.current_datetime == pd.Timestamp("2025-01-02 23:30")
        assert observation[3] == pytest.approx(0.20 * 0.47)
        observation, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert info["sell_tariff"] == pytest.approx(0.20 * 0.47)
        assert env.current_datetime == pd.Timestamp("2025-01-03 00:00")
        assert observation[3] == pytest.approx(0.0)
    finally:
        env.close()


def test_strategy_change_invalidates_cached_tariffs(tmp_path):
    env = _env(tmp_path)
    try:
        obs, _ = env.reset(start_date="2025-01-02 00:00")
        assert obs[2] == pytest.approx(0.20)
        env.set_price_strategy(ActionBasedPriceStrategy(
            action=np.full(48, 0.5, dtype=np.float32),
            use_asymmetric_pricing=False, price_min=0.0, price_max=0.20,
        ))
        changed = env._get_obs()
        assert changed[2] == pytest.approx(.10)
        assert changed[3] == pytest.approx(.10)
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
        observation, _ = env.reset(start_date="2025-01-02 00:00")
        assert observation[2] == pytest.approx(.20)
        assert observation[3] == pytest.approx(.05)
        env.step(np.array([0.0], dtype=np.float32))
        assert mock_iso.predict.call_count == 1
    finally:
        env.close()
