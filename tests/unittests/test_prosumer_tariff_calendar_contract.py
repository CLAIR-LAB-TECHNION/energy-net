
"""Prosumer POMDP calendar and tariff-admission tests."""

import numpy as np
import pandas as pd
import pytest

from energy_net.simulation import CALENDAR_FEATURE_COLUMNS, calendar_features
from energy_net.grid_entities.management.price_curve import (
    PriceCurveStrategy,
    FixedPriceStrategy,
)
from tests.unittests.test_prosumer_episode_data_validation import (
    _files,
    _env,
)


class ExplicitTariffStrategy(PriceCurveStrategy):
    requires_daily_observation = False

    def __init__(self, buy, sell):
        self.buy = buy
        self.sell = sell
        self.calls = 0

    def calculate_prosumer_tariffs(self, observation=None):
        self.calls += 1
        return self.buy, self.sell


@pytest.mark.parametrize("buy,sell,error", [
    (
        np.full(48, np.nan),
        np.full(48, .01),
        "finite",
    ),
    (
        np.full(48, .05),
        np.full(48, np.inf),
        "finite",
    ),
    (
        np.full(48, -.01),
        np.zeros(48),
        "nonnegative",
    ),
    (
        np.full(48, .02),
        np.full(48, .03),
        "sell tariff",
    ),
    (
        np.full(47, .10),
        np.full(48, .05),
        "48-slot",
    ),
])
def test_reject_invalid_price_curves_before_reset(
    tmp_path, buy, sell, error
):
    actual, predictions, timestamps, _ = _files(tmp_path)

    strategy = ExplicitTariffStrategy(buy, sell)
    env = _env(
        actual, predictions, price_strategy=strategy
    )

    try:
        with pytest.raises(ValueError, match=error):
            env.reset(options={"start_date": timestamps[48]})

        assert not env.simulator.has_episode
        assert env.current_step == 0
        assert env.simulator.prosumer.storage_level() == pytest.approx(0)

    finally:
        env.close()


def test_strategy_change_is_atomic_and_refreshes_observation(
    tmp_path,
):
    actual, predictions, timestamps, _ = _files(tmp_path)

    env = _env(
        actual, predictions,
        price_strategy=FixedPriceStrategy(.10),
    )

    try:
        original_obs, _ = env.reset(options={"start_date": timestamps[48]})

        assert env.encoder.field(original_obs, "buy_tariff")[0] == pytest.approx(.10)

        old_strategy = env.simulator.price_strategy
        old_cache = env.simulator.tariffs.cached_days()

        with pytest.raises(ValueError, match="sell tariff"):
            env.set_price_strategy(
                ExplicitTariffStrategy(
                    np.full(48, .02),
                    np.full(48, .03),
                )
            )

        assert env.simulator.price_strategy is old_strategy
        assert env.simulator.tariffs.cached_days() == old_cache

        decision = env.encoder.encode(env.simulator.observe())
        assert decision.tolist() == pytest.approx(original_obs.tolist())

        new_obs = env.set_price_strategy(
            FixedPriceStrategy(.15)
        )

        assert env.encoder.field(new_obs, "buy_tariff")[0] == pytest.approx(.15)
        assert env.encoder.field(new_obs, "sell_tariff")[0] == pytest.approx(.15)

        decision = env.encoder.encode(env.simulator.observe())
        assert decision.tolist() == pytest.approx(new_obs.tolist())
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert info["observation_at_decision"] == pytest.approx(new_obs.tolist())

    finally:
        env.close()


def test_terminal_prices_precomputed_for_cross_midnight_episode(
    tmp_path,
):
    actual, predictions, timestamps, _ = _files(tmp_path)

    strategy = ExplicitTariffStrategy(
        np.full(48, .15),
        np.full(48, .05),
    )

    env = _env(
        actual, predictions, price_strategy=strategy
    )

    try:
        env.reset(options={"start_date": pd.Timestamp("2025-01-02 18:00")})

        # One curve for January 2, one for January 3.
        assert strategy.calls == 2

        for _ in range(48):
            obs, reward, done, _, info = env.step(
                np.array([0.0], dtype=np.float32)
            )

        # No additional curves were generated during steps.
        assert done
        assert strategy.calls == 2

        assert info["terminal_state"]["episode_step"] == 48
        assert np.all(obs == 0.0)

    finally:
        env.close()


def test_calendar_channels_have_documented_order_and_values(
    tmp_path,
):
    actual, predictions, timestamps, _ = _files(tmp_path)

    env = _env(actual, predictions)

    try:
        t = pd.Timestamp("2025-01-02 18:00")
        obs, _ = env.reset(options={"start_date": t})

        j = 36
        weekday = t.day_of_week
        doy = t.day_of_year

        expected = [
            np.sin(2 * np.pi * j / 48),
            np.cos(2 * np.pi * j / 48),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            float(weekday >= 5),
        ]

        assert tuple(env.simulator.observe().calendar_columns) == (
            CALENDAR_FEATURE_COLUMNS
        )

        assert obs.shape == env.observation_space.shape

        np.testing.assert_allclose(
            env.encoder.field(obs, "calendar"), expected, atol=1e-6
        )

        day = pd.date_range(t, periods=48, freq="30min")
        assert calendar_features(day).size == 336

    finally:
        env.close()
