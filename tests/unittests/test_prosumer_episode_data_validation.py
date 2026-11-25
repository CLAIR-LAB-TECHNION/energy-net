
"""Episode eligibility and timestamp-aligned POMDP input tests."""

import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.prosumer_env import ProsumerEnv
from energy_net.grid_entities.management.price_curve import (
    PriceCurveStrategy,
)

FEATURES = (
    "sin_time", "cos_time", "sin_week", "cos_week",
    "sin_year", "cos_year", "is_weekend",
)


def _files(tmp_path, *, forecast_offset=0, scramble_forecasts=False):
    timestamps = pd.date_range(
        "2025-01-01", periods=4 * 48, freq="30min"
    )
    consumption_path = tmp_path / "consumption.csv"
    forecast_path = tmp_path / "forecasts.csv"

    demand = 2.0 + np.arange(len(timestamps)) / 1000.0
    pd.DataFrame({
        "Datetime": timestamps,
        "Consumption": demand,
    }).to_csv(consumption_path, index=False)

    forecast_times = pd.date_range(
        timestamps[0] - pd.Timedelta(minutes=30 * forecast_offset),
        periods=len(timestamps) + forecast_offset,
        freq="30min",
    )
    forecasts = pd.DataFrame({
        "timestamp": forecast_times,
        "predicted_consumption": (
            10.0 + np.arange(len(forecast_times)) / 10.0
        ),
        **{
            name: np.full(len(forecast_times), j + 1.0)
            for j, name in enumerate(FEATURES)
        },
    })

    if scramble_forecasts:
        forecasts = forecasts.sample(frac=1.0, random_state=5)

    forecasts.to_csv(forecast_path, index=False)
    return consumption_path, forecast_path, timestamps, forecasts


def _env(consumption_path, forecast_path, **kwargs):
    return ProsumerEnv(
        load_data_file=str(consumption_path),
        load_forecast_file=str(forecast_path),
        verbosity=0,
        **kwargs,
    )


def test_shuffled_forecasts_with_earlier_start_are_aligned_by_timestamp(
    tmp_path,
):
    actual, predicted, timestamps, forecast_df = _files(
        tmp_path, forecast_offset=10, scramble_forecasts=True
    )

    env = _env(actual, predicted)
    try:
        start = timestamps[48]
        obs, _ = env.reset(options={"start_date": start})

        expected = forecast_df.loc[
            forecast_df["timestamp"] == start,
            "predicted_consumption",
        ].iloc[0]

        assert env.encoder.field(obs, "load_forecast")[0] == pytest.approx(expected / 10.0)
        # Calendar features now come from timestamps,
        # not arbitrary values in the forecast CSV.
        weekday = start.day_of_week
        doy = start.day_of_year

        expected_features = np.array([
            0.0,
            1.0,
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            float(weekday >= 5),
        ])

        np.testing.assert_allclose(
            env.encoder.field(obs, "calendar"),
            expected_features,
            atol=1e-6,
        )

        for step in range(48):
            obs, _, done, _, info = env.step(
                np.array([0.0], dtype=np.float32)
            )
            assert info["consumption_units"] == pytest.approx(
                2.0 + (48 + step) / 1000.0
            )

        assert done
    finally:
        env.close()


def test_invalid_realized_demand_is_rejected_before_episode(tmp_path):
    actual, predicted, timestamps, _ = _files(tmp_path)
    readings = pd.read_csv(actual)
    readings.loc[60, "Consumption"] = np.nan
    readings.to_csv(actual, index=False)

    env = _env(actual, predicted)
    try:
        with pytest.raises(
            ValueError, match="Non-finite or negative.*consumption"
        ):
            env.reset(options={"start_date": timestamps[48]})

        assert env.current_step == 0
        assert env.simulator.prosumer.storage_level() == 0.0

        # A later episode that avoids the invalid row is eligible.
        env.reset(options={"start_date": timestamps[80]})
    finally:
        env.close()


@pytest.mark.parametrize("column", ["predicted_consumption", "sin_time"])
def test_invalid_forecasts_or_features_are_rejected(tmp_path, column):
    actual, predicted, timestamps, _ = _files(tmp_path)
    forecasts = pd.read_csv(predicted)
    forecasts.loc[60, column] = np.nan
    forecasts.to_csv(predicted, index=False)

    env = _env(actual, predicted)
    try:
        with pytest.raises(
            ValueError, match="Non-finite forecast or calendar"
        ):
            env.reset(options={"start_date": timestamps[48]})
    finally:
        env.close()


def test_missing_forecast_timestamp_is_rejected(tmp_path):
    actual, predicted, timestamps, _ = _files(tmp_path)
    forecasts = pd.read_csv(predicted)
    forecasts.drop(index=60).to_csv(predicted, index=False)

    env = _env(actual, predicted)
    try:
        with pytest.raises(
            ValueError, match="Missing forecast/calendar timestamp"
        ):
            env.reset(options={"start_date": timestamps[48]})
    finally:
        env.close()


def test_reject_duplicate_forecast_timestamps_at_load(tmp_path):
    actual, predicted, _, _ = _files(tmp_path)
    forecasts = pd.read_csv(predicted)
    forecasts = pd.concat(
        [forecasts, forecasts.iloc[[5]]],
        ignore_index=True,
    )
    forecasts.to_csv(predicted, index=False)

    with pytest.raises(
        ValueError, match="Forecast timestamps must be unique"
    ):
        _env(actual, predicted)


def test_terminal_transition_does_not_request_next_day_data(tmp_path):
    actual, predicted, timestamps, _ = _files(tmp_path)
    env = _env(actual, predicted)

    try:
        env.reset(options={"start_date": timestamps[-48]})

        for step in range(48):
            obs, reward, done, truncated, info = env.step(
                np.array([0.0], dtype=np.float32)
            )
            assert done == (step == 47)
            assert not truncated

        assert obs.shape == env.observation_space.shape
        assert np.all(obs == 0.0)

        with pytest.raises(RuntimeError, match="terminated"):
            env.step(np.array([0.0], dtype=np.float32))
    finally:
        env.close()


class DataDrivenPrice(PriceCurveStrategy):
    def calculate_price(self, observation=None):
        assert observation is not None
        return np.full(48, 0.1, dtype=float)


def test_data_driven_iso_requires_entire_calendar_day(tmp_path):
    actual, predicted, timestamps, _ = _files(tmp_path)
    forecasts = pd.read_csv(predicted)

    # Remove a late-day forecast outside the 48-step episode,
    # but inside the second calendar day needed for ISO pricing.
    forecasts = forecasts.loc[
        forecasts["timestamp"] != "2025-01-03 23:30:00"
    ]
    forecasts.to_csv(predicted, index=False)

    env = _env(
        actual, predicted, price_strategy=DataDrivenPrice()
    )
    try:
        with pytest.raises(
            ValueError, match="Missing forecast/calendar timestamp"
        ):
            env.reset(options={"start_date": "2025-01-02 18:00"})
    finally:
        env.close()


def test_final_forecast_entries_are_zero_padded_not_extrapolated(
    tmp_path,
):
    actual, predicted, timestamps, _ = _files(tmp_path)
    env = _env(actual, predicted)

    try:
        env.reset(options={"start_date": timestamps[48]})

        for _ in range(47):
            obs, reward, done, truncated, _ = env.step(
                np.array([0.0], dtype=np.float32)
            )

        # One interval remains.
        mask = env.encoder.field(obs, "forecast_mask")
        assert mask.sum() == pytest.approx(1.0)
        assert mask[0] == pytest.approx(1.0)
        assert np.all(mask[1:] == 0.0)
        assert np.all(env.encoder.field(obs, "load_forecast")[1:] == 0.0)
        assert np.all(env.encoder.field(obs, "generation_forecast")[1:] == 0.0)
    finally:
        env.close()

def test_price_strategy_daily_observation_contract():
    from energy_net.grid_entities.management.price_curve import (
        FixedPriceStrategy,
        SineWavePriceStrategy,
        ActionBasedPriceStrategy,
        RLPriceCurveStrategy,
    )

    # Data-dependent strategies require full-day ISO inputs.
    assert PriceCurveStrategy.requires_daily_observation is True
    assert RLPriceCurveStrategy.requires_daily_observation is True
    assert DataDrivenPrice.requires_daily_observation is True

    # These strategies generate prices without forecasting inputs.
    assert FixedPriceStrategy.requires_daily_observation is False
    assert SineWavePriceStrategy.requires_daily_observation is False
    assert ActionBasedPriceStrategy.requires_daily_observation is False