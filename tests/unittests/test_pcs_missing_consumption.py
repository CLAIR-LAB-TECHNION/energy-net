"""Regression expectations for missing CSV readings, not missing forecasts."""

import numpy as np
import pandas as pd
import pytest

from energy_net.grid_entities.consumption.consumption_dynamics import (
    CSV_DataConsumptionDynamics,
)
from energy_net.gym_envs.pcs_env import PCSEnv


@pytest.fixture
def missing_consumption_files(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=3 * 48, freq="30min")
    consumption = np.full(len(timestamps), 2.0)
    consumption[6:8] = [3.0, 4.0]
    data_file = tmp_path / "consumption.csv"
    predictions_file = tmp_path / "predictions.csv"
    readings = pd.DataFrame({
        "Datetime": timestamps,
        "Consumption": consumption,
    })
    # Keep forecasts finite to isolate the physical consumption input.
    pd.DataFrame({
        "timestamp": timestamps,
        "predicted_consumption": consumption,
    }).to_csv(predictions_file, index=False)
    readings.loc[6, "Consumption"] = np.nan
    readings.to_csv(data_file, index=False)
    return data_file, predictions_file, readings


@pytest.mark.parametrize("omit_timestamp", [False, True], ids=["existing-nan", "absent-key"])
def test_missing_interior_consumption_is_interpolated(
    missing_consumption_files, omit_timestamp
):
    data_file, _, readings = missing_consumption_files
    if omit_timestamp:
        readings.drop(index=6).to_csv(data_file, index=False)
    dynamics = CSV_DataConsumptionDynamics({"data_file": str(data_file)})

    # 03:00 is exactly representable, so timestamp rounding cannot select a branch.
    value = dynamics.get_value(time=0.125)

    assert value == pytest.approx(3.0), (
        "An interior missing reading must be interpolated just like an absent key"
    )


def test_missing_consumption_does_not_reach_step_observation(missing_consumption_files):
    data_file, predictions_file, _ = missing_consumption_files
    env = PCSEnv(
        test_data_file=str(data_file),
        predictions_file=str(predictions_file),
        verbosity=0,
    )
    try:
        obs, _ = env.reset()
        assert np.isfinite(obs).all()
        for _ in range(6):
            obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
            assert np.isfinite(obs).all()

        obs, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))

        assert np.isfinite(obs[4:]).all(), "Forecasts and capacities are finite controls"
        np.testing.assert_allclose(
            [
                env.pcs.consumption_units[0].get_state(),
                env.pcs.get_consumption(),
                info["consumption_units"],
                obs[1] * 10.0,
            ],
            [3.0] * 4,
            err_msg="Missing CSV demand must not propagate through the unit into observations",
        )
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
    finally:
        env.close()


@pytest.mark.parametrize(
    "values, expected",
    [
        ([2.0, np.nan, np.nan, 8.0], [2.0, 4.0, 6.0, 8.0]),
        ([np.nan, np.nan, 4.0, 6.0], [4.0, 4.0, 4.0, 6.0]),
        ([2.0, 4.0, np.nan, np.nan], [2.0, 4.0, 4.0, 4.0]),
        ([np.nan, 3.0, np.nan], [3.0, 3.0, 3.0]),
        ([2.0, np.inf, -np.inf, 8.0], [2.0, 4.0, 6.0, 8.0]),
        ([0.0, np.nan, 4.0], [0.0, 2.0, 4.0]),
    ],
    ids=["consecutive-gap", "leading-gap", "trailing-gap", "single-valid",
         "infinite-readings", "valid-zero"],
)
def test_missing_readings_are_cleaned_at_load_time(tmp_path, values, expected):
    data_file = tmp_path / "consumption.csv"
    pd.DataFrame({
        "Datetime": pd.date_range("2025-01-01", periods=len(values), freq="30min"),
        "Consumption": values,
    }).to_csv(data_file, index=False)

    dynamics = CSV_DataConsumptionDynamics({"data_file": str(data_file)})

    np.testing.assert_allclose(dynamics.consumption_data["Consumption"], expected)
    for step, value in enumerate(expected):
        assert dynamics.get_value(time=step / 48) == pytest.approx(value)


def test_interpolation_is_weighted_by_elapsed_time(tmp_path):
    data_file = tmp_path / "consumption.csv"
    pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 02:00",
        ]),
        "Consumption": [2.0, np.nan, 10.0],
    }).to_csv(data_file, index=False)

    dynamics = CSV_DataConsumptionDynamics({"data_file": str(data_file)})

    assert dynamics.get_value(time=0.5 / 24) == pytest.approx(4.0)


def test_completely_missing_data_cannot_be_interpolated(tmp_path):
    data_file = tmp_path / "consumption.csv"
    pd.DataFrame({
        "Datetime": pd.date_range("2025-01-01", periods=3, freq="30min"),
        "Consumption": [np.nan, np.inf, -np.inf],
    }).to_csv(data_file, index=False)

    with pytest.raises(ValueError, match="at least one finite reading"):
        CSV_DataConsumptionDynamics({"data_file": str(data_file)})
