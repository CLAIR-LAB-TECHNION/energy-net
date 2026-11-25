"""Missing CSV readings are refused, never interpolated or wrapped."""

import numpy as np
import pandas as pd
import pytest

from energy_net.dynamics import CSVDynamics
from energy_net.foundation.dynamics import GroundTruthUnavailableError
from energy_net.gym_envs.prosumer_env import ProsumerEnv


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
def test_missing_interior_consumption_is_refused(missing_consumption_files, omit_timestamp):
    data_file, _, readings = missing_consumption_files
    if omit_timestamp:
        readings.drop(index=6).to_csv(data_file, index=False)
    dynamics = CSVDynamics.from_csv(str(data_file), "Consumption", quantity="consumption")

    with pytest.raises(GroundTruthUnavailableError):
        dynamics.ground_truth(pd.Timestamp("2025-01-01 03:00"))
    assert dynamics.ground_truth(pd.Timestamp("2025-01-01 03:30"))[0] == 4.0


def test_prosumer_rejects_episode_with_raw_missing_demand(
    missing_consumption_files,
):
    data_file, predictions_file, readings = missing_consumption_files

    env = ProsumerEnv(
        load_data_file=str(data_file),
        load_forecast_file=str(predictions_file),
        verbosity=0,
    )

    try:
        # The strict baseline must not admit an episode containing a
        # missing raw demand measurement.
        with pytest.raises(
            ValueError,
            match="Non-finite or negative.*consumption",
        ):
            env.reset(options={"start_date": readings.loc[1, "Datetime"]})

        # Automatic reset skips the invalid region.
        obs, _ = env.reset()

        assert env.current_datetime == pd.Timestamp(
            readings.loc[8, "Datetime"]
        )
        assert np.isfinite(obs).all()

        for _ in range(env.max_steps):
            obs, reward, done, _, info = env.step(
                np.array([0.0], dtype=np.float32)
            )

            assert np.isfinite(reward)
            assert np.isfinite(info["consumption_units"])

        assert done
    finally:
        env.close()


@pytest.mark.parametrize(
    "values",
    [
        [2.0, np.nan, np.nan, 8.0],
        [np.nan, np.nan, 4.0, 6.0],
        [2.0, 4.0, np.nan, np.nan],
        [2.0, np.inf, -np.inf, 8.0],
    ],
    ids=["consecutive-gap", "leading-gap", "trailing-gap", "infinite-readings"],
)
def test_invalid_readings_are_kept_and_reported_not_repaired(tmp_path, values):
    data_file = tmp_path / "consumption.csv"
    times = pd.date_range("2025-01-01", periods=len(values), freq="30min")
    pd.DataFrame({"Datetime": times, "Consumption": values}).to_csv(data_file, index=False)

    dynamics = CSVDynamics.from_csv(str(data_file), "Consumption", quantity="consumption")

    assert len(dynamics.support()) == len(values)
    assert dynamics.availability_error(times) == "Non-finite or negative consumption"
    for timestamp, value in zip(times, values):
        if np.isfinite(value):
            assert dynamics.ground_truth(timestamp)[0] == value
        else:
            with pytest.raises(GroundTruthUnavailableError):
                dynamics.ground_truth(timestamp)


def test_completely_missing_data_offers_no_episode(tmp_path):
    data_file = tmp_path / "consumption.csv"
    times = pd.date_range("2025-01-01", periods=3, freq="30min")
    pd.DataFrame({
        "Datetime": times,
        "Consumption": [np.nan, np.inf, -np.inf],
    }).to_csv(data_file, index=False)

    dynamics = CSVDynamics.from_csv(str(data_file), "Consumption", quantity="consumption")
    assert dynamics.availability_error(times) is not None
