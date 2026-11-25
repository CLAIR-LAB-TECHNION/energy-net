"""Contract tests for ground-truth dynamics (role 1) and noisy sampling (role 2)."""

import numpy as np
import pandas as pd
import pytest

from energy_net.dynamics import ConstantDynamics, CSVDynamics, ProfileDynamics
from energy_net.foundation.dynamics import GroundTruthUnavailableError
from energy_net.foundation.noise import GaussianNoise, NoNoise
from energy_net.grid_entities.production.production_dynamics import (
    GMMProductionDynamics,
)


def _series(values, start="2025-01-01", name="Consumption"):
    index = pd.date_range(start, periods=len(values), freq="30min")
    return pd.Series(values, index=index, name=name)


def _csv(tmp_path, values, start="2025-01-01"):
    path = tmp_path / "load.csv"
    pd.DataFrame({
        "Datetime": pd.date_range(start, periods=len(values), freq="30min"),
        "Consumption": values,
    }).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- CSVDynamics

def test_csv_ground_truth_is_exact_lookup(tmp_path):
    dynamics = CSVDynamics.from_csv(
        _csv(tmp_path, [1.0, 2.5, 4.0]), "Consumption", quantity="consumption"
    )
    times = pd.date_range("2025-01-01", periods=3, freq="30min")
    np.testing.assert_array_equal(dynamics.ground_truth(times), [1.0, 2.5, 4.0])
    assert dynamics.get_value(timestamp=times[1]) == 2.5


def test_csv_ground_truth_is_idempotent_and_order_independent():
    dynamics = CSVDynamics(_series(np.arange(10.0)), quantity="consumption")
    times = pd.date_range("2025-01-01", periods=10, freq="30min")
    forward = dynamics.ground_truth(times)
    backward = dynamics.ground_truth(times[::-1])[::-1]
    np.testing.assert_array_equal(forward, backward)
    np.testing.assert_array_equal(forward, dynamics.ground_truth(times))


def test_csv_does_not_interpolate_between_readings():
    dynamics = CSVDynamics(_series([1.0, 3.0]), quantity="consumption")
    with pytest.raises(GroundTruthUnavailableError, match="Missing consumption"):
        dynamics.ground_truth(pd.Timestamp("2025-01-01 00:15"))


def test_csv_does_not_wrap_past_the_end_of_data():
    dynamics = CSVDynamics(_series([1.0, 3.0]), quantity="consumption")
    with pytest.raises(GroundTruthUnavailableError):
        dynamics.ground_truth(pd.Timestamp("2025-01-02 00:00"))


def test_csv_keeps_invalid_rows_and_refuses_them(tmp_path):
    dynamics = CSVDynamics.from_csv(
        _csv(tmp_path, [1.0, np.nan, 3.0]), "Consumption",
        quantity="consumption",
    )
    # Invalid readings stay in the support, so scans see every timestamp.
    assert len(dynamics.support()) == 3
    with pytest.raises(GroundTruthUnavailableError, match="Non-finite"):
        dynamics.ground_truth(pd.Timestamp("2025-01-01 00:30"))


def test_csv_availability_messages_match_legacy_wording():
    dynamics = CSVDynamics(_series([1.0, -2.0]), quantity="consumption")
    times = pd.date_range("2025-01-01", periods=2, freq="30min")
    assert dynamics.availability_error(times[:1], "preceding/episode") is None
    assert dynamics.availability_error(times, "preceding/episode") == (
        "Non-finite or negative preceding/episode consumption"
    )
    missing = pd.DatetimeIndex(["2024-12-31 23:30"])
    assert dynamics.availability_error(missing, "preceding/episode") == (
        "Missing preceding/episode consumption timestamp: 2024-12-31 23:30:00"
    )


def test_csv_rejects_unsorted_or_duplicate_timestamps():
    index = pd.DatetimeIndex(["2025-01-01 00:30", "2025-01-01 00:00"])
    with pytest.raises(ValueError, match="Consumption timestamps must be unique and sorted"):
        CSVDynamics(pd.Series([1.0, 2.0], index=index, name="Consumption"))


def test_csv_requires_value_column(tmp_path):
    with pytest.raises(ValueError, match="Production CSV must have a Production column"):
        CSVDynamics.from_csv(_csv(tmp_path, [1.0]), "Production")


def test_csv_copies_data_so_callers_cannot_corrupt_it():
    series = _series([1.0, 2.0])
    dynamics = CSVDynamics(series, quantity="consumption")
    series.iloc[0] = np.nan
    assert dynamics.ground_truth(series.index[:1])[0] == 1.0


# ------------------------------------------------------ synthetic dynamics

def test_profile_dynamics_uses_fraction_of_day():
    profile = GMMProductionDynamics({})
    dynamics = ProfileDynamics(profile, quantity="generation")
    times = pd.DatetimeIndex(["2025-03-01 00:00", "2025-03-01 12:00", "2025-03-02 18:00"])
    expected = [profile.get_value(time=f) for f in (0.0, 0.5, 0.75)]
    np.testing.assert_allclose(dynamics.ground_truth(times), expected)
    assert dynamics.support() is None


def test_constant_dynamics_is_always_available():
    dynamics = ConstantDynamics(0.0, quantity="generation")
    times = pd.date_range("1990-01-01", periods=5, freq="30min")
    np.testing.assert_array_equal(dynamics.ground_truth(times), np.zeros(5))
    assert dynamics.availability_error(times) is None


# ------------------------------------------------------ role 2: sampling

def test_sample_without_noise_equals_ground_truth():
    dynamics = CSVDynamics(_series([1.0, 2.0, 3.0]))
    times = dynamics.support()
    sampled = dynamics.sample(times, np.random.default_rng(0), NoNoise())
    np.testing.assert_array_equal(sampled, dynamics.ground_truth(times))


def test_gaussian_sampling_is_reproducible_and_clipped():
    dynamics = CSVDynamics(_series([0.0, 0.0, 5.0, 5.0]))
    times = dynamics.support()
    noise = GaussianNoise(std=2.0)
    first = dynamics.sample(times, np.random.default_rng(7), noise)
    second = dynamics.sample(times, np.random.default_rng(7), noise)
    np.testing.assert_array_equal(first, second)
    assert (first >= 0.0).all()
    assert not np.array_equal(first, dynamics.ground_truth(times))


def test_relative_gaussian_noise_scales_with_value():
    noise = GaussianNoise(std=0.1, relative=True, clip_min=None)
    values = np.array([0.0, 100.0])
    noisy = noise.apply(values, np.random.default_rng(3))
    assert noisy[0] == 0.0
    assert noisy[1] != 100.0


def test_gaussian_noise_rejects_negative_std():
    with pytest.raises(ValueError):
        GaussianNoise(std=-1.0)
