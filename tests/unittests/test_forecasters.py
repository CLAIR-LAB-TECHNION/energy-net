"""Tests for the built-in forecasters and entity-owned predictions."""

import numpy as np
import pandas as pd
import pytest

from energy_net.dynamics import ConstantDynamics, CSVDynamics
from energy_net.forecasting import (
    NoisyGroundTruthForecaster,
    PredictorForecaster,
    TableForecaster,
)
from energy_net.foundation.forecaster import BaseForecaster, Forecaster
from energy_net.foundation.model import State
from energy_net.foundation.noise import GaussianNoise
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.grid_entities.production.production_unit import ProductionUnit

TIMES = pd.date_range("2025-01-01", periods=6, freq="30min")


def _table(values, extra=None):
    frame = pd.DataFrame({"predicted_consumption": values}, index=TIMES[: len(values)])
    for name, column in (extra or {}).items():
        frame[name] = column
    return frame


# ------------------------------------------------------------ TableForecaster

def test_table_forecaster_is_strict_and_static():
    forecaster = TableForecaster(_table([1.0, 2.0, 3.0]), "predicted_consumption")
    np.testing.assert_array_equal(forecaster.forecast(TIMES[0], TIMES[:3]), [1, 2, 3])
    # Static forecasts: the issue time does not matter.
    np.testing.assert_array_equal(forecaster.forecast(TIMES[2], TIMES[1:3]), [2, 3])
    with pytest.raises(ValueError, match="Missing forecast/calendar timestamp"):
        forecaster.forecast(TIMES[0], pd.DatetimeIndex(["2025-01-01 00:15"]))


def test_table_forecaster_rejects_rows_with_any_invalid_column():
    table = _table([1.0, 2.0], extra={"sin_time": [0.0, np.nan]})
    forecaster = TableForecaster(table, "predicted_consumption")
    assert forecaster.availability_error(TIMES[0], TIMES[:1]) is None
    assert forecaster.availability_error(TIMES[0], TIMES[:2]) == (
        "Non-finite forecast or calendar features"
    )
    lenient = TableForecaster(table, "predicted_consumption", validate_all_columns=False)
    assert lenient.availability_error(TIMES[0], TIMES[:2]) is None


def test_table_forecaster_csv_contract(tmp_path):
    path = tmp_path / "forecast.csv"
    pd.DataFrame({
        "timestamp": [TIMES[1], TIMES[0]],
        "predicted_consumption": [2.0, 1.0],
    }).to_csv(path, index=False)
    forecaster = TableForecaster.from_csv(str(path))
    np.testing.assert_array_equal(forecaster.forecast(TIMES[0], TIMES[:2]), [1.0, 2.0])

    pd.DataFrame({
        "timestamp": [TIMES[0], TIMES[0]],
        "predicted_consumption": [2.0, 1.0],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Forecast timestamps must be unique"):
        TableForecaster.from_csv(str(path))


# -------------------------------------------------------- PredictorForecaster

class _PointPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, date_str, time_str):
        self.calls += 1
        hour, minute = map(int, time_str.split(":"))
        return hour + minute / 60.0


def test_predictor_forecaster_memoizes_point_predictions():
    predictor = _PointPredictor()
    forecaster = PredictorForecaster(predictor)
    first = forecaster.forecast(TIMES[0], TIMES)
    np.testing.assert_allclose(first, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    calls = predictor.calls
    np.testing.assert_array_equal(forecaster.forecast(TIMES[3], TIMES[2:]), first[2:])
    assert predictor.calls == calls


def test_predictor_forecaster_matches_energy_predictor(tmp_path):
    from prediction import EnergyPredictor

    stamps = pd.date_range("2025-01-01", periods=4 * 48, freq="30min")
    path = tmp_path / "history.csv"
    pd.DataFrame({
        "Datetime": stamps,
        "Production": np.clip(np.sin(np.arange(len(stamps)) / 7.6), 0, None),
    }).to_csv(path, index=False)
    predictor = EnergyPredictor(str(path), target_col="Production")

    targets = pd.date_range("2025-02-01", periods=10, freq="30min")
    vectorized = PredictorForecaster(predictor).forecast(targets[0], targets)
    pointwise = [
        predictor.predict(t.strftime("%Y-%m-%d"), t.strftime("%H:%M"))
        for t in targets
    ]
    np.testing.assert_allclose(vectorized, pointwise)


def test_predictor_forecaster_requires_a_predict_method():
    with pytest.raises(TypeError):
        PredictorForecaster(object())


# ------------------------------------------------- NoisyGroundTruthForecaster

def _dynamics():
    return CSVDynamics(pd.Series(np.arange(6.0) + 1.0, index=TIMES), quantity="generation")


def test_noiseless_forecast_equals_ground_truth():
    dynamics = _dynamics()
    forecaster = NoisyGroundTruthForecaster(dynamics)
    np.testing.assert_array_equal(
        forecaster.forecast(TIMES[0], TIMES), dynamics.ground_truth(TIMES)
    )


def test_noisy_forecast_is_consistent_across_overlapping_windows():
    forecaster = NoisyGroundTruthForecaster(_dynamics(), GaussianNoise(std=0.5))
    forecaster.reset(np.random.default_rng(11))
    full = forecaster.forecast(TIMES[0], TIMES)
    np.testing.assert_array_equal(forecaster.forecast(TIMES[0], TIMES[2:5]), full[2:5])
    np.testing.assert_array_equal(forecaster.forecast(TIMES[0], TIMES), full)
    # A different issue time is a different forecast.
    assert not np.array_equal(forecaster.forecast(TIMES[1], TIMES), full)


def test_noisy_forecast_is_reproducible_from_the_episode_rng():
    first = NoisyGroundTruthForecaster(_dynamics(), GaussianNoise(std=0.5))
    second = NoisyGroundTruthForecaster(_dynamics(), GaussianNoise(std=0.5))
    first.reset(np.random.default_rng(5))
    second.reset(np.random.default_rng(5))
    np.testing.assert_array_equal(
        first.forecast(TIMES[0], TIMES), second.forecast(TIMES[0], TIMES)
    )
    second.reset(np.random.default_rng(6))
    assert not np.array_equal(
        first.forecast(TIMES[0], TIMES), second.forecast(TIMES[0], TIMES)
    )


# ------------------------------------------------ entity-owned predictions

class _PlainForecaster:
    """Any object with forecast() works; no base class needed."""

    def forecast(self, issue_time, target_times):
        return np.full(len(target_times), 42.0)


class _LearningForecaster(BaseForecaster):
    def __init__(self):
        self.seen = []

    def forecast(self, issue_time, target_times):
        return np.zeros(len(target_times))

    def observe(self, times, realized):
        self.seen.append((times, realized))


@pytest.mark.parametrize("unit_cls, config", [
    (ConsumptionUnit, {"consumption_capacity": 10.0}),
    (ProductionUnit, {"production_capacity": 10.0}),
])
def test_units_own_ground_truth_and_prediction(unit_cls, config):
    dynamics = _dynamics()
    unit = unit_cls(dynamics, config, forecaster=_PlainForecaster())
    np.testing.assert_array_equal(unit.ground_truth(TIMES), dynamics.ground_truth(TIMES))
    np.testing.assert_array_equal(unit.predict(TIMES[0], TIMES[:2]), [42.0, 42.0])
    assert isinstance(unit.forecaster, Forecaster)

    unit.set_forecaster(_LearningForecaster())
    np.testing.assert_array_equal(unit.predict(TIMES[0], TIMES[:2]), [0.0, 0.0])


@pytest.mark.parametrize("unit_cls, config", [
    (ConsumptionUnit, {"consumption_capacity": 10.0}),
    (ProductionUnit, {"production_capacity": 10.0}),
])
def test_unit_without_forecaster_raises_clearly(unit_cls, config):
    unit = unit_cls(ConstantDynamics(1.0), config)
    with pytest.raises(RuntimeError, match="No forecaster configured"):
        unit.predict(TIMES[0], TIMES)
    with pytest.raises(TypeError):
        unit.set_forecaster(object())


def test_unit_update_realizes_ground_truth_from_state_timestamp():
    dynamics = _dynamics()
    unit = ProductionUnit(dynamics, {"production_capacity": 10.0})
    unit.update(State({"time": 0.0, "timestamp": TIMES[3]}))
    assert unit.get_state() == dynamics.ground_truth(TIMES[3:4])[0]


def test_csv_inputs_are_parsed_exactly():
    """
    Why setup.cfg requires pandas>=3: pandas 2's default float parser is one
    ulp off for ~15% of these values, so simulation inputs would silently
    depend on the installed pandas version.
    """
    import csv

    from energy_net.dynamics import CSVDynamics
    from tests.golden.scenarios import DEFAULT_FORECAST, DEFAULT_LOAD

    with open(DEFAULT_FORECAST, newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = pd.DatetimeIndex([row["timestamp"] for row in rows])
    exact = np.array([float(row["predicted_consumption"]) for row in rows])
    forecaster = TableForecaster.from_csv(DEFAULT_FORECAST)
    np.testing.assert_array_equal(forecaster.forecast(times[0], times), exact)

    with open(DEFAULT_LOAD, newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = pd.DatetimeIndex([row["Datetime"] for row in rows])
    # Empty cells are the dataset's invalid readings.
    exact = np.array([float(row["Consumption"] or "nan") for row in rows])
    dynamics = CSVDynamics.from_csv(DEFAULT_LOAD, "Consumption", quantity="consumption")
    valid = np.isfinite(exact)
    np.testing.assert_array_equal(dynamics.ground_truth(times[valid]), exact[valid])
