"""Built-in forecasters. Any object with ``forecast(...)`` also works."""

from typing import Optional

import numpy as np
import pandas as pd

from energy_net.foundation.dynamics import as_datetime_index
from energy_net.foundation.forecaster import BaseForecaster
from energy_net.foundation.noise import NoiseModel, NoNoise

_UINT64 = (1 << 64) - 1


class TableForecaster(BaseForecaster):
    """
    Precomputed forecasts, looked up strictly by target timestamp.

    Forecasts are static: the same value is returned for a target time
    regardless of the issue time. There is no interpolation. By default a
    target is usable only if *every* numeric column of its row is finite,
    which preserves the legacy PCS rule that also rejected rows whose
    calendar-feature columns were invalid.
    """

    def __init__(self, table: pd.DataFrame, value_column: str,
                 validate_all_columns: bool = True):
        if not isinstance(table.index, pd.DatetimeIndex):
            raise TypeError("TableForecaster requires a DatetimeIndex")
        if value_column not in table.columns:
            raise ValueError(f"Forecast CSV must have a {value_column} column")
        if table.index.has_duplicates:
            raise ValueError("Forecast timestamps must be unique")

        table = table.sort_index().apply(pd.to_numeric, errors="coerce")
        checked = table if validate_all_columns else table[[value_column]]
        row_ok = np.isfinite(checked.to_numpy(dtype=float)).all(axis=1)

        self.value_column = value_column
        self._index = table.index.copy()
        self._values = dict(zip(self._index, table[value_column].to_numpy(dtype=float)))
        self._valid = dict(zip(self._index, row_ok))

    @classmethod
    def from_csv(cls, path: str, value_column: str = "predicted_consumption",
                 timestamp_column: str = "timestamp",
                 validate_all_columns: bool = True) -> "TableForecaster":
        frame = pd.read_csv(path, parse_dates=[timestamp_column])
        frame = frame.set_index(timestamp_column)
        return cls(frame, value_column, validate_all_columns)

    def __len__(self) -> int:
        return len(self._index)

    def availability_error(self, issue_time, target_times) -> Optional[str]:
        targets = as_datetime_index(target_times)
        for timestamp in targets:
            if timestamp not in self._values:
                return f"Missing forecast/calendar timestamp: {timestamp}"
        if not all(self._valid[timestamp] for timestamp in targets):
            return "Non-finite forecast or calendar features"
        return None

    def forecast(self, issue_time, target_times) -> np.ndarray:
        targets = as_datetime_index(target_times)
        error = self.availability_error(issue_time, targets)
        if error is not None:
            raise ValueError(error)
        return np.array([self._values[t] for t in targets], dtype=float)


class PredictorForecaster(BaseForecaster):
    """
    Wraps a trained predictor, such as ``prediction.EnergyPredictor``.

    The predictor needs ``predict_timestamps(timestamps)`` (vectorized) or
    ``predict(date_str, time_str)``. Predictions are memoized per target
    timestamp, so repeated queries are cheap and always consistent.
    """

    def __init__(self, predictor):
        has_vectorized = callable(getattr(predictor, "predict_timestamps", None))
        if not has_vectorized and not callable(getattr(predictor, "predict", None)):
            raise TypeError(
                "predictor must provide predict_timestamps(timestamps) "
                "or predict(date_str, time_str)"
            )
        self.predictor = predictor
        self._vectorized = has_vectorized
        self._cache = {}

    def _fill(self, targets: pd.DatetimeIndex) -> None:
        missing = [t for t in targets if t not in self._cache]
        if not missing:
            return
        if self._vectorized:
            values = self.predictor.predict_timestamps(pd.DatetimeIndex(missing))
        else:
            values = [
                self.predictor.predict(t.strftime("%Y-%m-%d"), t.strftime("%H:%M"))
                for t in missing
            ]
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(missing):
            raise ValueError("predictor returned the wrong number of values")
        self._cache.update(zip(missing, values))

    def availability_error(self, issue_time, target_times) -> Optional[str]:
        try:
            values = self.forecast(issue_time, target_times)
        except Exception as exc:  # report, do not crash validation
            return f"Forecast unavailable: {exc}"
        if not np.isfinite(values).all():
            return "Non-finite forecast"
        return None

    def forecast(self, issue_time, target_times) -> np.ndarray:
        targets = as_datetime_index(target_times)
        self._fill(targets)
        return np.array([self._cache[t] for t in targets], dtype=float)


class NoisyGroundTruthForecaster(BaseForecaster):
    """
    Forecast = ground truth + noise (role 2 of the dynamics).

    This forecaster is privileged by construction: it holds the dynamics.
    Noise for each (episode seed, issue time, target time) uses its own
    seed, so overlapping windows and repeated queries agree exactly, and an
    episode is reproducible from the env's seed.
    """

    def __init__(self, dynamics, noise: Optional[NoiseModel] = None,
                 seed: int = 0):
        if not callable(getattr(dynamics, "sample", None)):
            raise TypeError("dynamics must provide sample(times, rng, noise)")
        self.dynamics = dynamics
        self.noise = noise if noise is not None else NoNoise()
        self._episode_seed = int(seed)

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self._episode_seed = int(rng.integers(0, 2 ** 63 - 1))

    def availability_error(self, issue_time, target_times) -> Optional[str]:
        return self.dynamics.availability_error(target_times)

    def forecast(self, issue_time, target_times) -> np.ndarray:
        targets = as_datetime_index(target_times)
        if isinstance(self.noise, NoNoise):
            return self.dynamics.ground_truth(targets)

        issue = pd.Timestamp(issue_time)
        values = np.empty(len(targets), dtype=float)
        for position, target in enumerate(targets):
            rng = np.random.default_rng([
                self._episode_seed,
                issue.value & _UINT64,
                target.value & _UINT64,
            ])
            lead = np.array([(target - issue) / pd.Timedelta(hours=1)])
            values[position] = self.dynamics.sample(
                targets[position:position + 1], rng, self.noise, lead
            )[0]
        return values
