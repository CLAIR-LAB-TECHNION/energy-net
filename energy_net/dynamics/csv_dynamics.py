"""Data-backed ground truth: a strict, timestamp-indexed series."""

from typing import Optional

import numpy as np
import pandas as pd

from energy_net.foundation.dynamics import (
    GroundTruthDynamics,
    GroundTruthUnavailableError,
    timestamps_of,
)


class CSVDynamics(GroundTruthDynamics):
    """
    Ground truth read from a timestamp-indexed series (usually a CSV file).

    The lookup is strict: no interpolation, no wrap-around, and no silent
    repair of invalid readings. Invalid readings are kept as NaN so that
    ``support()`` lists every timestamp in the file and validation can
    report them. The data is copied at construction, so later changes to
    the caller's objects cannot alter the ground truth.
    """

    def __init__(self, series: pd.Series, quantity: str = "value"):
        super().__init__(quantity)
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("CSVDynamics requires a DatetimeIndex")
        name = series.name if series.name is not None else quantity
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise ValueError(f"{name} timestamps must be unique and sorted")

        # Invalid numeric values are rejected by validation rather than
        # silently corrected.
        values = pd.to_numeric(series, errors="coerce").astype(float)
        self._index = series.index.copy()
        self._values = values.to_numpy(copy=True)
        self._lookup = dict(zip(self._index, self._values))

    @classmethod
    def from_csv(cls, path: str, value_column: str,
                 timestamp_column: Optional[str] = None,
                 quantity: str = "value") -> "CSVDynamics":
        """
        Load ground truth from a CSV file.

        By default the first column is the timestamp, matching how the
        legacy PCS environment read its consumption file.
        """
        if timestamp_column is None:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            frame = pd.read_csv(path, parse_dates=[timestamp_column])
            frame = frame.set_index(timestamp_column)
        if value_column not in frame.columns:
            raise ValueError(
                f"{value_column} CSV must have a {value_column} column"
            )
        return cls(frame[value_column], quantity=quantity)

    def ground_truth(self, times) -> np.ndarray:
        stamps = timestamps_of(times)
        values = np.empty(len(stamps), dtype=float)
        for position, timestamp in enumerate(stamps):
            value = self._lookup.get(timestamp)
            if value is None:
                raise GroundTruthUnavailableError(
                    f"Missing {self.quantity} timestamp: {timestamp}"
                )
            if not np.isfinite(value):
                raise GroundTruthUnavailableError(
                    f"Non-finite {self.quantity} at {timestamp}"
                )
            values[position] = value
        return values

    def availability_error(self, times, context: Optional[str] = None) -> Optional[str]:
        label = f"{context} {self.quantity}" if context else self.quantity
        values = []
        for timestamp in timestamps_of(times):
            value = self._lookup.get(timestamp)
            if value is None:
                return f"Missing {label} timestamp: {timestamp}"
            values.append(value)
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            return f"Non-finite or negative {label}"
        return None

    def support(self) -> pd.DatetimeIndex:
        return self._index
