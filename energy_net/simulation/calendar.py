"""Deterministic calendar features computed from absolute timestamps."""

import numpy as np
import pandas as pd

CALENDAR_FEATURE_COLUMNS = (
    "sin_time",
    "cos_time",
    "sin_week",
    "cos_week",
    "sin_year",
    "cos_year",
    "is_weekend",
)

# Daily tariffs and the time-of-day features use half-hour slots.
SLOTS_PER_DAY = 48
SLOT = pd.Timedelta(minutes=30)


def slot_index(timestamp) -> int:
    """Half-hour slot index (0..47) of a single timestamp."""
    timestamp = pd.Timestamp(timestamp)
    return timestamp.hour * 2 + timestamp.minute // 30


def slot_of_day(timestamps) -> np.ndarray:
    """Half-hour slot index (0..47) of each timestamp."""
    index = pd.DatetimeIndex(timestamps)
    return (index.hour * 2 + index.minute // 30).to_numpy()


def _calendar_fields(timestamps):
    """(slot, weekday, day_of_year) integer arrays; fast for one timestamp."""
    if isinstance(timestamps, (pd.Timestamp, str)) or not hasattr(timestamps, "__len__"):
        timestamp = pd.Timestamp(timestamps)
        return (
            np.array([slot_index(timestamp)]),
            np.array([timestamp.day_of_week]),
            np.array([timestamp.day_of_year]),
        )
    index = pd.DatetimeIndex(timestamps)
    return (
        slot_of_day(index),
        index.day_of_week.to_numpy(),
        index.day_of_year.to_numpy(),
    )


def calendar_features(timestamps) -> np.ndarray:
    """
    Return an (n, 7) array of calendar features in the documented order
    ``CALENDAR_FEATURE_COLUMNS``. A single timestamp gives shape (1, 7).
    """
    # Always evaluated on arrays (never scalar math), so a single timestamp
    # gets bit-identical values to the same timestamp inside a batch.
    slot, weekday, day_of_year = _calendar_fields(timestamps)

    return np.column_stack((
        np.sin(2 * np.pi * slot / SLOTS_PER_DAY),
        np.cos(2 * np.pi * slot / SLOTS_PER_DAY),
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * day_of_year / 365.25),
        np.cos(2 * np.pi * day_of_year / 365.25),
        np.isin(weekday, [5, 6]).astype(float),
    ))
