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


def slot_of_day(timestamps) -> np.ndarray:
    """Half-hour slot index (0..47) of each timestamp."""
    index = pd.DatetimeIndex(timestamps)
    return (index.hour * 2 + index.minute // 30).to_numpy()


def calendar_features(timestamps) -> np.ndarray:
    """
    Return an (n, 7) array of calendar features in the documented order
    ``CALENDAR_FEATURE_COLUMNS``.
    """
    index = pd.DatetimeIndex(timestamps)
    slot = slot_of_day(index)
    weekday = index.day_of_week.to_numpy()
    day_of_year = index.day_of_year.to_numpy()

    return np.column_stack((
        np.sin(2 * np.pi * slot / SLOTS_PER_DAY),
        np.cos(2 * np.pi * slot / SLOTS_PER_DAY),
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * day_of_year / 365.25),
        np.cos(2 * np.pi * day_of_year / 365.25),
        np.isin(weekday, [5, 6]).astype(float),
    ))
