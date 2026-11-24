"""Daily prosumer tariffs produced by a PriceCurveStrategy."""

from typing import Dict, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from energy_net.foundation.forecaster import forecast_availability_error
from energy_net.grid_entities.management.price_curve import PriceCurveStrategy
from energy_net.simulation.calendar import (
    SLOT,
    SLOTS_PER_DAY,
    calendar_features,
    slot_index,
)

Curves = Tuple[np.ndarray, np.ndarray]


class DailyObservationSource(Protocol):
    """Builds the observation a data-driven strategy prices a day from."""

    def observe(self, day: pd.Timestamp, issue_time: pd.Timestamp) -> np.ndarray:
        ...

    def availability_error(self, day: pd.Timestamp,
                           issue_time: pd.Timestamp) -> Optional[str]:
        ...


def day_slots(day) -> pd.DatetimeIndex:
    """The 48 half-hour timestamps of the calendar day containing ``day``."""
    return pd.date_range(pd.Timestamp(day).normalize(), periods=SLOTS_PER_DAY, freq=SLOT)


class ForecastDailyObservation:
    """
    Daily observation = load forecast for the day's 48 slots, divided by
    ``demand_scale``, followed by the flattened calendar features.

    ``load_unit`` is any entity with ``predict`` and ``forecaster`` (the
    prosumer's load).
    """

    def __init__(self, load_unit, demand_scale: float = 10.0):
        self.load_unit = load_unit
        self.demand_scale = float(demand_scale)

    def availability_error(self, day, issue_time) -> Optional[str]:
        return forecast_availability_error(
            self.load_unit.forecaster, issue_time, day_slots(day)
        )

    def observe(self, day, issue_time) -> np.ndarray:
        times = day_slots(day)
        demand = self.load_unit.predict(issue_time, times)
        return np.concatenate([
            demand / self.demand_scale,
            calendar_features(times).ravel(),
        ])


class TariffSchedule:
    """
    Validated, per-calendar-day prosumer import (buy) and export (sell)
    tariffs.

    Curves are prepared and validated before they are committed, so a
    failed preparation never changes the schedule.
    """

    def __init__(self, strategy: PriceCurveStrategy,
                 daily_observation: Optional[DailyObservationSource] = None):
        self._validate_strategy_type(strategy)
        self._strategy = strategy
        self.daily_observation = daily_observation
        self._curves: Dict[object, Curves] = {}

    @staticmethod
    def _validate_strategy_type(strategy) -> None:
        if not isinstance(strategy, PriceCurveStrategy):
            raise TypeError("The provided strategy must be a PriceCurveStrategy")

    @property
    def strategy(self) -> PriceCurveStrategy:
        return self._strategy

    def needs_daily_observation(self, strategy=None) -> bool:
        strategy = self._strategy if strategy is None else strategy
        return bool(strategy.requires_daily_observation)

    # ------------------------------------------------------------ building

    def build_day(self, day, strategy: Optional[PriceCurveStrategy] = None,
                  issue_time=None) -> Curves:
        """Compute and validate one day's curves without touching the cache."""
        strategy = self._strategy if strategy is None else strategy
        day = pd.Timestamp(day).normalize()
        issue_time = day if issue_time is None else pd.Timestamp(issue_time)

        observation = None
        if self.needs_daily_observation(strategy):
            if self.daily_observation is None:
                raise ValueError(
                    "This pricing strategy needs a daily observation, but "
                    "no DailyObservationSource is configured"
                )
            observation = self.daily_observation.observe(day, issue_time)

        buy, sell = strategy.calculate_prosumer_tariffs(observation)
        buy = np.asarray(buy, dtype=float)
        sell = np.asarray(sell, dtype=float)

        if buy.shape != (SLOTS_PER_DAY,) or sell.shape != (SLOTS_PER_DAY,):
            raise ValueError(
                "ISO pricing strategy must return two 48-slot daily curves"
            )
        if not np.isfinite(buy).all() or not np.isfinite(sell).all():
            raise ValueError("Tariffs must be finite")
        if (buy < 0).any() or (sell < 0).any():
            raise ValueError("Tariffs must be nonnegative")
        # Exporting must not pay more than importing costs.
        if (sell > buy + 1e-7).any():
            raise ValueError("Prosumer sell tariff cannot exceed prosumer buy tariff")
        upper = getattr(strategy, "price_max", None)
        if upper is not None and ((buy > upper + 1e-7).any() or (sell > upper + 1e-7).any()):
            raise ValueError("Tariffs exceed the strategy price_max")
        return buy.copy(), sell.copy()

    def prepare(self, times: pd.DatetimeIndex,
                strategy: Optional[PriceCurveStrategy] = None,
                issue_time=None) -> Dict[object, Curves]:
        """Validated curves for every calendar day touched by ``times``."""
        return {
            day.date(): self.build_day(day, strategy, issue_time)
            for day in pd.DatetimeIndex(times).normalize().unique()
        }

    def availability_error(self, times: pd.DatetimeIndex, issue_time,
                           strategy: Optional[PriceCurveStrategy] = None) -> Optional[str]:
        """Check the inputs a data-driven strategy needs for these times."""
        if not self.needs_daily_observation(strategy):
            return None
        if self.daily_observation is None:
            return "No daily observation source for a data-driven pricing strategy"
        for day in pd.DatetimeIndex(times).normalize().unique():
            error = self.daily_observation.availability_error(day, issue_time)
            if error is not None:
                return error
        return None

    # ------------------------------------------------------------ state

    def commit(self, curves: Dict[object, Curves],
               strategy: Optional[PriceCurveStrategy] = None) -> None:
        """Replace the cached curves (and the strategy, if given)."""
        if strategy is not None:
            self._validate_strategy_type(strategy)
            self._strategy = strategy
        self._curves = dict(curves)

    def cached_days(self):
        """Calendar dates with committed curves."""
        return set(self._curves)

    def prices_at(self, timestamp) -> Tuple[float, float]:
        """(buy, sell) tariffs for the half-hour slot containing ``timestamp``."""
        timestamp = pd.Timestamp(timestamp)
        day = timestamp.date()
        if day not in self._curves:
            self._curves[day] = self.build_day(timestamp, issue_time=timestamp)
        buy, sell = self._curves[day]
        slot = slot_index(timestamp)
        return float(buy[slot]), float(sell[slot])
