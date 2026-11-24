"""
The prosumer simulator: episode lifecycle, information and settlement.

This is the gym-free core. Agents interact with it only through
``reset``, ``step`` and the decision snapshot returned by ``observe``.
Ground truth is available through ``ground_truth``/``sample`` for
privileged use (analysis, oracle baselines, the ISO); it is never part of
the observation.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from energy_net.foundation.forecaster import forecast_availability_error
from energy_net.grid_entities.management.price_curve import PriceCurveStrategy
from energy_net.grid_entities.prosumer.prosumer import Prosumer
from energy_net.simulation.calendar import (
    CALENDAR_FEATURE_COLUMNS,
    SLOT,
    calendar_features,
)
from energy_net.simulation.contracts import (
    EpisodeMetrics,
    ProsumerObservation,
    StepResult,
)
from energy_net.simulation.settlement import is_grid_supplied, net_meter_money
from energy_net.simulation.tariffs import TariffSchedule

UNITS = ("load", "generator")


class ProsumerSimulator:
    """
    Simulates one prosumer in half-hour intervals.

    Timing (POMDP): at a decision point the agent knows the load and
    generation realized in the *previous* interval, forecasts for upcoming
    intervals, the current tariffs and the battery state. The battery
    command is applied before the current interval's load is realized.

    Args:
        prosumer: The physical prosumer. Its load and generator must have
            forecasters.
        tariffs: The tariff schedule.
        episode_steps: Number of intervals in an episode.
        forecast_horizon: Number of forecast positions per decision
            (defaults to ``episode_steps``).
        dt: Interval length. Only 30 minutes is supported, because daily
            tariffs and calendar features use 48 half-hour slots.
    """

    def __init__(self, prosumer: Prosumer, tariffs: TariffSchedule,
                 episode_steps: int, forecast_horizon: Optional[int] = None,
                 dt: pd.Timedelta = SLOT):
        dt = pd.Timedelta(dt)
        if dt != SLOT:
            raise ValueError(
                "dt must be 30 minutes: daily tariffs and calendar "
                "features use 48 half-hour slots"
            )
        if int(episode_steps) != episode_steps or episode_steps < 1:
            raise ValueError("episode_steps must be a positive integer")
        horizon = int(episode_steps) if forecast_horizon is None else int(forecast_horizon)
        if horizon < 0:
            raise ValueError("forecast_horizon must be nonnegative")
        for name in UNITS:
            if getattr(prosumer, name).forecaster is None:
                raise ValueError(f"The prosumer's {name} needs a forecaster")

        self.prosumer = prosumer
        self.tariffs = tariffs
        self.dt = dt
        self.episode_steps = int(episode_steps)
        self.forecast_horizon = horizon

        self.metrics = EpisodeMetrics()
        self._step_index = 0
        self._current_time = self._first_supported_time()
        self._has_episode = False
        self._snapshot: Optional[ProsumerObservation] = None
        self._previous_load: Optional[float] = None
        self._previous_generation: Optional[float] = None

    # ---------------------------------------------------------- properties

    @property
    def current_time(self) -> Optional[pd.Timestamp]:
        return self._current_time

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def has_episode(self) -> bool:
        """True once a reset has succeeded."""
        return self._has_episode

    @property
    def is_active(self) -> bool:
        """True while an episode is running and not yet terminated."""
        return self._has_episode and self._step_index < self.episode_steps

    @property
    def price_strategy(self) -> PriceCurveStrategy:
        return self.tariffs.strategy

    def _unit(self, unit: str):
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
        return getattr(self.prosumer, unit)

    def _first_supported_time(self) -> Optional[pd.Timestamp]:
        support = self._scan_domain()
        return None if support is None or len(support) == 0 else support[0]

    def _scan_domain(self) -> Optional[pd.DatetimeIndex]:
        for unit in UNITS:
            support = self._unit(unit).dynamics.support()
            if support is not None:
                return support
        return None

    def episode_times(self, start) -> pd.DatetimeIndex:
        return pd.date_range(pd.Timestamp(start), periods=self.episode_steps, freq=self.dt)

    # ---------------------------------------------------------- (b) queries

    def ground_truth(self, unit: str, times) -> np.ndarray:
        """Privileged: the values ``unit`` will realize (or realized)."""
        return self._unit(unit).ground_truth(times)

    def sample(self, unit: str, times, rng: np.random.Generator, noise) -> np.ndarray:
        """Privileged: ground truth plus noise for ``unit``."""
        return self._unit(unit).dynamics.sample(times, rng, noise)

    # ---------------------------------------------------------- validation

    def episode_error(self, start) -> Optional[str]:
        """Return None if an episode can start at ``start``, else the reason."""
        start = pd.Timestamp(start)
        times = self.episode_times(start)
        # Previous interval (observed at reset) + every realized interval.
        realized = times.insert(0, start - self.dt)

        for name in UNITS:
            error = self._unit(name).dynamics.availability_error(
                realized, "preceding/episode"
            )
            if error is not None:
                return error
        for name in UNITS:
            error = forecast_availability_error(
                self._unit(name).forecaster, start, times
            )
            if error is not None:
                return error
        return self.tariffs.availability_error(times, issue_time=start)

    def _prepare(self, start) -> Tuple[Optional[dict], Optional[str]]:
        error = self.episode_error(start)
        if error is not None:
            return None, error
        try:
            return self.tariffs.prepare(self.episode_times(start), issue_time=start), None
        except ValueError as exc:
            return None, str(exc)

    # ---------------------------------------------------------- lifecycle

    def reset(self, start=None, initial_storage: Optional[float] = None,
              rng: Optional[np.random.Generator] = None,
              rewind: bool = False) -> ProsumerObservation:
        """
        Start an episode. Nothing changes unless the episode is valid.

        Without ``start``, continue from the current time (or, with
        ``rewind``, from the start of the data); if that is not valid, use
        the first valid start in the data. No load is processed at reset:
        the observation holds the previous interval's values.
        """
        explicit = start is not None
        if explicit:
            candidate = pd.Timestamp(start)
        elif rewind:
            candidate = self._first_supported_time()
        else:
            candidate = self._current_time
        curves, error = (None, "no start time") if candidate is None else self._prepare(candidate)

        if error is not None:
            if explicit:
                raise ValueError(f"Cannot start episode at {candidate}: {error}")
            candidate, curves = None, None
            scan = self._scan_domain()
            for timestamp in scan if scan is not None else []:
                trial, trial_error = self._prepare(timestamp)
                if trial_error is None:
                    candidate, curves = timestamp, trial
                    break
            if candidate is None:
                raise ValueError("No eligible episode with complete valid data and tariffs")

        # All inputs are valid: commit.
        self._step_index = 0
        self.metrics = EpisodeMetrics()
        self._current_time = candidate
        self.prosumer.reset(initial_storage=initial_storage, rng=rng)
        previous = candidate - self.dt
        self._previous_load = float(self.prosumer.load.ground_truth(previous)[0])
        self._previous_generation = float(self.prosumer.generator.ground_truth(previous)[0])
        self.tariffs.commit(curves)
        self._has_episode = True
        self._snapshot = self._build_snapshot()
        return self._snapshot

    def observe(self) -> ProsumerObservation:
        """The decision snapshot for the current interval."""
        if self._snapshot is None:
            raise RuntimeError("No decision observation available; call reset()")
        return self._snapshot

    def step(self, command: float) -> StepResult:
        """Apply a battery command for the current interval and advance."""
        if not self._has_episode:
            raise RuntimeError("No decision observation available; call reset()")
        if self._step_index >= self.episode_steps:
            raise RuntimeError("Episode has terminated; call reset() before step()")
        command = float(command)
        if not np.isfinite(command):
            # NaN would otherwise fall through every battery branch and be
            # silently treated as "do nothing".
            raise ValueError(f"Battery command must be finite, got {command}")

        decision = self._snapshot
        interval_start = self._current_time
        buy, sell = self.tariffs.prices_at(interval_start)

        transition = self.prosumer.step(
            interval_start, command,
            time=self._step_index * (self.dt / pd.Timedelta(days=1)),
        )
        money = net_meter_money(transition.grid_exchange, buy, sell)
        grid_supplied = is_grid_supplied(transition.uncovered_load)
        self.metrics = self.metrics.add(transition, money, grid_supplied)

        # The interval just realized becomes the previous one.
        self._previous_load = transition.load
        self._previous_generation = transition.generation
        step_index = self._step_index
        self._step_index += 1
        self._current_time = interval_start + self.dt
        terminated = self._step_index >= self.episode_steps
        self._snapshot = None if terminated else self._build_snapshot()

        return StepResult(
            step_index=step_index,
            max_steps=self.episode_steps,
            interval_start=interval_start,
            interval_end=self._current_time,
            decision=decision,
            transition=transition,
            buy_tariff=buy,
            sell_tariff=sell,
            money=money,
            grid_supplied=grid_supplied,
            terminated=terminated,
            metrics=self.metrics,
            price_strategy=type(self.tariffs.strategy).__name__,
            next_observation=self._snapshot,
        )

    def set_price_strategy(self, strategy: PriceCurveStrategy) -> bool:
        """
        Atomically replace the pricing strategy.

        During an episode the remaining days are re-priced and validated
        first; a failure leaves everything unchanged. Returns True if an
        episode is active, in which case the decision snapshot was
        refreshed and must replace the previous one.
        """
        if not isinstance(strategy, PriceCurveStrategy):
            raise TypeError("The provided strategy must be a PriceCurveStrategy")
        active = self.is_active
        curves = {}
        if active:
            remaining = pd.date_range(
                self._current_time,
                periods=self.episode_steps - self._step_index,
                freq=self.dt,
            )
            error = self.tariffs.availability_error(
                remaining, self._current_time, strategy
            )
            if error is not None:
                raise ValueError(error)
            curves = self.tariffs.prepare(remaining, strategy, self._current_time)
        self.tariffs.commit(curves, strategy)
        if active:
            self._snapshot = self._build_snapshot()
        return active

    # ---------------------------------------------------------- snapshot

    def _build_snapshot(self) -> ProsumerObservation:
        now = self._current_time
        steps_remaining = self.episode_steps - self._step_index
        valid = min(self.forecast_horizon, steps_remaining)
        targets = pd.date_range(now, periods=valid, freq=self.dt)
        if valid > 0:
            load_forecast = self.prosumer.load.predict(now, targets)
            generation_forecast = self.prosumer.generator.predict(now, targets)
        else:
            load_forecast = generation_forecast = np.zeros(0)
        buy, sell = self.tariffs.prices_at(now)

        return ProsumerObservation(
            timestamp=now,
            step_index=self._step_index,
            steps_remaining=steps_remaining,
            forecast_horizon=self.forecast_horizon,
            storage=self.prosumer.storage_level(),
            storage_capacity=self.prosumer.storage.energy_max,
            available_charge=self.prosumer.available_charge(),
            available_discharge=self.prosumer.available_discharge(),
            previous_load=self._previous_load,
            previous_generation=self._previous_generation,
            buy_tariff=buy,
            sell_tariff=sell,
            load_forecast=load_forecast,
            generation_forecast=generation_forecast,
            calendar=calendar_features(now)[0],
            calendar_columns=CALENDAR_FEATURE_COLUMNS,
        )
