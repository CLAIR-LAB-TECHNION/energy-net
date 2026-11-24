"""Immutable records exchanged between the simulator and its consumers."""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from energy_net.grid_entities.prosumer.prosumer_classes import ProsumerTransition
from energy_net.simulation.calendar import CALENDAR_FEATURE_COLUMNS


def _frozen(array) -> np.ndarray:
    array = np.array(array, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ProsumerObservation:
    """
    Everything an agent may know at a decision point, by name.

    Forecasts cover the next ``min(forecast_horizon, steps_remaining)``
    intervals and are fetched once per decision, so every consumer of this
    snapshot sees the same prediction.
    """

    timestamp: pd.Timestamp
    step_index: int
    steps_remaining: int
    forecast_horizon: int
    storage: float
    storage_capacity: float
    available_charge: float
    available_discharge: float
    previous_load: float
    previous_generation: float
    buy_tariff: float
    sell_tariff: float
    load_forecast: np.ndarray
    generation_forecast: np.ndarray
    calendar: np.ndarray
    calendar_columns: Tuple[str, ...] = CALENDAR_FEATURE_COLUMNS

    def __post_init__(self):
        for name in ("load_forecast", "generation_forecast", "calendar"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        if len(self.load_forecast) != len(self.generation_forecast):
            raise ValueError("load and generation forecasts must have equal length")

    def calendar_dict(self) -> Dict[str, float]:
        return {name: float(value) for name, value in zip(self.calendar_columns, self.calendar)}


@dataclass(frozen=True)
class EpisodeMetrics:
    """Physical and financial totals of the current episode."""

    total_money: float = 0.0
    grid_supplied_count: int = 0
    total_grid_import: float = 0.0
    total_grid_export: float = 0.0
    total_consumption: float = 0.0
    total_battery_served: float = 0.0
    total_grid_supplied_demand: float = 0.0
    total_generation: float = 0.0
    total_exported_surplus: float = 0.0

    def add(self, transition: ProsumerTransition, money: float,
            grid_supplied: bool) -> "EpisodeMetrics":
        return replace(
            self,
            total_money=self.total_money + money,
            grid_supplied_count=self.grid_supplied_count + int(grid_supplied),
            total_grid_import=self.total_grid_import + transition.grid_import,
            total_grid_export=self.total_grid_export + transition.grid_export,
            total_consumption=self.total_consumption + transition.load,
            total_battery_served=self.total_battery_served + transition.battery_to_load,
            total_grid_supplied_demand=(
                self.total_grid_supplied_demand + transition.uncovered_load
            ),
            total_generation=self.total_generation + transition.generation,
            total_exported_surplus=(
                self.total_exported_surplus + transition.exported_surplus
            ),
        )


@dataclass(frozen=True)
class StepResult:
    """The complete, immutable outcome of one simulator step."""

    step_index: int
    max_steps: int
    interval_start: pd.Timestamp
    interval_end: pd.Timestamp
    decision: ProsumerObservation
    transition: ProsumerTransition
    buy_tariff: float
    sell_tariff: float
    money: float
    grid_supplied: bool
    terminated: bool
    metrics: EpisodeMetrics
    price_strategy: str
    next_observation: Optional[ProsumerObservation] = field(default=None)

    def state_after(self) -> Dict[str, Any]:
        t = self.transition
        return {
            "storage_units": float(t.storage_after),
            "previous_load_units": float(t.load),
            "previous_generation_units": float(t.generation),
            "datetime": self.interval_end.isoformat(),
            "episode_step": int(self.step_index + 1),
            "available_charge_units": float(t.available_charge),
            "available_discharge_units": float(t.available_discharge),
        }

    def to_info(self) -> Dict[str, Any]:
        """Physical and financial diagnostics, keyed as in the legacy env."""
        t = self.transition
        m = self.metrics
        decision = self.decision
        state_after = self.state_after()
        current_price = self.buy_tariff if t.grid_exchange >= 0 else self.sell_tariff
        first = lambda values: float(values[0]) if len(values) else None  # noqa: E731

        return {
            # Physical and financial decomposition of the interval.
            "storage_before_units": t.storage_before,
            "storage_after_command_units": t.storage_after_command,
            "storage_after_units": t.storage_after,
            "battery_action": t.requested_command,
            "energy_sold_or_bought_units": t.applied_command,
            "consumption_units": t.load,
            "generation_units": t.generation,
            "battery_energy_used_for_consumption_units": t.battery_to_load,
            "battery_charged_from_surplus_units": t.surplus_to_storage,
            "exported_surplus_units": t.exported_surplus,
            "uncovered_consumption_units": t.uncovered_load,
            "available_charge_units": t.available_charge,
            "available_discharge_units": t.available_discharge,
            "grid_exchange_units": t.grid_exchange,
            "grid_import_units": t.grid_import,
            "grid_export_units": t.grid_export,
            "shortage": self.grid_supplied,
            "grid_supplied_demand_event": self.grid_supplied,
            "buy_tariff": self.buy_tariff,
            "sell_tariff": self.sell_tariff,
            "current_price": current_price,
            "step_money": self.money,
            "electricity_cost": -self.money,
            # Episode context and information at decision time.
            "step_index": self.step_index,
            "max_steps": self.max_steps,
            "interval_start": self.interval_start.isoformat(),
            "interval_end": self.interval_end.isoformat(),
            "price_strategy": self.price_strategy,
            "previous_load_at_decision_units": decision.previous_load,
            "previous_generation_at_decision_units": decision.previous_generation,
            "forecast_at_decision_units": first(decision.load_forecast),
            "generation_forecast_at_decision_units": first(decision.generation_forecast),
            "calendar_features_at_decision": decision.calendar_dict(),
            # Cumulative episode statistics.
            "total_money_so_far": m.total_money,
            "shortage_count_so_far": m.grid_supplied_count,
            "total_grid_import_units": m.total_grid_import,
            "total_grid_export_units": m.total_grid_export,
            "total_consumption_units": m.total_consumption,
            "total_generation_units": m.total_generation,
            "total_exported_surplus_units": m.total_exported_surplus,
            "total_battery_served_units": m.total_battery_served,
            "total_grid_supplied_demand_units": m.total_grid_supplied_demand,
            # Resulting state, also available at termination.
            "state_after": state_after,
            "terminal_state": dict(state_after) if self.terminated else None,
        }
