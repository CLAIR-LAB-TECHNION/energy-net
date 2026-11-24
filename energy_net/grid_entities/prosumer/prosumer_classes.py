"""Immutable value types describing a prosumer and one transition of it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryLimits:
    """Physical limits of the prosumer's storage."""

    min: float
    max: float
    charge_rate_max: float
    discharge_rate_max: float

    def available_charge(self, level: float) -> float:
        """Energy the battery can still accept in one interval."""
        return min(self.charge_rate_max, self.max - level)

    def available_discharge(self, level: float) -> float:
        """Energy the battery can still deliver in one interval."""
        return min(self.discharge_rate_max, level - self.min)


@dataclass(frozen=True)
class ProsumerTransition:
    """
    Everything that physically happened in one interval.

    Energies are per interval. ``grid_exchange`` is positive for net import
    and negative for net export (net metering).
    """

    storage_before: float
    storage_after_command: float
    storage_after: float
    requested_command: float
    applied_command: float
    load: float
    generation: float
    battery_to_load: float
    surplus_to_storage: float
    exported_surplus: float
    uncovered_load: float
    grid_exchange: float
    available_charge: float
    available_discharge: float

    @property
    def grid_import(self) -> float:
        return max(self.grid_exchange, 0.0)

    @property
    def grid_export(self) -> float:
        return max(-self.grid_exchange, 0.0)

    @property
    def storage_change(self) -> float:
        return self.storage_after - self.storage_before
