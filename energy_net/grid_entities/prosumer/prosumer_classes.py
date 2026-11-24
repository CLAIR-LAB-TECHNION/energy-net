"""Immutable value types describing a prosumer and one transition of it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryLimits:
    """Physical limits of the prosumer's storage (rates are grid-side)."""

    min: float
    max: float
    charge_rate_max: float
    discharge_rate_max: float

    def available_charge(self, level: float, charge_efficiency: float = 1.0) -> float:
        """Grid-side energy the battery can still accept in one interval."""
        return min(self.charge_rate_max, (self.max - level) / charge_efficiency)

    def available_discharge(self, level: float, discharge_efficiency: float = 1.0) -> float:
        """Grid-side energy the battery can still deliver in one interval."""
        return min(self.discharge_rate_max, (level - self.min) * discharge_efficiency)


@dataclass(frozen=True)
class ProsumerTransition:
    """
    Everything that physically happened in one interval.

    Energies are per interval and grid-side unless named ``storage_*``.
    ``grid_exchange`` is positive for net import and negative for net
    export (net metering). With efficiencies below 1, ``losses`` is the
    energy dissipated by the battery, so that
    ``grid_exchange + generation - load == storage_change + losses``.
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
    losses: float = 0.0

    @property
    def grid_import(self) -> float:
        return max(self.grid_exchange, 0.0)

    @property
    def grid_export(self) -> float:
        return max(-self.grid_exchange, 0.0)

    @property
    def storage_change(self) -> float:
        return self.storage_after - self.storage_before
