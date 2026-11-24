"""
Pure prosumer transition: one storage, one load, one generator.

The function has no side effects, so it is the single definition of the
prosumer's physics. With zero generation it reproduces the legacy
PCSUnit.update + PCSEnv.step accounting bit for bit; the floating-point
expressions below deliberately mirror the legacy ones.
"""

from energy_net.grid_entities.prosumer.prosumer_classes import (
    BatteryLimits,
    ProsumerTransition,
)


def _battery_step(battery_dynamics, level: float, limits: BatteryLimits,
                  action: float, time: float) -> float:
    """New storage level after asking the battery for ``action`` energy."""
    return battery_dynamics.get_value(
        time=time,
        action=action,
        current_energy=level,
        min_energy=limits.min,
        max_energy=limits.max,
        charge_rate_max=limits.charge_rate_max,
        discharge_rate_max=limits.discharge_rate_max,
    )


def prosumer_transition(level: float, limits: BatteryLimits, battery_dynamics,
                        command: float, load: float, generation: float,
                        time: float = 0.0) -> ProsumerTransition:
    """
    Compute one interval of the prosumer.

    Order of events:
      1. The agent's battery command is applied (clipped by the battery).
      2. The residual ``generation - load`` is balanced against the
         battery: surplus charges it, deficit discharges it.
      3. Load the battery could not cover is imported from the grid.
      4. Surplus the battery could not absorb is exported to the grid.

    Args:
        level: Storage level at the start of the interval.
        limits: Battery limits.
        battery_dynamics: Object with the DeterministicBattery ``get_value``
            signature; it is only queried, never mutated.
        command: Requested battery energy (positive charges).
        load: Realized household demand in the interval.
        generation: Realized generation in the interval.
        time: Passed through to the battery dynamics.
    """
    storage_before = level

    # 1. Agent command. Like Battery.update, a zero command is a no-op.
    if command != 0:
        after_command = _battery_step(battery_dynamics, level, limits, command, time)
    else:
        after_command = level
    applied = after_command - storage_before

    # 2. Self-balancing of the residual, as in PCSUnit._distribute_*.
    storage_after = after_command
    surplus_to_storage = 0.0
    energy_diff = generation - load  # positive = surplus, negative = deficit
    if energy_diff != 0:
        if energy_diff > 0:
            capacity = limits.available_charge(after_command)
            if capacity > 0:
                surplus_to_storage = capacity if capacity < energy_diff else energy_diff
                storage_after = _battery_step(
                    battery_dynamics, after_command, limits, surplus_to_storage, time
                )
        else:
            deficit = abs(energy_diff)
            capacity = limits.available_discharge(after_command)
            if capacity > 0:
                amount = capacity if capacity < deficit else deficit
                storage_after = _battery_step(
                    battery_dynamics, after_command, limits, -amount, time
                )

    # 3. Grid covers what the battery did not (legacy expressions).
    battery_to_load = max(storage_before + applied - storage_after, 0.0)
    uncovered_load = max(load - generation - battery_to_load, 0.0)

    # 4. Surplus the battery could not absorb is exported.
    exported_surplus = max(energy_diff - surplus_to_storage, 0.0) if energy_diff > 0 else 0.0

    # x - 0.0 == x exactly, so with no export this equals the legacy
    # ``applied + uncovered`` bit for bit.
    grid_exchange = applied + uncovered_load - exported_surplus

    return ProsumerTransition(
        storage_before=storage_before,
        storage_after_command=after_command,
        storage_after=storage_after,
        requested_command=command,
        applied_command=applied,
        load=load,
        generation=generation,
        battery_to_load=battery_to_load,
        surplus_to_storage=surplus_to_storage,
        exported_surplus=exported_surplus,
        uncovered_load=uncovered_load,
        grid_exchange=grid_exchange,
        available_charge=limits.available_charge(storage_after),
        available_discharge=limits.available_discharge(storage_after),
    )
