"""
Record legacy PCSUnit transitions (provenance only; needs the pre-refactor
code base, see the commit stored in the fixture).

Usage (from the repository root):
    python -m tests.golden.record_legacy_pcs_unit
"""

import os
import subprocess

import numpy as np

from energy_net.foundation.model import Action, State
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.consumption.consumption_dynamics import (
    ConstantConsumptionDynamics,
)
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.grid_entities.production.production_dynamics import (
    ConstantProductionDynamics,
)
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_pcs_unit_transitions.npz")
INPUTS = ("level", "command", "load", "generation", "charge_rate", "discharge_rate")
OUTPUTS = ("storage_after", "applied_command", "battery_to_load",
           "uncovered_load", "grid_exchange", "available_charge", "available_discharge")
INF = float("inf")


def _legacy(level, command, load, generation, charge_rate, discharge_rate):
    battery = Battery(DeterministicBattery({}), {
        "min": 0.0, "max": 100.0, "init": level,
        "charge_rate_max": charge_rate, "discharge_rate_max": discharge_rate,
    })
    production = []
    if generation:
        production = [ProductionUnit(
            ConstantProductionDynamics({"base_load": generation}),
            {"production_capacity": 10.0},
        )]
    pcs = PCSUnit(
        storage_units=[battery],
        production_units=production,
        consumption_units=[ConsumptionUnit(
            ConstantConsumptionDynamics({"base_load": load}),
            {"consumption_capacity": 10.0},
        )],
    )
    before = pcs.get_total_storage()
    applied = pcs.update(State({"time": 0.0}), {"Battery_0": Action({"value": command})})
    after = pcs.get_total_storage()
    used = max(before + applied - after, 0.0)
    uncovered = max(float(pcs.get_consumption()) - float(pcs.get_production()) - used, 0.0)
    return (after, applied, used, uncovered, applied + uncovered,
            battery.get_available_charge_capacity(),
            battery.get_available_discharge_capacity())


def _cases(rng, count, with_generation):
    rows = []
    for _ in range(count):
        charge_rate = float(rng.choice([INF, rng.uniform(1.0, 12.0)]))
        discharge_rate = float(rng.choice([INF, rng.uniform(1.0, 12.0)]))
        # Legacy asserts when the command exceeds the rate; stay inside it.
        command = float(np.clip(rng.uniform(-15.0, 15.0), -discharge_rate, charge_rate))
        if rng.random() < 0.15:
            command = 0.0
        level = float(rng.choice([0.0, 100.0, rng.uniform(0.0, 100.0)]))
        load = float(rng.choice([0.0, rng.uniform(0.0, 8.0)]))
        generation = float(rng.uniform(0.0, 10.0)) if with_generation else 0.0
        rows.append((level, command, load, generation, charge_rate, discharge_rate))
    return rows


def record():
    rng = np.random.default_rng(2025)
    cases = _cases(rng, 1500, False) + _cases(rng, 500, True)
    inputs = np.array(cases, dtype=float)
    outputs = np.array([_legacy(*case) for case in cases], dtype=float)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    np.savez_compressed(FIXTURE, inputs=inputs, outputs=outputs,
                        input_names=np.array(INPUTS), output_names=np.array(OUTPUTS),
                        recorded_at_commit=np.array(commit))
    print(f"Recorded {len(cases)} PCSUnit transitions to {FIXTURE}")


if __name__ == "__main__":
    record()
