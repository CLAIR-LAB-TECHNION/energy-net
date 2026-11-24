"""Tests for the one-of-each Prosumer and its pure transition."""

import numpy as np
import pandas as pd
import pytest

from energy_net.dynamics import ConstantDynamics, CSVDynamics
from energy_net.foundation.forecaster import BaseForecaster
from energy_net.foundation.model import Action, State
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.prosumer import (
    BatteryLimits,
    Prosumer,
    prosumer_transition,
)
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery

INF = float("inf")
T0 = pd.Timestamp("2025-01-01 00:00")


def _battery(level=0.0, **config):
    return Battery(DeterministicBattery({}), {"min": 0.0, "max": 100.0, "init": level, **config})


def _prosumer(load=3.0, generation=0.0, level=0.0, **battery_config):
    return Prosumer(
        storage=_battery(level, **battery_config),
        load=ConsumptionUnit(ConstantDynamics(load, "consumption"), {"consumption_capacity": 10.0}),
        generator=ProductionUnit(ConstantDynamics(generation, "generation"), {"production_capacity": 10.0}),
    )


def _transition(level, command, load, generation, limits=None):
    limits = limits or BatteryLimits(0.0, 100.0, INF, INF)
    return prosumer_transition(level, limits, DeterministicBattery({}), command, load, generation)


# ------------------------------------------------ equivalence with legacy

def _legacy_reference(level, command, load, generation, charge_rate, discharge_rate):
    """Run the legacy PCSUnit and PCSEnv.step accounting on one interval."""
    from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
    from energy_net.grid_entities.consumption.consumption_dynamics import (
        ConstantConsumptionDynamics,
    )
    from energy_net.grid_entities.production.production_dynamics import (
        ConstantProductionDynamics,
    )

    battery = _battery(level, charge_rate_max=charge_rate, discharge_rate_max=discharge_rate)
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
    storage_before = pcs.get_total_storage()
    applied = pcs.update(State({"time": 0.0}), {"Battery_0": Action({"value": command})})
    storage_after = pcs.get_total_storage()
    battery_used = max(storage_before + applied - storage_after, 0.0)
    uncovered = max(float(pcs.get_consumption()) - float(pcs.get_production()) - battery_used, 0.0)
    return {
        "storage_after": storage_after,
        "applied_command": applied,
        "battery_to_load": battery_used,
        "uncovered_load": uncovered,
        "grid_exchange": applied + uncovered,
        "available_charge": battery.get_available_charge_capacity(),
        "available_discharge": battery.get_available_discharge_capacity(),
    }


def _cases(seed, count):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        charge_rate = rng.choice([INF, rng.uniform(1.0, 12.0)])
        discharge_rate = rng.choice([INF, rng.uniform(1.0, 12.0)])
        command = rng.uniform(-15.0, 15.0)
        # Legacy asserts when the command exceeds the rate; stay inside it.
        command = float(np.clip(command, -discharge_rate, charge_rate))
        if rng.random() < 0.15:
            command = 0.0
        level = float(rng.choice([0.0, 100.0, rng.uniform(0.0, 100.0)]))
        load = float(rng.choice([0.0, rng.uniform(0.0, 8.0)]))
        yield level, command, load, float(charge_rate), float(discharge_rate)


@pytest.mark.parametrize("seed", range(4))
def test_transition_matches_legacy_pcs_bit_for_bit(seed):
    for level, command, load, charge_rate, discharge_rate in _cases(seed, 250):
        expected = _legacy_reference(level, command, load, 0.0, charge_rate, discharge_rate)
        got = _transition(level, command, load, 0.0,
                          BatteryLimits(0.0, 100.0, charge_rate, discharge_rate))
        for key, value in expected.items():
            assert getattr(got, key) == value, (key, level, command, load)
        assert got.exported_surplus == 0.0


@pytest.mark.parametrize("seed", range(2))
def test_storage_evolution_matches_legacy_with_generation(seed):
    rng = np.random.default_rng(100 + seed)
    for level, command, load, charge_rate, discharge_rate in _cases(seed, 150):
        generation = float(rng.uniform(0.0, 10.0))
        expected = _legacy_reference(level, command, load, generation, charge_rate, discharge_rate)
        got = _transition(level, command, load, generation,
                          BatteryLimits(0.0, 100.0, charge_rate, discharge_rate))
        assert got.storage_after == expected["storage_after"]
        assert got.applied_command == expected["applied_command"]


# ------------------------------------------------------ physical invariants

@pytest.mark.parametrize("seed", range(3))
def test_energy_is_conserved_and_bounds_hold(seed):
    rng = np.random.default_rng(seed)
    for _ in range(500):
        level = float(rng.uniform(0.0, 100.0))
        t = _transition(level, float(rng.uniform(-15, 15)),
                        float(rng.uniform(0, 8)), float(rng.uniform(0, 8)))
        assert 0.0 <= t.storage_after <= 100.0
        assert t.grid_exchange + t.generation - t.load == pytest.approx(
            t.storage_after - t.storage_before, abs=1e-9
        )
        assert t.uncovered_load >= 0.0 and t.exported_surplus >= 0.0
        # Never import for load and export surplus in the same interval.
        assert t.uncovered_load == 0.0 or t.exported_surplus == 0.0


def test_surplus_fills_battery_before_export():
    partial = _transition(level=97.0, command=0.0, load=1.0, generation=6.0)
    assert partial.surplus_to_storage == pytest.approx(3.0)
    assert partial.storage_after == pytest.approx(100.0)
    assert partial.exported_surplus == pytest.approx(2.0)
    assert partial.grid_exchange == pytest.approx(-2.0)

    roomy = _transition(level=10.0, command=0.0, load=1.0, generation=6.0)
    assert roomy.exported_surplus == 0.0
    assert roomy.storage_after == pytest.approx(15.0)


def test_generation_covers_load_before_battery_or_grid():
    t = _transition(level=50.0, command=0.0, load=5.0, generation=3.0)
    assert t.battery_to_load == pytest.approx(2.0)
    assert t.uncovered_load == 0.0
    assert t.grid_exchange == 0.0


# ------------------------------------------------------------- the entity

def test_prosumer_requires_exactly_one_of_each_part():
    load = ConsumptionUnit(ConstantDynamics(1.0), {"consumption_capacity": 1.0})
    generator = ProductionUnit(ConstantDynamics(0.0), {"production_capacity": 1.0})
    with pytest.raises(TypeError, match="storage"):
        Prosumer([_battery()], load, generator)
    with pytest.raises(TypeError, match="load"):
        Prosumer(_battery(), generator, generator)
    with pytest.raises(TypeError, match="generator"):
        Prosumer(_battery(), load, None)


def test_prosumer_requires_ground_truth_dynamics():
    from energy_net.grid_entities.consumption.consumption_dynamics import (
        ConstantConsumptionDynamics,
    )

    load = ConsumptionUnit(ConstantConsumptionDynamics({}), {"consumption_capacity": 1.0})
    generator = ProductionUnit(ConstantDynamics(0.0), {"production_capacity": 1.0})
    with pytest.raises(TypeError, match="GroundTruthDynamics"):
        Prosumer(_battery(), load, generator)


def test_prosumer_has_stable_part_names():
    prosumer = _prosumer()
    assert set(prosumer.sub_entities) == {"storage", "load", "generator"}
    assert prosumer.get_sub_entity("storage") is prosumer.storage


def test_step_realizes_ground_truth_and_commits_storage():
    times = pd.date_range(T0, periods=3, freq="30min")
    prosumer = Prosumer(
        storage=_battery(10.0),
        load=ConsumptionUnit(
            CSVDynamics(pd.Series([2.0, 4.0, 1.0], index=times), "consumption"),
            {"consumption_capacity": 10.0},
        ),
        generator=ProductionUnit(ConstantDynamics(0.5, "generation"), {"production_capacity": 10.0}),
    )
    transition = prosumer.step(times[1], command=3.0)
    assert transition.load == 4.0
    assert transition.generation == 0.5
    assert prosumer.storage_level() == transition.storage_after == pytest.approx(9.5)
    assert prosumer.get_state() == {"storage": 9.5, "load": 4.0, "generation": 0.5}


def test_update_accepts_only_storage_actions_and_needs_a_timestamp():
    prosumer = _prosumer(level=5.0)
    transition = prosumer.update(
        State({"time": 0.0, "timestamp": T0}), {"storage": Action({"value": 2.0})}
    )
    assert transition.applied_command == 2.0
    with pytest.raises(ValueError, match="Only the 'storage'"):
        prosumer.update(State({"timestamp": T0}), {"load": Action({"value": 1.0})})
    with pytest.raises(ValueError, match="timestamp"):
        prosumer.update(State({"time": 0.0}))


class _RecordingForecaster(BaseForecaster):
    def __init__(self):
        self.observed = []
        self.resets = []

    def forecast(self, issue_time, target_times):
        return np.zeros(len(target_times))

    def reset(self, rng=None):
        self.resets.append(rng)

    def observe(self, times, realized):
        self.observed.append((list(times), list(realized)))


def test_forecasters_see_only_realized_values_and_episode_resets():
    prosumer = _prosumer(load=2.5, generation=1.5)
    load_fc, gen_fc = _RecordingForecaster(), _RecordingForecaster()
    prosumer.load.set_forecaster(load_fc)
    prosumer.generator.set_forecaster(gen_fc)

    rng = np.random.default_rng(0)
    prosumer.reset(rng=rng)
    assert load_fc.resets == [rng] and gen_fc.resets == [rng]

    for offset in range(3):
        prosumer.step(T0 + pd.Timedelta(minutes=30 * offset), 0.0)
    assert load_fc.observed == [
        ([T0 + pd.Timedelta(minutes=30 * k)], [2.5]) for k in range(3)
    ]
    assert gen_fc.observed[-1] == ([T0 + pd.Timedelta(minutes=60)], [1.5])


def test_reset_uses_configured_initial_level_unless_overridden():
    prosumer = _prosumer(level=40.0)
    prosumer.step(T0, -10.0)
    prosumer.reset()
    assert prosumer.storage_level() == 40.0
    prosumer.reset(initial_storage=7.0)
    assert prosumer.storage_level() == 7.0


def test_battery_set_energy_level_validates_bounds():
    battery = _battery(10.0)
    battery.set_energy_level(20.0)
    assert battery.energy_level == 20.0 and battery.energy_change == 10.0
    with pytest.raises(ValueError):
        battery.set_energy_level(100.5)
    with pytest.raises(ValueError):
        battery.set_energy_level(float("nan"))
