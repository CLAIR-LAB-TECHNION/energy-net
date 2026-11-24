"""The prosumer: exactly one storage, one load and one generator."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from energy_net.foundation.dynamics import GroundTruthDynamics
from energy_net.foundation.forecaster import notify_forecaster, reset_forecaster
from energy_net.foundation.grid_entity import CompositeGridEntity
from energy_net.foundation.model import Action, State
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.prosumer.prosumer_classes import (
    BatteryLimits,
    ProsumerTransition,
)
from energy_net.grid_entities.prosumer.prosumer_dynamics import prosumer_transition
from energy_net.grid_entities.storage.battery import Battery


class Prosumer(CompositeGridEntity):
    """
    A household with one battery, one load and one generator.

    The load and the generator are not controlled by any agent: their
    dynamics give the ground truth, and their forecasters (if any) give
    predictions. Only the battery is controlled, through ``command``.
    A household without generation uses a zero-output generator.
    """

    STORAGE = "storage"
    LOAD = "load"
    GENERATOR = "generator"

    def __init__(self, storage: Battery, load: ConsumptionUnit,
                 generator: ProductionUnit):
        if not isinstance(storage, Battery):
            raise TypeError("storage must be a Battery")
        if not isinstance(load, ConsumptionUnit):
            raise TypeError("load must be a ConsumptionUnit")
        if not isinstance(generator, ProductionUnit):
            raise TypeError("generator must be a ProductionUnit")
        for role, unit in ((self.LOAD, load), (self.GENERATOR, generator)):
            if not isinstance(unit.dynamics, GroundTruthDynamics):
                raise TypeError(
                    f"The {role} dynamics must be a GroundTruthDynamics so "
                    "the prosumer can realize and query its ground truth"
                )

        super().__init__(
            [storage, load, generator],
            entity_names={0: self.STORAGE, 1: self.LOAD, 2: self.GENERATOR},
        )
        self.storage = storage
        self.load = load
        self.generator = generator

    # ------------------------------------------------------------ queries

    def battery_limits(self) -> BatteryLimits:
        """Current limits, read from the battery every time."""
        return BatteryLimits(
            min=self.storage.energy_min,
            max=self.storage.energy_max,
            charge_rate_max=self.storage.charge_rate_max,
            discharge_rate_max=self.storage.discharge_rate_max,
        )

    def storage_level(self) -> float:
        return self.storage.get_state()

    def available_charge(self) -> float:
        """Grid-side energy the battery can accept in one interval."""
        return self.battery_limits().available_charge(
            self.storage_level(), self.storage.charge_efficiency
        )

    def available_discharge(self) -> float:
        """Grid-side energy the battery can deliver in one interval."""
        return self.battery_limits().available_discharge(
            self.storage_level(), self.storage.discharge_efficiency
        )

    def get_state(self) -> Dict[str, float]:
        return {
            "storage": self.storage_level(),
            "load": self.load.get_state(),
            "generation": self.generator.get_state(),
        }

    # ------------------------------------------------------------ dynamics

    def step(self, timestamp, command: float, time: float = 0.0) -> ProsumerTransition:
        """
        Realize one interval starting at ``timestamp`` and commit it.

        Load and generation come from the units' ground truth. Afterwards
        each forecaster is told only the values that were just realized.
        """
        timestamp = pd.Timestamp(timestamp)
        state = State({"time": time, "timestamp": timestamp})
        self._state.set_attribute("time", time)

        self.load.update(state)
        self.generator.update(state)
        self.storage.update(state)  # advances the battery clock only

        transition = prosumer_transition(
            level=self.storage_level(),
            limits=self.battery_limits(),
            battery_dynamics=self.storage.dynamics,
            command=float(command),
            load=float(self.load.get_state()),
            generation=float(self.generator.get_state()),
            time=time,
        )
        self.storage.set_energy_level(transition.storage_after)

        realized_at = pd.DatetimeIndex([timestamp])
        for unit, value in ((self.load, transition.load),
                            (self.generator, transition.generation)):
            if unit.forecaster is not None:
                notify_forecaster(unit.forecaster, realized_at, np.array([value]))
        return transition

    def update(self, state: State,
               actions: Optional[Dict[str, Action]] = None) -> ProsumerTransition:
        """GridEntity API. Only the storage accepts an action."""
        if not isinstance(state, State):
            raise TypeError("Prosumer.update requires a State object")
        timestamp = state.get_attribute("timestamp")
        if timestamp is None:
            raise ValueError("Prosumer.update requires a 'timestamp' in the State")

        actions = actions or {}
        unknown = set(actions) - {self.STORAGE}
        if unknown:
            raise ValueError(
                f"Only the '{self.STORAGE}' entity accepts actions; got {sorted(unknown)}"
            )
        command = 0.0
        if self.STORAGE in actions:
            value = actions[self.STORAGE].get_action("value")
            command = 0.0 if value is None else float(value)
        return self.step(timestamp, command, time=state.get_attribute("time") or 0.0)

    def reset(self, initial_storage: Optional[float] = None,
              rng: Optional[np.random.Generator] = None) -> None:
        """
        Reset all parts. Storage starts at ``initial_storage``, or at the
        battery's configured initial level when it is None.
        """
        self.storage.reset(initial_storage)
        self.load.reset()
        self.generator.reset()
        for unit in (self.load, self.generator):
            if unit.forecaster is not None:
                reset_forecaster(unit.forecaster, rng)
        self._state.clear_attributes()
        self._state.set_attribute("time", 0.0)
