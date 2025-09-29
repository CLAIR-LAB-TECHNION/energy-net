from typing import Any, Dict, Optional
import math
import sys
from energy_net.grid_entity import ElementaryGridEntity
from energy_net.dynamics import EnergyDynamics
from energy_net.utils import setup_logger
from energy_net.model.action import Action, ActionType


class Battery(ElementaryGridEntity):
    """
    Battery component managing energy storage.

    Args:
        dynamics (EnergyDynamics): Object that computes the next energy value.
        config (Dict[str, Any]): Configuration with required keys:
            - min, max (float): energy bounds [MWh]
            - charge_rate_max, discharge_rate_max (float): power limits [MW]
            - charge_efficiency, discharge_efficiency (float in (0, 1])
            - init (float): initial energy [MWh]
        log_file (Optional[str]): Path to the log file. Defaults to 'logs/storage.log'.
    """

    def __init__(
        self,
        dynamics: EnergyDynamics,
        config: Dict[str, Any],
        log_file: Optional[str] = "logs/storage.log",
    ) -> None:
        super().__init__(dynamics, log_file)

        # Set up logger
        self.logger = setup_logger("Battery", log_file)
        self.logger.info("Initializing Battery component.")

        # Validate required configuration parameters
        required = {
            "min", "max", "charge_rate_max", "discharge_rate_max",
            "charge_efficiency", "discharge_efficiency", "init"
        }
        missing = required.difference(config)
        if missing:
            raise ValueError(f"Missing required Battery config keys: {sorted(missing)}")

        # Initialize parameters with validation
        self.energy_min = float(config["min"])
        self.energy_max = float(config["max"])
        self.charge_rate_max = float(config["charge_rate_max"])
        self.discharge_rate_max = float(config["discharge_rate_max"])
        self.charge_efficiency = float(config["charge_efficiency"])
        self.discharge_efficiency = float(config["discharge_efficiency"])
        self.initial_energy = float(config["init"])

        # Validate parameter values
        if not math.isfinite(self.energy_min) or not math.isfinite(self.energy_max):
            raise ValueError("Energy bounds must be finite.")
        if self.energy_min >= self.energy_max:
            raise ValueError("'min' must be < 'max'.")
        if self.charge_rate_max <= 0 or self.discharge_rate_max <= 0:
            raise ValueError("Rate limits must be positive.")
        for name, eff in [("charge_efficiency", self.charge_efficiency),
                         ("discharge_efficiency", self.discharge_efficiency)]:
            if not (0 < eff <= 1):
                raise ValueError(f"{name} must be in (0, 1].")

        # Clamp initial energy to bounds if needed
        if not (self.energy_min <= self.initial_energy <= self.energy_max):
            self.logger.warning(
                f"Initial energy {self.initial_energy:.4f} is out of bounds [{self.energy_min:.4f}, {self.energy_max:.4f}]; clamping.")
            self.initial_energy = min(max(self.initial_energy, self.energy_min), self.energy_max)

        # Initialize state variables
        self.energy_level = self.initial_energy
        self.energy_change = 0.0
        self.current_time = 0.0
        self.dynamics = dynamics

        self.logger.info(f"Battery initialized at {self.energy_level:.4f} MWh")

    def _get_value(self, **kwargs: Any) -> float:
        """Helper method to get value from dynamics with proper error handling."""
        try:
            return self.dynamics.get_value(**kwargs)
        except TypeError:
            # Remove efficiency parameters if not supported by dynamics
            kwargs.pop("charge_efficiency", None)
            kwargs.pop("discharge_efficiency", None)
            return self.dynamics.get_value(**kwargs)

    def perform_action(self, action: Action) -> None:
        """
        Perform charging/discharging based on action.

        Args:
            action (Action): Action dataclass with id and amount
                           (amount: positive for charging, negative for discharging)
        """
        if not isinstance(action, Action):
            raise ValueError("action must be an Action dataclass instance")
        if action.id != ActionType.STORAGE:
            raise ValueError("Battery only accepts ActionType.STORAGE actions")
        if not math.isfinite(action.amount):
            raise ValueError("Action amount must be a finite float")

        self.logger.debug(f"Performing action: {action.amount:.6f} MW")
        prev_energy = self.energy_level

        # Get new energy value from dynamics
        kwargs = {
            "time": self.current_time,
            "action": action,
            "current_energy": self.energy_level,
            "min_energy": self.energy_min,
            "max_energy": self.energy_max,
            "charge_rate_max": self.charge_rate_max,
            "discharge_rate_max": self.discharge_rate_max,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
        }

        new_value = self._get_value(**kwargs)

        if not math.isfinite(new_value):
            raise ValueError("Dynamics returned a non-finite energy value")

        # Clamp energy to bounds
        self.energy_level = min(max(new_value, self.energy_min), self.energy_max)
        if self.energy_level != new_value:
            self.logger.debug(
                f"Clamped energy from {new_value:.6f} to bounds [{self.energy_min:.6f}, {self.energy_max:.6f}] -> {self.energy_level:.6f}")

        self.energy_change = self.energy_level - prev_energy
        self.logger.info(
            f"Battery energy level changed from {prev_energy:.6f} MWh to {self.energy_level:.6f} MWh (Δ={self.energy_change:.6f})")

    def get_state(self) -> float:
        """
        Get current energy level.

        Returns:
            float: Current energy level in MWh
        """
        self.logger.debug(f"Retrieving storage state: {self.energy_level:.6f} MWh")
        return self.energy_level

    def update(self, time: float, action: Action = None) -> None:
        """
        Update the storage state.

        Args:
            time (float): Current time as a fraction of the day [0, 1]
            action (Action, optional): Action dataclass (default: zero storage action)
        """
        if not math.isfinite(time):
            raise ValueError("time must be a finite float")
        if not (0.0 <= time <= 1.0):
            self.logger.warning(f"Time {time:.6f} outside [0, 1]; continuing anyway")

        if action is None:
            action = Action(id=ActionType.STORAGE, amount=0.0)

        self.logger.debug(f"Updating Battery at time: {time:.6f} with action: {action.amount:.6f} MW")
        self.current_time = time
        self.perform_action(action)

    def reset(self, initial_level: Optional[float] = None) -> None:
        """
        Reset the storage to a specified or default initial level.

        Args:
            initial_level (Optional[float]): If provided, sets the energy level
                                            to this value (clamped to bounds)
        """
        level = self.initial_energy if initial_level is None else float(initial_level)
        if not math.isfinite(level):
            raise ValueError("initial_level must be a finite float")

        # Clamp to bounds if needed
        clamped = min(max(level, self.energy_min), self.energy_max)
        if initial_level is not None and clamped != level:
            self.logger.info(
                f"Requested reset level {level:.6f} out of bounds; clamped to {clamped:.6f}")

        self.energy_level = clamped
        self.energy_change = 0.0
        self.logger.info(f"Battery reset. Current energy level: {self.energy_level:.6f} MWh")
        self.logger.debug("Battery reset complete.")
