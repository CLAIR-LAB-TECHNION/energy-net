from typing import Any, Dict, Optional
from energy_net.foundation.grid_entity import ElementaryGridEntity
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import setup_logger
from energy_net.foundation.model import State, Action


class Battery(ElementaryGridEntity):
    """
    Battery component managing energy storage.
    """

    def __init__(self, dynamics: EnergyDynamics, config: Dict[str, Any], log_file: Optional[str] = 'logs/storage.log'):
        """
        Initializes the Battery with dynamics and configuration parameters.

        Args:
            dynamics (EnergyDynamics): The dynamics defining the storage's behavior.
            config (Dict[str, Any]): Configuration parameters for the storage.
                Optional parameters with defaults:
                - min: Minimum energy level (default: 0.0)
                - max: Maximum energy level (default: inf)
                - charge_rate_max: Maximum charge rate (default: inf)
                - discharge_rate_max: Maximum discharge rate (default: inf)
                - charge_efficiency: Charging efficiency (default: 1.0)
                - discharge_efficiency: Discharging efficiency (default: 1.0)
                - init: Initial energy level (default: 0.0)
            log_file (str, optional): Path to the Battery log file.
        """
        super().__init__(dynamics, log_file)

        # Set up logger
        self.logger = setup_logger('Battery', log_file)
        self.logger.info("Initializing Battery component.")

        # Set parameters with least restrictive defaults
        self.energy_min: float = config.get('min', 0.0)
        self.energy_max: float = config.get('max', float('inf'))
        self.charge_rate_max: float = config.get('charge_rate_max', float('inf'))
        self.discharge_rate_max: float = config.get('discharge_rate_max', float('inf'))
        self.charge_efficiency: float = config.get('charge_efficiency', 1.0)
        self.discharge_efficiency: float = config.get('discharge_efficiency', 1.0)
        self.initial_energy: float = config.get('init', 0.0)
        self.energy_level: float = self.initial_energy
        self.energy_change: float = 0.0
        self.current_time: float = 0.0

        # Initialize internal state using State class
        self._state = State({
            'energy_level': self.energy_level,
            'energy_change': self.energy_change,
            'time': self.current_time
        })

        self.logger.info(f"Battery initialized with energy level: {self.energy_level} MWh")

    def perform_action(self, action: Action) -> None:
        """
        Performs charging or discharging based on the action by delegating to the dynamic.

        Args:
            action: Action object containing the action to perform.
                   Positive for charging, negative for discharging.
        """
        # Extract action value from Action object
        action_value = action.get_action('value')
        if action_value is None:
            action_value = 0.0

        self.logger.debug(f"Performing action: {action_value} MW")

        # Delegate the calculation to the dynamics
        previous_energy = self.energy_level
        self.energy_level = self.dynamics.get_value(
            time=self.current_time,
            action=action_value,
            current_energy=self.energy_level,
            min_energy=self.energy_min,
            max_energy=self.energy_max,
            charge_rate_max=self.charge_rate_max,
            discharge_rate_max=self.discharge_rate_max
        )
        self.logger.info(f"Battery energy level changed from {previous_energy} MWh to {self.energy_level} MWh")
        self.energy_change = self.energy_level - previous_energy

        # Update internal state
        self._state.set_attribute('energy_level', self.energy_level)
        self._state.set_attribute('energy_change', self.energy_change)

    def get_available_charge_capacity(self) -> float:
        """
        Calculates how much energy this battery can accept (charge).

        Returns:
            float: Maximum energy that can be charged, considering both
                   the charge rate limit and remaining capacity to max.
        """
        remaining_capacity = self.energy_max - self.energy_level
        return min(self.charge_rate_max, remaining_capacity)

    def get_available_discharge_capacity(self) -> float:
        """
        Calculates how much energy this battery can provide (discharge).

        Returns:
            float: Maximum energy that can be discharged, considering both
                   the discharge rate limit and available energy above min.
        """
        available_energy = self.energy_level - self.energy_min
        return min(self.discharge_rate_max, available_energy)

    def get_state(self) -> float:
        """
        Retrieves the current energy level of the battery as a float.

        Returns:
            float: Current energy level in MWh.
        """
        self.logger.debug(f"Retrieving battery energy level: {self.energy_level} MWh")
        return self.energy_level

    def get_energy_change(self) -> float:
        """
        Retrieves the current energy change of the battery.

        Returns:
            float: Current energy change in MWh.
        """
        self.logger.debug(f"Retrieving battery energy change: {self.energy_change} MWh")
        return self.energy_change

    def update(self, state: State, action: Optional[Action] = None) -> None:
        """
        Updates the storage's state based on dynamics, state, and action.

        Args:
            state: State object containing time and other state information.
            action: Action object containing actions to perform (optional).
                   Positive for charging, negative for discharging.
        """
        # Extract time from State
        time_value = state.get_attribute('time')
        if time_value is None:
            self.logger.warning("State object missing 'time' attribute, using 0.0")
            time_value = 0.0

        # Extract action value from Action object
        action_value = 0.0
        if action is not None:
            action_value = action.get_action('value')
            if action_value is None:
                action_value = 0.0

        self.current_time = time_value

        # Update internal state time
        self._state.set_attribute('time', self.current_time)

        if action_value != 0 and action is not None:
            self.logger.debug(f"Updating Battery at time: {time_value} with action: {action_value} MW")
            self.perform_action(action)

    def reset(self, initial_level: Optional[float] = None) -> None:
        """
        Resets the storage to specified or default initial level.

        Args:
            initial_level: Optional override for initial energy level
        """
        if initial_level is not None:
            self.energy_level = initial_level
            self.logger.info(f"Reset Battery to specified level: {self.energy_level} MWh")
        else:
            self.energy_level = self.initial_energy
            self.logger.info(f"Reset Battery to default level: {self.energy_level} MWh")

        self.energy_change = 0.0
        self.current_time = 0.0

        # Reset internal state
        self._state.set_attribute('energy_level', self.energy_level)
        self._state.set_attribute('energy_change', self.energy_change)
        self._state.set_attribute('time', self.current_time)

        self.logger.debug(f"Battery reset complete. Current energy level: {self.energy_level} MWh")