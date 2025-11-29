from typing import Any, Dict, Optional, Union
from energy_net.foundation.grid_entity import ElementaryGridEntity
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import setup_logger
from energy_net.foundation.model import State, Action  # Import new classes


class Battery(ElementaryGridEntity):
    """
    Battery component managing energy storage.

    Supports both legacy (float) and new (State/Action) interfaces.
    """

    def __init__(self, dynamics: EnergyDynamics, config: Dict[str, Any], log_file: Optional[str] = 'logs/storage.log'):
        """
        Initializes the Battery with dynamics and configuration parameters.

        Args:
            dynamics (EnergyDynamics): The dynamics defining the storage's behavior.
            config (Dict[str, Any]): Configuration parameters for the storage.
            log_file (str, optional): Path to the Battery log file.

        Raises:
            AssertionError: If required configuration parameters are missing.
        """
        super().__init__(dynamics, log_file)

        # Set up logger
        self.logger = setup_logger('Battery', log_file)
        self.logger.info("Initializing Battery component.")

        # Ensure that all required configuration parameters are provided
        required_params = [
            'min', 'max', 'charge_rate_max', 'discharge_rate_max',
            'charge_efficiency', 'discharge_efficiency', 'init'
        ]
        for param in required_params:
            assert param in config, f"Missing required parameter '{param}' in Battery configuration."

        self.energy_min: float = config['min']
        self.energy_max: float = config['max']
        self.charge_rate_max: float = config['charge_rate_max']
        self.discharge_rate_max: float = config['discharge_rate_max']
        self.charge_efficiency: float = config['charge_efficiency']
        self.discharge_efficiency: float = config['discharge_efficiency']
        self.initial_energy: float = config['init']
        self.energy_level: float = self.initial_energy
        self.energy_change: float = 0.0
        self.current_time: float = 0.0

        # Initialize internal state using new State class
        self._state = State({
            'energy_level': self.energy_level,
            'energy_change': self.energy_change,
            'time': self.current_time
        })

        self.logger.info(f"Battery initialized with energy level: {self.energy_level} MWh")

    def perform_action(self, action: Union[float, Action]) -> None:
        """
        Performs charging or discharging based on the action by delegating to the dynamic.

        Args:
            action: Either a float (legacy) or Action object (new interface).
                   Positive for charging, negative for discharging.
        """
        # Extract action value from either format
        if isinstance(action, Action):
            action_value = action.get_action('value')
            if action_value is None:
                action_value = 0.0
        else:
            action_value = action

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

    def get_state(self) -> Union[float, State]:
        """
        Retrieves the current energy level of the storage.

        Returns:
            float or State: Current energy level in MWh (legacy mode) or 
                           State object with full state information.
        """
        self.logger.debug(f"Retrieving storage state: {self.energy_level} MWh")
        # Return float for backward compatibility, but State is available internally
        return self.energy_level

    def get_full_state(self) -> State:
        """
        Retrieves the full state as a State object.

        Returns:
            State: Complete state including energy level, energy change, time, etc.
        """
        # Update state before returning
        self._state.set_attribute('energy_level', self.energy_level)
        self._state.set_attribute('energy_change', self.energy_change)
        self._state.set_attribute('time', self.current_time)
        return self._state

    def update(self, time: Union[float, State], action: Union[float, Action] = 0.0) -> None:
        """
        Updates the storage's state based on dynamics, time, and action.

        Supports both legacy and new interfaces:
        - Legacy: update(time=0.5, action=5.0)
        - New: update(state, action_obj)

        Args:
            time: Current time as float (0 to 1) OR State object containing time.
            action: Action value as float OR Action object (default is 0.0).
                   Positive for charging, negative for discharging.
        """
        # Handle both legacy (float) and new (State/Action) interfaces
        if isinstance(time, State):
            # New interface: extract time from State
            state_obj = time
            time_value = state_obj.get_attribute('time')
            if time_value is None:
                self.logger.warning("State object missing 'time' attribute, using 0.0")
                time_value = 0.0
        else:
            # Legacy interface: time is a float
            time_value = time
            state_obj = None

        if isinstance(action, Action):
            # New interface: extract action value from Action object
            action_value = action.get_action('value')
            if action_value is None:
                action_value = 0.0
        else:
            # Legacy interface: action is a float
            action_value = action

        self.current_time = time_value

        # Update internal state time
        self._state.set_attribute('time', self.current_time)

        if action_value != 0:
            self.logger.debug(f"Updating Battery at time: {time_value} with action: {action_value} MW")
            self.perform_action(action if isinstance(action, Action) else action_value)

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