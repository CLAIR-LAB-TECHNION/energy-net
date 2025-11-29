from typing import Any, Dict, Optional, Union
from energy_net.foundation.grid_entity import ElementaryGridEntity
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import setup_logger
from energy_net.foundation.model import State, Action


class ConsumptionUnit(ElementaryGridEntity):
    """
    Consumption Unit component managing energy consumption.

    Supports both legacy (float) and new (State/Action) interfaces.
    """

    def __init__(self, dynamics: EnergyDynamics, config: Dict[str, Any],
                 log_file: Optional[str] = 'logs/consumption_unit.log'):
        """
        Initializes the ConsumptionUnit with dynamics and configuration parameters.

        Args:
            dynamics (EnergyDynamics): The dynamics defining the consumption unit's behavior.
            config (Dict[str, Any]): Configuration parameters for the consumption unit.
            log_file (str, optional): Path to the ConsumptionUnit log file.

        Raises:
            AssertionError: If required configuration parameters are missing.
        """
        super().__init__(dynamics, log_file)

        # Set up logger
        self.logger = setup_logger('ConsumptionUnit', log_file)
        self.logger.info("Initializing ConsumptionUnit component.")

        # Ensure that 'consumption_capacity' is provided in the configuration
        assert 'consumption_capacity' in config, "Missing 'consumption_capacity' in ConsumptionUnit configuration."

        self.consumption_capacity: float = config['consumption_capacity']
        self.current_consumption: float = 0.0
        self.initial_consumption: float = self.current_consumption

        # Initialize internal state using new State class
        self._state = State({
            'consumption': self.current_consumption,
            'time': 0.0
        })

        self.logger.info(
            f"ConsumptionUnit initialized with capacity: {self.consumption_capacity} MWh and initial consumption: {self.current_consumption} MWh")

    def perform_action(self, action: Union[float, Action]) -> None:
        """
        Perform an action on the consumption unit.

        Args:
            action: Either a float (legacy) or Action object (new interface).
        """
        # Extract action value from either format
        pass
    def get_state(self) -> Union[float, State]:
        """
        Retrieves the current consumption level.

        Returns:
            float or State: Current consumption in MWh (legacy mode) or
                           State object with full state information.
        """
        self.logger.debug(f"Retrieving consumption state: {self.current_consumption} MWh")
        # Return float for backward compatibility, but State is available internally
        return self.current_consumption

    def update(self, state: Union[float, State], action: Union[float, Action] = 0.0) -> None:
        """
        Updates the consumption level based on dynamics and state.

        Supports both legacy and new interfaces:
        - Legacy: update(0.5, action=0.0)  # float is interpreted as time
        - New: update(state_obj, action_obj)

        Args:
            state: State object containing time and other state information OR
                   float (legacy) representing time as fraction of day (0 to 1).
            action: Action value as float OR Action object (default is 0.0).
        """
        # Handle both legacy (float) and new (State) interfaces
        if isinstance(state, State):
            # New interface: extract time from State
            state_obj = state
            time_value = state_obj.get_attribute('time')
            if time_value is None:
                self.logger.warning("State object missing 'time' attribute, using 0.0")
                time_value = 0.0
        else:
            # Legacy interface: state parameter is actually just time as a float
            time_value = state
            state_obj = None

        if isinstance(action, Action):
            # New interface: extract action value from Action object
            # For consumption units, we might look for a 'consumption_target' action
            action_value = action.get_action('value')
            if action_value is None:
                action_value = 0.0
        else:
            # Legacy interface: action is a float
            action_value = action

        # First, perform the action (this may modify internal state)
        self.perform_action(action if isinstance(action, Action) else action_value)

        # Check if there's a pending action from a previous perform_action call
        pending_action = self._state.get_attribute('pending_action')
        if pending_action is not None:
            action_value = pending_action
            # Clear the pending action after using it
            self._state.remove_attribute('pending_action')

        self.logger.debug(f"Updating ConsumptionUnit at time: {time_value} with action: {action_value} MW")

        # Delegate the consumption calculation to the dynamics
        previous_consumption = self.current_consumption
        self.current_consumption = self.dynamics.get_value(time=time_value, action=action_value)

        # Update internal state
        self._state.set_attribute('time', time_value)
        self._state.set_attribute('consumption', self.current_consumption)

        self.logger.info(
            f"ConsumptionUnit consumption changed from {previous_consumption} MWh to {self.current_consumption} MWh")

    def reset(self) -> None:
        """
        Resets the consumption unit to its initial consumption level.
        """
        self.logger.info(
            f"Resetting ConsumptionUnit from {self.current_consumption} MWh to initial consumption level: {self.initial_consumption} MWh")
        self.current_consumption = self.initial_consumption

        # Reset internal state
        self._state.set_attribute('consumption', self.initial_consumption)
        self._state.set_attribute('time', 0.0)

        self.logger.debug(f"ConsumptionUnit reset complete. Current consumption: {self.current_consumption} MWh")