from typing import Any, Dict, Optional, Union
from energy_net.foundation.grid_entity import ElementaryGridEntity
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import setup_logger
from energy_net.foundation.model import State, Action  # Import new classes


class ProductionUnit(ElementaryGridEntity):
    """
    Production Unit component managing energy production.

    Supports both legacy (float) and new (State/Action) interfaces.
    """

    def __init__(self, dynamics: EnergyDynamics, config: Dict[str, Any],
                 log_file: Optional[str] = 'logs/production_unit.log'):
        """
        Initializes the ProductionUnit with dynamics and configuration parameters.

        Args:
            dynamics (EnergyDynamics): The dynamics defining the production unit's behavior.
            config (Dict[str, Any]): Configuration parameters for the production unit.
            log_file (str, optional): Path to the ProductionUnit log file.

        Raises:
            AssertionError: If required configuration parameters are missing.
        """
        super().__init__(dynamics, log_file)

        # Set up logger
        self.logger = setup_logger('ProductionUnit', log_file)
        self.logger.info("Initializing ProductionUnit component.")

        # Ensure that 'production_capacity' is provided in the configuration
        assert 'production_capacity' in config, "Missing 'production_capacity' in ProductionUnit configuration."

        self.production_capacity: float = config['production_capacity']
        self.current_production: float = 0.0
        self.initial_production: float = self.current_production

        # Initialize internal state using new State class
        self._state = State({
            'production': self.current_production,
            'time': 0.0
        })

        self.logger.info(
            f"ProductionUnit initialized with capacity: {self.production_capacity} MWh and initial production: {self.current_production} MWh")

    def perform_action(self, action: Union[float, Action]) -> None:
        """
        Perform an action on the production unit.

        Args:
            action: Either a float (legacy) or Action object (new interface).
        """
        pass
    def get_state(self) -> Union[float, State]:
        """
        Retrieves the current production level.

        Returns:
            float or State: Current production in MWh (legacy mode) or
                           State object with full state information.
        """
        self.logger.debug(f"Retrieving production state: {self.current_production} MWh")
        # Return float for backward compatibility, but State is available internally
        return self.current_production

    def update(self, time: Union[float, State], action: Union[float, Action] = 0.0) -> None:
        """
        Updates the production level based on dynamics and time.

        Supports both legacy and new interfaces:
        - Legacy: update(time=0.5, action=0.0)
        - New: update(state, action_obj)

        Args:
            time: Current time as float (0 to 1) OR State object containing time.
            action: Action value as float OR Action object (default is 0.0).
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
            # For production units, we might look for a 'production_target' action
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

        self.logger.debug(f"Updating ProductionUnit at time: {time_value} with action: {action_value} MW")

        # Delegate the production calculation to the dynamics
        previous_production = self.current_production
        self.current_production = self.dynamics.get_value(time=time_value, action=action_value)

        # Update internal state
        self._state.set_attribute('time', time_value)
        self._state.set_attribute('production', self.current_production)

        self.logger.info(
            f"ProductionUnit production changed from {previous_production} MWh to {self.current_production} MWh")

    def reset(self) -> None:
        """
        Resets the production unit to its initial production level.
        """
        self.logger.info(
            f"Resetting ProductionUnit from {self.current_production} MWh to initial production level: {self.initial_production} MWh")
        self.current_production = self.initial_production

        # Reset internal state
        self._state.set_attribute('production', self.initial_production)
        self._state.set_attribute('time', 0.0)

        self.logger.debug(f"ProductionUnit reset complete. Current production: {self.current_production} MWh")