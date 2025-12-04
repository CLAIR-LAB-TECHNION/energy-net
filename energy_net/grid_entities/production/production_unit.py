from typing import Any, Dict, Optional
from energy_net.foundation.grid_entity import ElementaryGridEntity
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import setup_logger
from energy_net.foundation.model import State, Action


class ProductionUnit(ElementaryGridEntity):
    """
    Production Unit component managing energy production.
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

        # Initialize internal state using State class
        self._state = State({
            'production': self.current_production,
            'time': 0.0
        })

        self.logger.info(
            f"ProductionUnit initialized with capacity: {self.production_capacity} MWh and initial production: {self.current_production} MWh")

    def perform_action(self, action: Action) -> None:
        """
        Perform an action on the production unit.

        Args:
            action: Action object containing the action to perform.
        """
        pass

    def get_state(self) -> float:
        """
        Retrieves the current production as a float.

        Returns:
            float: Current production in MWh.
        """
        self.logger.debug(f"Retrieving current production: {self.current_production} MWh")
        return self.current_production

    def update(self, state: State, action: Optional[Action] = None) -> None:
        """
        Updates the production level based on dynamics and state.

        Args:
            state: State object containing time and other state information.
            action: Action object containing actions to perform (optional).
        """
        # Extract time from State
        time_value = state.get_attribute('time')
        if time_value is None:
            self.logger.warning("State object missing 'time' attribute, using 0.0")
            time_value = 0.0

        # Extract action value from Action object
        action_value = 0.0
        if action is not None:
            # For production units, we look for a 'value' action
            action_value = action.get_action('value')
            if action_value is None:
                action_value = 0.0

        # First, perform the action (this may modify internal state)
        if action is not None:
            self.perform_action(action)

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