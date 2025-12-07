from typing import Any, Dict, Optional
from energy_net.foundation.grid_entity import ElementaryGridEntity
from energy_net.foundation.dynamics import EnergyDynamics
from energy_net.common.utils import setup_logger
from energy_net.foundation.model import State, Action


class ConsumptionUnit(ElementaryGridEntity):
    """
    Consumption Unit component managing energy consumption.
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

        # Initialize internal state using State class
        self._state = State({
            'consumption': self.current_consumption,
            'time': 0.0
        })

        self.logger.info(
            f"ConsumptionUnit initialized with capacity: {self.consumption_capacity} MWh and initial consumption: {self.current_consumption} MWh")

    def perform_action(self, action: Action) -> None:
        """
        Perform an action on the consumption unit.

        Args:
            action: Action object containing the action to perform.
        """
        # Extract action value from Action object
        pass

    def get_state(self) -> float:
        """
        Retrieves the current consumption as a float.

        Returns:
            float: Current consumption in MWh.
        """
        self.logger.debug(f"Retrieving current consumption: {self.current_consumption} MWh")
        return self.current_consumption

    def update(self, state: State, action: Optional[Action] = None) -> None:
        """
        Updates the consumption level based on dynamics and state.

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
            # For consumption units, we look for a 'value' action
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