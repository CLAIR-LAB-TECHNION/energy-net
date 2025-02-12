# components/consumption_unit.py

from typing import Any, Dict, Optional
from energy_net.components.grid_entity import ElementaryGridEntity
from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.utils.logger import setup_logger  # Import the logger setup


class ConsumptionUnit(ElementaryGridEntity):
    """
    Consumption Unit component managing energy consumption.
    """

    def __init__(self, dynamics: EnergyDynamics, config: Dict[str, Any], log_file: Optional[str] = 'logs/consumption_unit.log'):
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
        self.initial_consumption: float = self.current_consumption  # Assuming initial consumption is 0.0

        self.logger.info(f"ConsumptionUnit initialized with capacity: {self.consumption_capacity} MWh and initial consumption: {self.current_consumption} MWh")

    def perform_action(self, action: float) -> None:
        """
        Consumption units typically do not require actions, but the method is defined for interface consistency.

        Args:
            action (float): Not used in this implementation.
        """
        self.logger.debug(f"Performing action: {action} MW (no effect on ConsumptionUnit)")
        pass  # Consumption is typically autonomous and does not respond to actions

    def get_state(self) -> float:
        """
        Retrieves the current consumption level.

        Returns:
            float: Current consumption in MWh.
        """
        self.logger.debug(f"Retrieving consumption state: {self.current_consumption} MWh")
        return self.current_consumption

    def update(self, time: float, action: float = 0.0) -> None:
        """
        Updates the consumption level based on dynamics and time.

        Args:
            time (float): Current time as a fraction of the day (0 to 1).
            action (float, optional): Action to perform (default is 0.0).
                                       Not used in this implementation.
        """
        self.logger.debug(f"Updating ConsumptionUnit at time: {time} with action: {action} MW")
        # Delegate the consumption calculation to the dynamics
        previous_consumption = self.current_consumption
        self.current_consumption = self.dynamics.get_value(time=time, action=action)
        self.logger.info(f"ConsumptionUnit consumption changed from {previous_consumption} MWh to {self.current_consumption} MWh")

    def reset(self) -> None:
        """
        Resets the consumption unit to its initial consumption level.
        """
        self.logger.info(f"Resetting ConsumptionUnit from {self.current_consumption} MWh to initial consumption level: {self.initial_consumption} MWh")
        self.current_consumption = self.initial_consumption
        self.logger.debug(f"ConsumptionUnit reset complete. Current consumption: {self.current_consumption} MWh")
