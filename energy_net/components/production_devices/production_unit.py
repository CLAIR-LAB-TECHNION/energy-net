# components/production_unit.py

from typing import Any, Dict, Optional
from energy_net.components.grid_entity import ElementaryGridEntity
from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.utils.logger import setup_logger  # Import the logger setup


class ProductionUnit(ElementaryGridEntity):
    """
    Production Unit component managing energy production.
    """

    def __init__(self, dynamics: EnergyDynamics, config: Dict[str, Any], log_file: Optional[str] = 'logs/production_unit.log'):
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
        self.initial_production: float = self.current_production  # Assuming initial production is 0.0

        self.logger.info(f"ProductionUnit initialized with capacity: {self.production_capacity} MWh and initial production: {self.current_production} MWh")

    def perform_action(self, action: float) -> None:
        """
        Production units typically do not require actions, but the method is defined for interface consistency.

        Args:
            action (float): Not used in this implementation.
        """
        self.logger.debug(f"Performing action: {action} MW (no effect on ProductionUnit)")
        pass  # Production is typically autonomous and does not respond to actions

    def get_state(self) -> float:
        """
        Retrieves the current production level.

        Returns:
            float: Current production in MWh.
        """
        self.logger.debug(f"Retrieving production state: {self.current_production} MWh")
        return self.current_production

    def update(self, time: float, action: float = 0.0) -> None:
        """
        Updates the production level based on dynamics and time.

        Args:
            time (float): Current time as a fraction of the day (0 to 1).
            action (float, optional): Action to perform (default is 0.0).
                                       Not used in this implementation.
        """
        self.logger.debug(f"Updating ProductionUnit at time: {time} with action: {action} MW")
        # Delegate the production calculation to the dynamics
        previous_production = self.current_production
        self.current_production = self.dynamics.get_value(time=time, action=action)
        self.logger.info(f"ProductionUnit production changed from {previous_production} MWh to {self.current_production} MWh")

    def reset(self) -> None:
        """
        Resets the production unit to its initial production level.
        """
        self.logger.info(f"Resetting ProductionUnit from {self.current_production} MWh to initial production level: {self.initial_production} MWh")
        self.current_production = self.initial_production
        self.logger.debug(f"ProductionUnit reset complete. Current production: {self.current_production} MWh")
