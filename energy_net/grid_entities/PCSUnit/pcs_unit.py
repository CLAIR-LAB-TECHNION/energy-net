from typing import Any, Dict, Optional, List

from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.foundation.grid_entity import CompositeGridEntity

class PCSUnit(CompositeGridEntity):
    """
    Power Conversion System Unit (PCSUnit) managing StorageUnits, ProductionUnits, and ConsumptionUnits.

    This class integrates the storage, production, and consumption components, allowing for
    coordinated updates and state management within the smart grid simulation.
    Inherits from CompositeGridEntity to manage its sub-entities.
    """

    def __init__(self,
                 storage_units: List[Battery],
                 production_units: List[ProductionUnit],
                 consumption_units: List[ConsumptionUnit],
                 log_file: Optional[str] = 'logs/pcs_unit.log') -> None:
        """
        Initializes the PCSUnit with the provided components.

        Args:
            storage_units (List[Battery]): List of Battery instances.
            production_units (List[ProductionUnit]): List of ProductionUnit instances.
            consumption_units (List[ConsumptionUnit]): List of ConsumptionUnit instances.
            log_file (Optional[str]): Path to the log file.
        """
        # Combine all sub-entities into a single list
        sub_entities = storage_units + production_units + consumption_units

        # Check for duplicate objects in sub_entities
        seen = set()
        duplicates = [entity for entity in sub_entities if entity in seen or seen.add(entity)]
        if duplicates:
            self.logger.warning(f"Duplicate references to entities found in sub_entities: {duplicates}")

        # Initialize the CompositeGridEntity with sub-entities
        super().__init__(sub_entities=sub_entities, log_file=log_file)

        # Store references to the components
        self.storage_units = storage_units
        self.production_units = production_units
        self.consumption_units = consumption_units
    def get_production(self) -> float:
        """
        Calculates the total current production of all production units.

        Returns:
            float: Total production in MWh.
        """
        if not self.production_units:
            self.logger.error("No production units available in PCSUnit.")
            return 0.0

        total_production = sum(production_unit.get_state() for production_unit in self.production_units)
        self.logger.debug(f"Total production calculated: {total_production} MWh")
        return total_production

    def get_consumption(self) -> float:
        """
        Calculates the total current consumption of all consumption units.


        Returns:
            float: Total consumption in MWh.
        """
        if not self.consumption_units:
            self.logger.error("No consumption units available in PCSUnit.")
            return 0.0

        total_consumption = sum(consumption_unit.get_state() for consumption_unit in self.consumption_units)
        self.logger.debug(f"Total consumption calculated: {total_consumption} MWh")
        return total_consumption

    def get_total_storage(self) -> float:
        """
        Calculates the total current storage of all storage_units.

        Returns:
            float: Total storage in MWh.
        """
        if not self.storage_units:
            self.logger.error("No storage units available in PCSUnit.")
            return 0.0

        total_storage = sum(battery.get_state() for battery in self.storage_units)
        self.logger.debug(f"Total storage calculated: {total_storage} MWh")
        return total_storage

    def get_energy_change(self) -> float:
        """
        Retrieves the total energy change from all storage units.

        Returns:
            float: Total energy change in MWh.
        """
        if not self.storage_units:
            self.logger.error("No storage units available in PCSUnit.")
            return 0.0

        total_energy_change = sum(battery.energy_change for battery in self.storage_units)
        self.logger.debug(f"Total energy change calculated: {total_energy_change} MWh")
        return total_energy_change

    def reset(self, initial_storage_unit_level: Optional[float] = None) -> None:
        """
        Resets all components of the PCSUnit, including storage units, production units, and consumption units.

        Args:
            initial_storage_unit_level (Optional[float]): Optional initial level for storage units.
        """
        self.logger.info("Resetting PCSUnit components.")

        # Reset storage
        for storage_unit in self.storage_units:
            if initial_storage_unit_level is not None:
                storage_unit.reset(initial_storage_unit_level)
            else:
                storage_unit.reset()
        self.logger.debug("All storage units have been reset.")

        # Reset production units
        for production_unit in self.production_units:
            production_unit.reset()
        self.logger.debug("All production units have been reset.")

        # Reset consumption units
        for consumption_unit in self.consumption_units:
            consumption_unit.reset()
        self.logger.debug("All consumption units have been reset.")

        self.logger.info("PCSUnit reset complete.")