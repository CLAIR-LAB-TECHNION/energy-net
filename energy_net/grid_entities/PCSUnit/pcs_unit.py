from typing import Any, Dict, Optional, List

from original.components.storage_devices.battery import Battery
from original.components.production_devices.production_unit import ProductionUnit
from original.components.consumption_devices.consumption_unit import ConsumptionUnit
from energy_net.dynamics import EnergyDynamics
from energy_net.grid_entity import CompositeGridEntity

class PCSUnit(CompositeGridEntity):
    """
    Power Conversion System Unit (PCSUnit) managing Batteries, ProductionUnits, ConsumptionUnits, and other PCSUnits.
    If you add a PCSUnit to another PCSUnit, the batteries, production units, and consumption units of the added PCSUnit
    will be added to the current PCSUnit.

    This class integrates the storage, production, and consumption components, allowing for
    coordinated updates and state management within the smart grid simulation.
    Inherits from CompositeGridEntity to manage its sub-entities.
    """

    def __init__(self,
                 batteries: List[Battery],
                 production_units: List[ProductionUnit],
                 consumption_units: List[ConsumptionUnit],
                 pcs_units: List['PCSUnit'],
                 log_file: Optional[str] = 'logs/pcs_unit.log') -> None:
        """
        Initializes the PCSUnit with the provided components.

        Args:
            batteries (List[Battery]): List of Battery instances.
            production_units (List[ProductionUnit]): List of ProductionUnit instances.
            consumption_units (List[ConsumptionUnit]): List of ConsumptionUnit instances.
            pcs_units (List[PCSUnit]): List of other PCSUnit instances.
            log_file (Optional[str]): Path to the log file.
        """
        # Combine all sub-entities into a single list
        sub_entities = batteries + production_units + consumption_units + pcs_units

        # Check for duplicate objects in sub_entities
        seen = set()
        duplicates = [entity for entity in sub_entities if entity in seen or seen.add(entity)]
        if duplicates:
            self.logger.warning(f"Duplicate references to entities found in sub_entities: {duplicates}")

        # Initialize the CompositeGridEntity with sub-entities
        super().__init__(sub_entities=sub_entities, log_file=log_file)

        # Store references to the components
        self.batteries = batteries + [battery for pcs in pcs_units for battery in pcs.batteries]
        self.production_units = production_units + [unit for pcs in pcs_units for unit in pcs.production_units]
        self.consumption_units = consumption_units + [unit for pcs in pcs_units for unit in pcs.consumption_units]

    def perform_collective_actions(self, time: float, battery_action: float, consumption_action: float = None,
                          production_action: float = None) -> None:
        """
        Updates the state of all components (batteries, consumption units, production units), based on a shared action for
        each. Also based on the current time. Alternative implementation for perform_action than in the parent class, but
        differently named because of the lack of method overloading in Python.

        Args:
            time (float): Current time as a fraction of the day (0 to 1).
            battery_action (float): Charging (+) or discharging (-) power (MW).
            consumption_action (float, optional): Power consumed by consumption units (MW).
            production_action (float, optional): Power produced by production units (MW).
        """
        self.logger.info(f"Updating PCSUnit at time: {time}, with battery_action: {battery_action} MW")

        # Update Battery with the action
        for battery in self.batteries:
            battery.update(time=time, action=battery_action)
        self.logger.debug(f"PCS battery updated to energy level: {self.get_total_battery_capacity()} MWh")

        # Update ProductionUnit
        for production_unit in self.production_units:
            production_unit.update(time=time, action=production_action)
        self.logger.debug(f"PCS production updated to: {self.get_production()} MWh")

        # Update ConsumptionUnit
        for consumption_unit in self.consumption_units:
            consumption_unit.update(time=time, action=consumption_action)
        self.logger.debug(f"PCS consumption updated to : {self.get_consumption()} MWh")
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

    def get_total_battery_capacity(self) -> float:
        """
        Calculates the total current capacity of all batteries.

        Returns:
            float: Total battery capacity in MWh.
        """
        if not self.batteries:
            self.logger.error("No batteries available in PCSUnit.")
            return 0.0

        total_capacity = sum(battery.get_state() for battery in self.batteries)
        self.logger.debug(f"Total battery capacity calculated: {total_capacity} MWh")
        return total_capacity

    def get_energy_change(self) -> float:
        """
        Retrieves the total energy change from all batteries.

        Returns:
            float: Total energy change in MWh.
        """
        if not self.batteries:
            self.logger.error("No batteries available in PCSUnit.")
            return 0.0

        total_energy_change = sum(battery.energy_change for battery in self.batteries)
        self.logger.debug(f"Total energy change calculated: {total_energy_change} MWh")
        return total_energy_change

    def reset(self, initial_battery_level: Optional[float] = None) -> None:
        """
        Resets all components of the PCSUnit, including batteries, production units, and consumption units.

        Args:
            initial_battery_level (Optional[float]): Optional initial level for batteries.
        """
        self.logger.info("Resetting PCSUnit components.")

        # Reset batteries
        for battery in self.batteries:
            if initial_battery_level is not None:
                battery.reset(initial_battery_level)
            else:
                battery.reset()
        self.logger.debug("All batteries have been reset.")

        # Reset production units
        for production_unit in self.production_units:
            production_unit.reset()
        self.logger.debug("All production units have been reset.")

        # Reset consumption units
        for consumption_unit in self.consumption_units:
            consumption_unit.reset()
        self.logger.debug("All consumption units have been reset.")

        self.logger.info("PCSUnit reset complete.")