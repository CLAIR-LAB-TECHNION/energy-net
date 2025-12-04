from typing import Any, Dict, Optional, List, Union

from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.foundation.grid_entity import CompositeGridEntity
from energy_net.foundation.model import State, Action  # Import new classes


class PCSUnit(CompositeGridEntity):
    """
    Power Conversion System Unit (PCSUnit) managing StorageUnits, ProductionUnits, and ConsumptionUnits.

    This class integrates the storage, production, and consumption components, allowing for
    coordinated updates and state management within the smart grid simulation.
    Inherits from CompositeGridEntity to manage its sub-entities.

    Supports both legacy (float) and new (State/Action) interfaces.
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

        # Initialize internal state using new State class
        self._state = State({
            'production': 0.0,
            'consumption': 0.0,
            'total_storage': 0.0,
            'energy_change': 0.0,
            'time': 0.0
        })

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

        # Update internal state
        self._state.set_attribute('production', total_production)

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

        # Update internal state
        self._state.set_attribute('consumption', total_consumption)

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

        # Update internal state
        self._state.set_attribute('total_storage', total_storage)

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

        total_energy_change = sum(battery.get_energy_change() for battery in self.storage_units)
        self.logger.debug(f"Total energy change calculated: {total_energy_change} MWh")

        # Update internal state
        self._state.set_attribute('energy_change', total_energy_change)

        return total_energy_change

    def get_state(self) -> Dict[str, float]:
        """
        Retrieves the current state of the PCSUnit as a dictionary.

        Returns:
            Dict[str, float]: Dictionary with production, consumption, storage, and energy change.
        """
        return {
            'production': self.get_production(),
            'consumption': self.get_consumption(),
            'total_storage': self.get_total_storage(),
            'energy_change': self.get_energy_change()
        }

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

        # Reset internal state
        self._state.set_attribute('production', 0.0)
        self._state.set_attribute('consumption', 0.0)
        self._state.set_attribute('total_storage', 0.0)
        self._state.set_attribute('energy_change', 0.0)
        self._state.set_attribute('time', 0.0)

        self.logger.info("PCSUnit reset complete.")

    def update(self, state: State, actions: Optional[Dict[str, Action]] = None) -> None:
        """
        Updates all sub-entities based on the provided state, production, and consumption.

        Args:
            state: State object containing time and other state information.
            actions: Optional user-defined actions for batteries (only applied if surplus exists).
        """
        # --- Extract time from state ---
        current_time = state.get_attribute('time') or 0.0
        self._state.set_attribute('time', current_time)

        # --- Get current totals ---
        total_production = self.get_production()  # float
        total_consumption = self.get_consumption()  # float
        energy_diff = total_production - total_consumption  # positive = surplus, negative = deficit

        self.logger.debug(f"Time {current_time}: production={total_production}, "
                          f"consumption={total_consumption}, diff={energy_diff}")

        # --- Prepare battery actions based on surplus/deficit ---
        battery_actions: Dict[str, Action] = {}
        if self.storage_units:
            num_batteries = len(self.storage_units)

            for idx, battery in enumerate(self.storage_units):
                batt_id = f"Battery_{idx}"

                if energy_diff > 0:
                    # Surplus → charge
                    max_charge = 20.0
                    charge_value = min(max_charge, energy_diff / num_batteries)

                    # If user provided an action, respect it but cap at available surplus
                    if actions and batt_id in actions:
                        charge_value = min(charge_value, actions[batt_id].get_action('value') or 0.0)

                    battery_actions[batt_id] = Action({'value': charge_value})

                elif energy_diff < 0:
                    # Deficit → discharge to cover deficit
                    discharge_value = energy_diff / num_batteries  # negative value
                    battery_actions[batt_id] = Action({'value': discharge_value})

                else:
                    # Balanced → do nothing
                    battery_actions[batt_id] = Action({'value': 0.0})

        # --- Update batteries ---
        for idx, battery in enumerate(self.storage_units):
            batt_id = f"Battery_{idx}"
            action = battery_actions.get(batt_id)
            battery.update(state, action)

        # --- Update production and consumption units ---
        for idx, prod_unit in enumerate(self.production_units):
            prod_unit.update(state)

        for idx, cons_unit in enumerate(self.consumption_units):
            cons_unit.update(state)
