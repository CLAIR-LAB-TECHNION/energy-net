from typing import Dict, Optional, List, Any

from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.foundation.grid_entity import CompositeGridEntity
from energy_net.foundation.model import State, Action


class PCSUnit(CompositeGridEntity):
    """
    Power Conversion System Unit (PCSUnit) managing StorageUnits, ProductionUnits, and ConsumptionUnits.

    This class integrates the storage, production, and consumption components, allowing for
    coordinated updates and state management within the smart grid simulation.
    Inherits from CompositeGridEntity to manage its sub-entities.
    """

    def __init__(self,
                 storage_units: Optional[List[Battery]] = None,
                 production_units: Optional[List[ProductionUnit]] = None,
                 consumption_units: Optional[List[ConsumptionUnit]] = None,
                 state_attributes: Optional[Dict[str, Any]] = None,
                 log_file: Optional[str] = 'logs/pcs_unit.log') -> None:
        """
        Initializes the PCSUnit with the provided components.

        Args:
            storage_units (Optional[List[Battery]]): List of Battery instances.
            production_units (Optional[List[ProductionUnit]]): List of ProductionUnit instances.
            consumption_units (Optional[List[ConsumptionUnit]]): List of ConsumptionUnit instances.
            state_attributes (Optional[Dict[str, Any]]): Additional custom state attributes to track.
            log_file (Optional[str]): Path to the log file.
        """
        # Handle None inputs by converting to empty lists
        storage_units = storage_units or []
        production_units = production_units or []
        consumption_units = consumption_units or []

        # Combine all sub-entities into a single list
        sub_entities = storage_units + production_units + consumption_units

        # Initialize the CompositeGridEntity with sub-entities FIRST
        # This sets up self.logger which we need for the warning below
        super().__init__(sub_entities=sub_entities, log_file=log_file)

        # NOW we can safely use self.logger
        # Check for duplicate objects in sub_entities
        seen = set()
        duplicates = [entity for entity in sub_entities if entity in seen or seen.add(entity)]
        if duplicates:
            self.logger.warning(f"Duplicate references to entities found in sub_entities: {duplicates}")

        # Store references to the components
        self.storage_units = storage_units
        self.production_units = production_units
        self.consumption_units = consumption_units

        # Get custom state attributes
        state_config = state_attributes or {}

        # Initialize internal state
        self._state = State({
            'production': 0.0,
            'consumption': 0.0,
            'total_storage': 0.0,
            'energy_change': 0.0,
            'time': 0.0,
            **state_config
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

    def get_total_available_discharge_capacity(self) -> float:
        """
        Calculates the total energy that all storage units can discharge
        during the current timestep.

        Returns:
            float: Total available discharge capacity (same units as battery capacity).
        """
        if not self.storage_units:
            self.logger.error("No storage units available in PCSUnit.")
            return 0.0

        total_capacity = sum(
            battery.get_available_discharge_capacity()
            for battery in self.storage_units
        )

        self.logger.debug(
            f"Total available discharge capacity calculated: {total_capacity}"
        )

        # Optional: track it in PCS internal state if you want
        # self._state.set_attribute('available_discharge_capacity', total_capacity)

        return total_capacity

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

    def update(self, state: State, actions: Optional[Dict[str, Action]] = None) -> float:
        """
        Updates all sub-entities and handles surplus/deficit distribution.
        Now returns the actual energy change due to the applied actions (intentionally not including the changes
        affected by the production/consumption deficit).
        """

        # --- Step 1: Compute totals ---
        total_production = self.get_production()
        total_consumption = self.get_consumption()
        energy_diff = total_production - total_consumption  # positive = surplus, negative = deficit

        current_time = state.get_attribute('time') or 0.0
        self._state.set_attribute('time', current_time)
        self.logger.debug(f"Time {current_time}: production={total_production}, "
                          f"consumption={total_consumption}, diff={energy_diff}")

        # --- Step 2: Apply PCS energy distribution ---
        if self.storage_units and energy_diff != 0:
            if energy_diff > 0:
                self._distribute_surplus(energy_diff)
            else:
                self._distribute_deficit(abs(energy_diff))

        # --- Step 3: Apply user actions and measure effect ---
        storage_before = self.get_total_storage()  # total storage before applying actions

        super().update(state, actions)  # applies the battery actions

        storage_after = self.get_total_storage()  # total storage after applying actions

        # Compute actual energy change caused by the action
        action_energy_change = storage_after - storage_before
        self._state.set_attribute('energy_change', action_energy_change)

        # Return it if needed for reward
        return action_energy_change

    def _distribute_surplus(self, surplus: float) -> None:
        """
        Distributes surplus energy across batteries for charging.

        Args:
            surplus (float): Amount of surplus energy to distribute (positive value).
        """
        # Calculate how much each battery can accept
        battery_capacities = []
        for battery in self.storage_units:
            capacity = battery.get_available_charge_capacity()
            battery_capacities.append(capacity)

        total_capacity = sum(battery_capacities)

        if total_capacity == 0:
            self.logger.warning(f"Cannot store surplus of {surplus} MWh - all batteries at max capacity")
            return

        # Check if we can store all the surplus
        if total_capacity < surplus:
            self.logger.warning(f"Cannot store all surplus: {surplus} MWh available, "
                                f"but only {total_capacity} MWh capacity")
            amount_to_distribute = total_capacity
        else:
            amount_to_distribute = surplus

        # Distribute proportionally based on available capacity
        for idx, (battery, capacity) in enumerate(zip(self.storage_units, battery_capacities)):
            if capacity > 0:
                proportion = capacity / total_capacity
                charge_amount = amount_to_distribute * proportion
                action = Action({'value': charge_amount})
                battery.perform_action(action)
                self.logger.debug(f"Battery_{idx} charged by {charge_amount} MWh "
                                  f"(capacity: {capacity}, proportion: {proportion:.2%})")

    def _distribute_deficit(self, deficit: float) -> None:
        """
        Distributes deficit energy across batteries for discharging.

        Args:
            deficit (float): Amount of deficit energy to cover (positive value).
        """
        # Calculate how much each battery can provide
        battery_capacities = []
        for battery in self.storage_units:
            capacity = battery.get_available_discharge_capacity()
            battery_capacities.append(capacity)

        total_capacity = sum(battery_capacities)

        if total_capacity == 0:
            self.logger.warning(f"Cannot cover deficit of {deficit} MWh - all batteries at min capacity")
            return

        # Check if we can cover all the deficit
        if total_capacity < deficit:
            self.logger.warning(f"Cannot cover all deficit: {deficit} MWh needed, "
                                f"but only {total_capacity} MWh available")
            amount_to_distribute = total_capacity
        else:
            amount_to_distribute = deficit

        # Distribute proportionally based on available capacity
        for idx, (battery, capacity) in enumerate(zip(self.storage_units, battery_capacities)):
            if capacity > 0:
                proportion = capacity / total_capacity
                discharge_amount = -(amount_to_distribute * proportion)  # Negative for discharge
                action = Action({'value': discharge_amount})
                battery.perform_action(action)
                self.logger.debug(f"Battery_{idx} discharged by {abs(discharge_amount)} MWh "
                                  f"(capacity: {capacity}, proportion: {proportion:.2%})")