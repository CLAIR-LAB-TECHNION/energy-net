# components/pcs_unit.py

from typing import Any, Dict, Optional, List

from energy_net.components.consumption_devices.consumption_unit import ConsumptionUnit
from energy_net.components.production_devices.production_unit import ProductionUnit
from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.grid_entity import CompositeGridEntity

class Electrolyzer(CompositeGridEntity):
    """
    Power Conversion System Unit (electrolyzer) managing Battery, ProductionUnit, and ConsumptionUnit.

    This class integrates the battery, production, and consumption components, allowing for
    coordinated updates and state management within the smart grid simulation.
    Inherits from CompositeGridEntity to manage its sub-entities.
    """

    def __init__(self, config: Dict[str, Any], log_file: Optional[str] = 'logs/electrolyzer.log'):
        """
        Initializes the electrolyzer with its sub-entities based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration parameters for the electrolyzer components.
            log_file (str, optional): Path to the electrolyzer log file.
        """

        # Initialize sub-entities
        sub_entities: List[ConsumptionUnit | ProductionUnit] = []


        # Initialize ConsumptionUnit
        consumption_config = config.get('consumption_unit', {})
        consumption_dynamics_type = consumption_config.get('dynamic_type', 'model_based')
        if consumption_dynamics_type == 'model_based':
            consumption_model_type = consumption_config.get('model_type', 'deterministic_consumption')
            if consumption_model_type == 'deterministic_consumption':
                from energy_net.dynamics.consumption_dynamics.consumption_dynamics_det import ConsumptionDynamicsDet

                consumption_dynamics: EnergyDynamics = ConsumptionDynamicsDet(
                    model_parameters=consumption_config.get('model_parameters', {})
                )
            else:
                raise ValueError(f"Unsupported consumption_unit model type: {consumption_model_type}")
        elif consumption_dynamics_type == 'data_driven':
            from energy_net.dynamics.energy_dynamcis import DataDrivenDynamics

            consumption_dynamics: EnergyDynamics = DataDrivenDynamics(
                data_file=consumption_config.get('data_file', 'consumption_data.csv'),
                value_column=consumption_config.get('value_column', 'consumption_value')
            )
        else:
            raise ValueError(f"Unsupported consumption_unit dynamic type: {consumption_dynamics_type}")

        consumption_unit = ConsumptionUnit(dynamics=consumption_dynamics,
                                           config=consumption_config.get('model_parameters', {}), log_file=log_file)
        sub_entities.append(consumption_unit)
        self.consumption_unit = consumption_unit

        # Initialize ProductionUnit
        production_config = config.get('production_unit', {})
        production_dynamics_type = production_config.get('dynamic_type', 'model_based')
        if production_dynamics_type == 'model_based':
            production_model_type = production_config.get('model_type', 'deterministic_production')
            if production_model_type == 'deterministic_production':
                from energy_net.dynamics.production_dynamics.production_dynmaics_det import ProductionDynamicsDet

                production_dynamics: EnergyDynamics = ProductionDynamicsDet(
                    model_parameters=production_config.get('model_parameters', {})
                )
            else:
                raise ValueError(f"Unsupported production_unit model type: {production_model_type}")
        elif production_dynamics_type == 'data_driven':
            from energy_net.dynamics.energy_dynamcis import DataDrivenDynamics

            production_dynamics: EnergyDynamics = DataDrivenDynamics(
                data_file=production_config.get('data_file', 'production_data.csv'),
                value_column=production_config.get('value_column', 'production_value')
            )
        else:
            raise ValueError(f"Unsupported production_unit dynamic type: {production_dynamics_type}")

        production_unit = ProductionUnit(dynamics=production_dynamics,
                                         config=production_config.get('model_parameters', {}), log_file=log_file)
        sub_entities.append(production_unit)
        self.production_unit = production_unit

        # Initialize the CompositeGridEntity with sub-entities
        super().__init__(sub_entities=sub_entities, log_file=log_file)

    def update(self, time: float, consumption_action: float = None,
               production_action: float = None) -> None:
        """
        Updates the state of all components based on the current time and battery action.

        Args:
            time (float): Current time as a fraction of the day (0 to 1).
            battery_action (float): Charging (+) or discharging (-) power (MW).
        """

        # Update ProductionUnit
        self.production_unit.update(time=time, action=consumption_action)
        self.logger.debug(f"ProductionUnit updated to production: {self.production_unit.get_state()} MWh")

        # Update ConsumptionUnit
        self.consumption_unit.update(time=time, action=production_action)
        self.logger.debug(f"ConsumptionUnit updated to consumption: {self.consumption_unit.get_state()} MWh")

    def get_self_production(self) -> float:
        """
        Retrieves the current self-production value from the ProductionUnit.

        Returns:
            float: Current production in MWh.
        """
        # Assuming the identifier for ProductionUnit is 'ProductionUnit_1' or similar
        production_unit = next((entity for key, entity in self.sub_entities.items()
                                if isinstance(entity, ProductionUnit)), None)
        if production_unit:
            self.logger.debug(f"Retrieving self-production: {production_unit.get_state()} MWh")
            return production_unit.get_state()
        else:
            self.logger.error("ProductionUnit not found in electrolyzer sub-entities.")
            return 0.0

    def get_self_consumption(self) -> float:
        """
        Retrieves the current self-consumption value from the ConsumptionUnit.

        Returns:
            float: Current consumption in MWh.
        """
        # Assuming the identifier for ConsumptionUnit is 'ConsumptionUnit_2' or similar
        consumption_unit = next((entity for key, entity in self.sub_entities.items()
                                 if isinstance(entity, ConsumptionUnit)), None)
        if consumption_unit:
            self.logger.debug(f"Retrieving self-consumption: {consumption_unit.get_state()} MWh")
            return consumption_unit.get_state()
        else:
            self.logger.error("ConsumptionUnit not found in electrolyzer sub-entities.")
            return 0.0

    def reset(self) -> None:
        """Resets all components with optional initial values"""
        for entity in self.sub_entities.values():
           entity.reset()  # Normal reset for other components
