# components/pcs_unit.py

from typing import Any, Dict, Optional, List

from original.components.storage_devices.battery import Battery
from original.components.production_devices.production_unit import ProductionUnit
from original.components.consumption_devices.consumption_unit import ConsumptionUnit
from energy_net.core.dynamics import EnergyDynamics
from energy_net.core.grid_entity import CompositeGridEntity
from energy_net.grid_entities.storage.battery_dynamics import Deterministicbattery
from original.dynamics.production_dynamics.production_dynmaics_det import DeterministicProduction
from original.dynamics.consumption_dynamics.consumption_dynamics_det import DeterministicConsumption
from energy_net.core.dynamics import DataDrivenDynamics




class PCSUnit(CompositeGridEntity):
    """
    Power Conversion System Unit (PCSUnit) managing Battery, ProductionUnit, and ConsumptionUnit.

    This class integrates the storage, production, and consumption components, allowing for
    coordinated updates and state management within the smart grid simulation.
    Inherits from CompositeGridEntity to manage its sub-entities.
    """

    def __init__(self, config: Dict[str, Any], log_file: Optional[str] = 'logs/pcs_unit.log'):
        """
        Initializes the PCSUnit with its sub-entities based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration parameters for the PCSUnit components.
            log_file (str, optional): Path to the PCSUnit log file.
        """
        # Initialize sub-entities
        sub_entities: List[Battery | ProductionUnit | ConsumptionUnit] = []

        # Initialize Battery
        battery_config = config.get('storage', {})
        battery_dynamics_type = battery_config.get('dynamic_type', 'model_based')
        if battery_dynamics_type == 'model_based':
            battery_model_type = battery_config.get('model_type', 'deterministic_battery')
            if battery_model_type == 'deterministic_battery':

                battery_dynamics: EnergyDynamics = Deterministicbattery(
                    model_parameters=battery_config.get('model_parameters', {})
                )
            else:
                raise ValueError(f"Unsupported storage model type: {battery_model_type}")
        elif battery_dynamics_type == 'data_driven':

            battery_dynamics: EnergyDynamics = DataDrivenDynamics(
                data_file=battery_config.get('data_file', 'battery_data.csv'),
                value_column=battery_config.get('value_column', 'battery_value')
            )
        else:
            raise ValueError(f"Unsupported storage dynamic type: {battery_dynamics_type}")

        battery = Battery(dynamics=battery_dynamics, config=battery_config.get('model_parameters', {}), log_file=log_file)
        sub_entities.append(battery)
        self.battery = battery

        # Initialize ProductionUnit
        production_config = config.get('production_unit', {})
        production_dynamics_type = production_config.get('dynamic_type', 'model_based')
        if production_dynamics_type == 'model_based':
            production_model_type = production_config.get('model_type', 'deterministic_production')
            if production_model_type == 'deterministic_production':

                production_dynamics: EnergyDynamics = DeterministicProduction(
                    model_parameters=production_config.get('model_parameters', {})
                )
            else:
                raise ValueError(f"Unsupported production_unit model type: {production_model_type}")
        elif production_dynamics_type == 'data_driven':


            production_dynamics: EnergyDynamics = DataDrivenDynamics(
                data_file=production_config.get('data_file', 'production_data.csv'),
                value_column=production_config.get('value_column', 'production_value')
            )
        else:
            raise ValueError(f"Unsupported production_unit dynamic type: {production_dynamics_type}")

        production_unit = ProductionUnit(dynamics=production_dynamics, config=production_config.get('model_parameters', {}),  log_file=log_file)
        sub_entities.append(production_unit)
        self.production_unit = production_unit

        # Initialize ConsumptionUnit
        consumption_config = config.get('consumption_unit', {})
        consumption_dynamics_type = consumption_config.get('dynamic_type', 'model_based')
        if consumption_dynamics_type == 'model_based':
            consumption_model_type = consumption_config.get('model_type', 'deterministic_consumption')
            if consumption_model_type == 'deterministic_consumption':


                consumption_dynamics: EnergyDynamics = DeterministicConsumption(
                    model_parameters=consumption_config.get('model_parameters', {})
                )
            else:
                raise ValueError(f"Unsupported consumption_unit model type: {consumption_model_type}")
        elif consumption_dynamics_type == 'data_driven':

            consumption_dynamics: EnergyDynamics = DataDrivenDynamics(
                data_file=consumption_config.get('data_file', 'consumption_data.csv'),
                value_column=consumption_config.get('value_column', 'consumption_value')
            )
        else:
            raise ValueError(f"Unsupported consumption_unit dynamic type: {consumption_dynamics_type}")

        consumption_unit = ConsumptionUnit(dynamics=consumption_dynamics, config=consumption_config.get('model_parameters', {}),  log_file=log_file)
        sub_entities.append(consumption_unit)
        self.consumption_unit = consumption_unit
        # Initialize the CompositeGridEntity with sub-entities
        super().__init__(sub_entities=sub_entities, log_file=log_file)
        
        # Initialize background processes configuration for autonomous injection/withdrawal
        self.background_processes: list[dict] = []
        for bg in config.get('background_processes', []):
            name = bg.get('name')
            interval = bg.get('interval')
            start_time = bg.get('start_time', 0.0)
            end_time = bg.get('end_time', 1.0)
            quantity = bg.get('quantity', 0.0)
            signed_quantity = quantity if bg.get('type') == 'production' else -quantity
            # Determine if this uses step-based window (start_time >= 1 interpreted as step index)
            use_step = isinstance(start_time, (int, float)) and start_time >= 1.0
            bp = {
                'name': name,
                'signed_quantity': signed_quantity,
                'use_step': use_step
            }
            if use_step:
                # For step-based events: define start, end, and interval in steps
                bp['start_step'] = int(start_time)
                bp['end_step'] = int(end_time)
                bp['interval_step'] = int(interval)
            else:
                # For time-based events: track fractional-day times
                bp['interval'] = interval
                bp['start_time'] = start_time
                bp['end_time'] = end_time
                bp['next_fire'] = start_time % 1.0
            self.background_processes.append(bp)
        # Initialize last background action tracking
        self.last_background: dict[str, float] = {bp['name']: 0.0 for bp in self.background_processes}
        # Initialize background residuals tracking
        self.background_residuals: dict[str, float] = {bp['name']: 0.0 for bp in self.background_processes}
        
    def update(self, time: float, battery_action: float, consumption_action: float = None, production_action: float = None, step: int = None) -> None:
        """
        Updates the state of all components based on the current time and storage action.

        Args:
            time (float): Current time as a fraction of the day (0 to 1).
            battery_action (float): Charging (+) or discharging (-) power (MW).
        """
        self.logger.info(f"Updating PCSUnit at time: {time}, with battery_action: {battery_action} MW")

        # Update Battery with the action
        self.battery.update(time=time, action=battery_action)
        self.logger.debug(f"Battery updated to energy level: {self.battery.get_state()} MWh")
        
        # Autonomous background processes
        # First handle any step-based schedules
        for bp in self.background_processes:
            if not bp.get('use_step'):
                continue
            # Reset background tracking
            self.last_background[bp['name']] = 0.0
            self.background_residuals[bp['name']] = 0.0
            if step is None:
                continue
            start = bp['start_step']
            end = bp['end_step']
            interval = bp.get('interval_step', 1)
            # Fire at each step in [start, end] with specified interval
            if start <= step <= end and ((step - start) % interval == 0):
                requested = bp['signed_quantity']
                self.battery.perform_action(requested)
                actual = self.battery.energy_change
                residual = requested - actual
                self.last_background[bp['name']] = requested
                self.background_residuals[bp['name']] = residual
                print(f"[PCSUnit] Background '{bp['name']}': requested {requested} MWh, actual {actual:.4f}, residual {residual:.4f}")
                self.logger.debug(f"Background process '{bp['name']}' applied {requested} MWh at time {time:.4f}")
        # Then handle time-based schedules
        time_mod = time % 1.0
        for bp in self.background_processes:
            if bp.get('use_step'):
                continue
            # Determine if current time_mod falls within the configured window (support wrap-around)
            start = bp['start_time']
            end = bp['end_time']
            if start <= end:
                in_time_window = (start <= time_mod <= end)
            else:
                # Wrap-around window spans midnight
                in_time_window = (time_mod >= start or time_mod <= end)
            time_to_fire = time_mod + 1e-8 >= bp['next_fire']
            # For one-time events (like solar burst), only fire once within window
            if time_to_fire and in_time_window:
                requested = bp['signed_quantity']
                self.battery.perform_action(requested)
                actual = self.battery.energy_change
                residual = requested - actual
                self.last_background[bp['name']] = requested
                self.background_residuals[bp['name']] = residual
                print(f"[PCSUnit] Background '{bp['name']}': requested {requested} MWh, actual {actual:.4f}, residual {residual:.4f}")
                # Schedule next firing
                bp['next_fire'] = (bp['next_fire'] + bp['interval']) % 1.0
                self.logger.debug(f"Background process '{bp['name']}' applied {requested} MWh at time {time_mod:.4f}")
            else:
                self.last_background[bp['name']] = 0.0
                self.background_residuals[bp['name']] = 0.0

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
            self.logger.error("ProductionUnit not found in PCSUnit sub-entities.")
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
            self.logger.error("ConsumptionUnit not found in PCSUnit sub-entities.")
            return 0.0
    
    def get_energy_change(self) -> float:
        """
        Retrieves the energy change from the Battery.

        Returns:
            float: Energy change in MWh.
        """
        # Assuming the identifier for Battery is 'Battery_0' or similar
        battery = next((entity for key, entity in self.sub_entities.items()
                        if isinstance(entity, Battery)), None)
        if battery:
            self.logger.debug(f"Retrieving energy change: {battery.energy_change} MWh")
            return battery.energy_change
        else:
            self.logger.error("Battery not found in PCSUnit sub-entities.")
            return 0.0

    def reset(self, initial_battery_level: Optional[float] = None) -> None:
        """Resets all components with optional initial storage level"""
        for entity in self.sub_entities.values():
            if isinstance(entity, Battery) and initial_battery_level is not None:
                entity.reset(initial_battery_level)  # Pass initial level to storage
            else:
                entity.reset()  # Normal reset for other components
        
        # Reset background process schedule and clear last actions for new day
        for bp in self.background_processes:
            # Reset the next firing time to the configured start_time modulo one day
            bp['next_fire'] = bp.get('start_time', 0.0) % 1.0
        # Clear last background actions and residuals
        self.last_background = {bp['name']: 0.0 for bp in self.background_processes}
        self.background_residuals = {bp['name']: 0.0 for bp in self.background_processes}

    def get_background_actions(self) -> dict[str, float]:
        """
        Retrieve the last background process actions applied during the most recent update.
        """
        return self.last_background

    def get_background_residuals(self) -> dict[str, float]:
        """
        Retrieve the last background process residuals (spill/shortfall) applied during the most recent update.
        """
        return self.background_residuals
