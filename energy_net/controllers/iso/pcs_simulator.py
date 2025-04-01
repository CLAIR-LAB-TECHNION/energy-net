"""
PCS (Power Consumption & Storage) Simulator Module

This module provides simulation capabilities for Power Consumption and Storage units
in response to ISO pricing decisions. It simulates how consumers with battery storage
capacity respond to electricity prices set by the ISO.

Key features:
1. Simulation of multiple PCS units with individual battery states
2. Integration with trained RL agents to model PCS decision-making
3. Calculation of aggregate production, consumption, and grid exchange
4. Translation between ISO and PCS observation spaces
5. Support for both rule-based and learned PCS behaviors

This module enables the ISO agent to train with realistic consumer responses
rather than simplified assumptions.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
from energy_net.market.iso.pcs_manager import PCSManager

class PCSSimulator:
    """
    Handles the simulation of PCS (Power Consumption & Storage) units for the ISO controller.
    
    This class is responsible for:
    1. Managing the PCS units and their responses to market conditions
    2. Simulating the behavior of PCS units with or without trained agents
    3. Aggregating production, consumption, and demand from all PCS units
    4. Tracking battery levels and actions across time steps
    
    By extracting this logic from the ISO controller, we create a cleaner separation of concerns
    and make the PCS simulation aspects more maintainable and testable.
    """
    
    def __init__(
        self, 
        num_pcs_agents: int, 
        pcs_unit_config: Dict[str, Any], 
        log_file: str, 
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the PCS simulator.
        
        Args:
            num_pcs_agents: Number of PCS agents to simulate
            pcs_unit_config: Configuration for PCS units
            log_file: Path to the log file
            logger: Logger instance for logging PCS simulation details
        """
        self.logger = logger
        self.pcs_unit_config = pcs_unit_config
        
        # Initialize the PCSManager with the given configuration
        self.pcs_manager = PCSManager(
            num_agents=num_pcs_agents,
            pcs_unit_config=pcs_unit_config,
            log_file=log_file
        )
        
        # Initialize trained agent reference
        self.trained_pcs_agent = None
        
        if self.logger:
            self.logger.info(f"Initialized PCS simulator with {num_pcs_agents} agents")
    
    def set_trained_agent(self, agent_idx: int, model_path: str) -> bool:
        """
        Set a trained agent for a specific PCS unit.
        
        This method loads a trained RL model from disk and assigns it to control
        a specific PCS unit's charging/discharging behavior. This allows for
        simulating sophisticated consumer responses based on learned policies.
        
        Args:
            agent_idx: Index of the PCS unit to set the agent for
            model_path: Path to the trained agent model
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            Exception: If model loading fails (error is caught and logged)
        """
        try:
            # Load the trained agent directly
            from stable_baselines3 import PPO
            
            # First load the model to verify it works
            trained_agent = PPO.load(model_path)
            print(f"Loading model for agent {agent_idx} from {model_path}")
            
            # Test the model with a dummy observation
            dummy_obs = np.zeros(4, dtype=np.float32)  # Match PCS observation size
            test_result = trained_agent.predict(dummy_obs, deterministic=True)
            print(f"Test prediction successful: {test_result}")
            
            # Store the agent directly in PCSManager
            success = self.pcs_manager.set_trained_agent(agent_idx, model_path)
            
            # For compatibility with older code, store a direct reference
            # to the first trained agent
            if success and agent_idx == 0:
                self.trained_pcs_agent = trained_agent
                print(f"Agent {agent_idx} fully initialized")
            
            if self.logger:
                self.logger.info(f"Successfully set trained agent {agent_idx} from {model_path}")
                
            return success
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to set trained agent {agent_idx} from {model_path}: {e}")
            print(f"Error loading model: {str(e)}")
            return False
    
    def translate_to_pcs_observation(self, current_time: float, pcs_idx: int = 0) -> np.ndarray:
        """
        Converts current state to PCS observation format for a specific PCS unit.
        
        The PCS observation includes:
        1. Current battery level (state of charge)
        2. Time of day (normalized to [0,1])
        3. Current local production (e.g., from solar panels)
        4. Current local consumption (electricity demand)
        
        This observation format matches what PCS agents were trained with.
        
        Args:
            current_time: Current time as a fraction of day
            pcs_idx: Index of the PCS unit (default: 0 for first unit)
            
        Returns:
            np.ndarray: Observation array for the PCS agent
        """
        if pcs_idx >= len(self.pcs_manager.agents):
            if self.logger:
                self.logger.error(f"PCS index {pcs_idx} out of range (max: {len(self.pcs_manager.agents)-1})")
            # Return zeros as fallback
            return np.zeros(4, dtype=np.float32)
            
        pcs_unit = self.pcs_manager.agents[pcs_idx]
        
        pcs_observation = np.array([
            pcs_unit.battery.get_state(),
            current_time,
            pcs_unit.get_self_production(),
            pcs_unit.get_self_consumption()
        ], dtype=np.float32)
        
        if self.logger:
            self.logger.debug(
                f"PCS Observation (unit {pcs_idx}):\n"
                f"  Battery Level: {pcs_observation[0]:.2f} MWh\n"
                f"  Time: {pcs_observation[1]:.3f}\n"
                f"  Production: {pcs_observation[2]:.2f} MWh\n"
                f"  Consumption: {pcs_observation[3]:.2f} MWh"
            )
        
        return pcs_observation
    
    def simulate_pcs_response(self, observation: np.ndarray, pcs_idx: int = 0) -> float:
        """
        Simulates a specific PCS unit's response to current market conditions.
        
        This method:
        1. Gets the observation for the specific PCS unit
        2. Uses the unit's trained agent (if available) to determine the battery action
        3. Returns the battery action (charging/discharging rate)
        
        If no trained agent is available, it uses a default charging behavior.
        
        Args:
            observation: Current state observation for PCS unit
            pcs_idx: Index of the PCS unit to simulate (default: 0 for first unit)
            
        Returns:
            float: Battery action (positive for charging, negative for discharging)
        """
        if pcs_idx >= len(self.pcs_manager.agents):
            if self.logger:
                self.logger.error(f"PCS index {pcs_idx} out of range (max: {len(self.pcs_manager.agents)-1})")
            return 0.0
            
        # Get the trained agent for this PCS unit
        agent = self.pcs_manager.agents[pcs_idx].trained_agent
            
        if agent is None:
            if self.logger:
                self.logger.warning(f"No trained agent available for PCS unit {pcs_idx} - simulating default charging behavior")
                print(self.pcs_unit_config['battery']['model_parameters']['charge_rate_max'])
                print("debug")
            return self.pcs_unit_config['battery']['model_parameters']['charge_rate_max']
            
        if self.logger:
            self.logger.debug(f"Sending observation to PCS agent {pcs_idx}: {observation}")
            
        action, _ = agent.predict(observation, deterministic=True)
        battery_action = action.item()
        
        energy_config = self.pcs_unit_config['battery']['model_parameters']
        if self.logger:
            self.logger.info(
                f"PCS Response (unit {pcs_idx}):\n"
                f"  Battery Action: {battery_action:.2f} MWh\n"
                f"  Max Charge: {energy_config['charge_rate_max']:.2f} MWh\n"
                f"  Max Discharge: {energy_config['discharge_rate_max']:.2f} MWh"
            )
            
        return battery_action
    
    def simulate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate the response of all PCS units to the current market conditions.
        
        This is the main method called by the ISO controller to get the aggregate
        response of all PCS units to the current ISO prices. It:
        1. Updates all PCS units based on the current time and prices
        2. Calculates total production, consumption, and net demand
        3. Tracks battery levels and actions
        
        Args:
            state: Dictionary containing the current state
                Required keys:
                - current_time: Current time as a fraction of day
                - iso_buy_price: Current ISO buy price
                - iso_sell_price: Current ISO sell price
                
        Returns:
            Dictionary containing simulation results:
                - production: Total production from all PCS units
                - consumption: Total consumption from all PCS units
                - pcs_demand: Net demand from all PCS units (positive for buying, negative for selling)
                - battery_levels: Current battery levels of all PCS units
                - battery_actions: Current battery actions of all PCS units
        """
        # Extract required values
        current_time = state['current_time']
        iso_buy_price = state['iso_buy_price']
        iso_sell_price = state['iso_sell_price']
        
        # Log the market conditions
        if self.logger:
            self.logger.debug(
                f"Simulating PCS response to market conditions:\n"
                f"  Time: {current_time:.3f}\n"
                f"  ISO Buy Price: {iso_buy_price:.2f}\n"
                f"  ISO Sell Price: {iso_sell_price:.2f}"
            )
        
        # Simulate the step using the PCSManager
        production, consumption, pcs_demand = self.pcs_manager.simulate_step(
            current_time=current_time,
            iso_buy_price=iso_buy_price,
            iso_sell_price=iso_sell_price
        )
        
        # Log the simulation results
        if self.logger:
            self.logger.info(
                f"PCS Simulation Results:\n"
                f"  Production: {production:.2f} MWh\n"
                f"  Consumption: {consumption:.2f} MWh\n"
                f"  Net Demand: {pcs_demand:.2f} MWh"
            )
        
        # Get battery levels and actions
        battery_levels = self.pcs_manager.battery_levels[-1] if self.pcs_manager.battery_levels else []
        battery_actions = self.pcs_manager.battery_actions[-1] if self.pcs_manager.battery_actions else []
        
        return {
            'production': production,
            'consumption': consumption,
            'pcs_demand': pcs_demand,
            'battery_levels': battery_levels,
            'battery_actions': battery_actions
        }
    
    def reset(self) -> None:
        """
        Reset all PCS units to their initial state.
        """
        self.pcs_manager.reset_all()
        
        if self.logger:
            self.logger.info("Reset all PCS units to initial state")
            
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of all PCS units.
        
        Returns:
            Dictionary containing the current state:
                - battery_levels: Current battery levels of all PCS units
                - battery_actions: Most recent battery actions of all PCS units
        """
        return {
            'battery_levels': self.pcs_manager.battery_levels[-1] if self.pcs_manager.battery_levels else [],
            'battery_actions': self.pcs_manager.battery_actions[-1] if self.pcs_manager.battery_actions else []
        }