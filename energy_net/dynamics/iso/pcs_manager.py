from typing import List, Dict, Optional, Tuple
import numpy as np
from stable_baselines3 import PPO
from energy_net.components.pcsunit import PCSUnit
import logging
import os
import yaml

class PCSManager:
    def __init__(self, num_agents: int, pcs_unit_config: dict, log_file: str):
        self.num_agents = num_agents
        self.pcs_units = []
        self.trained_agents = []
        self.default_config = pcs_unit_config  # Store default config
        
        # Load individual configs
        configs_path = os.path.join("configs", "pcs_configs.yaml")
        with open(configs_path, "r") as file:
            all_configs = yaml.safe_load(file)
        
        # Initialize PCS units with their specific configs
        for i in range(num_agents):
            agent_key = f"pcs_{i + 1}"
            agent_config = all_configs.get(agent_key, pcs_unit_config)  # Fallback to default if not found
            
            pcs_unit = PCSUnit(
                config=agent_config,
                log_file=log_file
            )
            self.pcs_units.append(pcs_unit)
            self.trained_agents.append(None)
            
    def set_trained_agent(self, agent_idx: int, model_path: str) -> bool:
        """Set trained agent for specific PCS unit"""
        try:
            trained_agent = PPO.load(model_path)
            self.trained_agents[agent_idx] = trained_agent
            return True
        except Exception as e:
            logging.error(f"Failed to load agent {agent_idx}: {e}")
            return False
            
    def simulate_step(
        self, 
        current_time: float,
        iso_buy_price: float,
        iso_sell_price: float
    ) -> Tuple[float, float, float]:
        """
        Simulate one step for all PCS units
        
        Returns:
            total_production: Sum of all production
            total_consumption: Sum of all consumption
            total_net_exchange: Net grid exchange from all units
        """
        total_production = 0.0
        total_consumption = 0.0
        total_net_exchange = 0.0
        
        for idx, (pcs_unit, trained_agent) in enumerate(zip(self.pcs_units, self.trained_agents)):
            if trained_agent is not None:
                # Create observation for this PCS
                pcs_obs = np.array([
                    pcs_unit.battery.get_state(),
                    current_time,
                    pcs_unit.get_self_production(),
                    pcs_unit.get_self_consumption()
                ], dtype=np.float32)
                
                # Get action from trained agent
                battery_action = trained_agent.predict(pcs_obs, deterministic=True)[0].item()
            else:
                # Default behavior for units without trained agent
                battery_params = self.default_config['battery']['model_parameters']  # Use default config
                charge_rate_max = battery_params['charge_rate_max']
                battery_action = np.random.uniform(
                    max(-pcs_unit.battery.get_state(), -charge_rate_max),
                    charge_rate_max
                )
                
            # Update PCS unit state
            pcs_unit.update(time=current_time, battery_action=battery_action)
            
            # Get production and consumption
            production = pcs_unit.get_self_production()
            consumption = pcs_unit.get_self_consumption()
            
            # Calculate net exchange
            if battery_action > 0:
                net_exchange = (consumption + battery_action) - production
            elif battery_action < 0:
                net_exchange = consumption - (production + abs(battery_action))
            else:
                net_exchange = consumption - production
                
            # Add to totals
            total_production += production
            total_consumption += consumption
            total_net_exchange += net_exchange
            
        return total_production, total_consumption, total_net_exchange
        
    def reset_all(self) -> None:
        """Reset all PCS units"""
        for pcs_unit in self.pcs_units:
            pcs_unit.reset()
