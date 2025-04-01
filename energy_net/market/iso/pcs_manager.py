from typing import List, Dict, Optional, Tuple
import numpy as np
from stable_baselines3 import PPO
from energy_net.components.pcsunit import PCSUnit
import logging
import os
import yaml
import random

class PCSManager:
    def __init__(self, num_agents: int, pcs_unit_config: dict, log_file: str):
        self.num_agents = num_agents
        self.pcs_units = []
        self.trained_agents = []
        self.default_config = pcs_unit_config
        self.battery_actions = []  
        self.battery_levels = [] 
        
        # Try to load individual configs, fallback to default if not found
        configs_path = os.path.join("configs", "pcs_configs.yaml")
        try:
            with open(configs_path, "r") as file:
                all_configs = yaml.safe_load(file)
                logging.info("Loaded individual PCS configs from pcs_configs.yaml")
        except FileNotFoundError:
            logging.info(f"No individual configs found at {configs_path}, using default config for all agents")
            all_configs = {}
        
        # Initialize PCS units
        for i in range(num_agents):
            agent_key = f"pcs_{i + 1}"
            # Use agent-specific config if available, otherwise use default
            agent_config = all_configs.get(agent_key, pcs_unit_config)
            
            pcs_unit = PCSUnit(
                config=agent_config,
                log_file=log_file
            )
            self.pcs_units.append(pcs_unit)
            self.trained_agents.append(None)
            
            if agent_config == pcs_unit_config:
                logging.info(f"Using default config for agent {agent_key}")
            else:
                logging.info(f"Using custom config for agent {agent_key}")
            
    def set_trained_agent(self, agent_idx: int, model_path: str) -> bool:
        """Set trained agent for specific PCS unit"""
        try:
            print(f"Loading model for agent {agent_idx} from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            trained_agent = PPO.load(model_path)
            print(f"Model loaded successfully, testing prediction...")
            
            # Test the model with a dummy observation
            test_obs = np.zeros(4, dtype=np.float32)  # 4 is the observation space size
            try:
                test_action = trained_agent.predict(test_obs, deterministic=True)
                print(f"Test prediction successful: {test_action}")
            except Exception as e:
                print(f"Test prediction failed: {e}")
                return False
            
            self.trained_agents[agent_idx] = trained_agent
            print(f"Agent {agent_idx} fully initialized")
            return True
        except Exception as e:
            logging.error(f"Failed to load agent {agent_idx}: {e}")
            print(f"Error in set_trained_agent: {str(e)}")
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
        current_actions = [] 
        current_levels = []  
        
        for idx, (pcs_unit, trained_agent) in enumerate(zip(self.pcs_units, self.trained_agents)):
            if trained_agent is not None:
                pcs_obs = np.array([
                    pcs_unit.battery.get_state(),
                    current_time,
                    pcs_unit.get_self_production(),
                    pcs_unit.get_self_consumption()
                ], dtype=np.float32)
                
                try:
                    logging.info(f"PCS Agent {idx} making prediction with observation: {pcs_obs}")
                    battery_action = trained_agent.predict(pcs_obs, deterministic=True)[0].item()
                    logging.info(f"PCS Agent {idx} action: {battery_action}")
                except Exception as e:
                    logging.error(f"Error in PCS Agent {idx} prediction: {e}")
                    battery_action = 0
            else:
                battery_action = 0
                

            current_actions.append(battery_action)  
            current_levels.append(pcs_unit.battery.get_state()) 
                
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
            
        self.battery_actions.append(current_actions)  
        self.battery_levels.append(current_levels) 
        return total_production, total_consumption, total_net_exchange
        
    def reset_all(self) -> None:
        """Reset all PCS units"""
        for pcs_unit in self.pcs_units:
            pcs_unit.reset()
        self.battery_actions = []
        self.battery_levels = []
