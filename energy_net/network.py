
from energy_net.components.grid_entity import GridEntity
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
from gymnasium import spaces


from energy_net.stratigic_entity import StrategicEntity
from energy_net.defs import Bounds


class Network():
    def __init__(self, name: str, strategic_entities: list[StrategicEntity], market_network: Union[list[GridEntity], None] = None,
                 electrical_grid: Union[GridEntity, None] = None) -> None:

        self.strategic_entities = {se.name: se for se in strategic_entities}
        self.stratigic_to_network_entity = {se.name: se.network_entity for se in strategic_entities}
        self.market_network = market_network
        self.electrical_grid = electrical_grid
        self.name = name

    def step(self, joint_actions: dict[str, np.ndarray]):
        """
        Advances the simulation by one time step.
        This method should update the state of each network entity.
        """
        rewards = {}
        term = {}
        info = {}
        states = {}
        
        for agent_name, action  in joint_actions.items():
            state = self.stratigic_to_network_entity[agent_name].get_state()
            self.stratigic_to_network_entity[agent_name].step(action) 
            
            new_state = self.stratigic_to_network_entity[agent_name].get_state()
            rewards[agent_name] = self.strategic_entities[agent_name].reward_function(state, action, new_state)
            term[agent_name] = self.strategic_entities[agent_name].is_done()
            info[agent_name] = self.strategic_entities[agent_name].get_info()
            states[agent_name] = new_state
            
        
        return states, rewards, term, info
    
    def reset(self):
        """
        Resets the state of the network and all its entities to their initial state.
        This is typically used at the beginning of a new episode.
        """
        for entity in self.stratigic_to_network_entity.values():
            entity.reset()


    def get_state(self) -> dict[str, np.ndarray]:
        """
        Returns the current state of the network.
        """
        state_dict = {}
        for name, entity in self.stratigic_to_network_entity.items():
            state_dict[name] = entity.get_state(numpy_arr=True)
        
        
        return state_dict  

    def get_observation_space(self) -> dict[str, Bounds]:
        """
        Returns the observation space of the network.
        """
        return {agent_name: entity.get_observation_space() for agent_name, entity in self.stratigic_to_network_entity.items()}
        
        
    
    def get_action_space(self) -> dict[str, Bounds]:
        """
        Returns the action space of the network.
        """ 
        return {agent_name: entity.get_action_space() for agent_name, entity in self.stratigic_to_network_entity.items()}
        

