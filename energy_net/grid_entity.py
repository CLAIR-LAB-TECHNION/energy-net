import copy
from abc import abstractmethod
from collections import OrderedDict
from typing import Union
import numpy as np

from energy_net.dynamics.energy_dynamcis import EnergyDynamics
from energy_net.utils.utils import AggFunc
from energy_net.model.action import EnergyAction
from energy_net.model.state import State
from energy_net.model.reward import Reward
from energy_net.defs import Bounds


class GridEntity:
    """
    This is a base class for all network entities. It provides an interface for stepping through actions,
    predicting the outcome of actions, getting the current state, updating the state, and getting the reward.
    """

    def __init__(self, name: str):
        """
        Constructor for the GridEntity class.

        Parameters:
        name (str): The name of the network entity.
        """
        self.name = name

    @abstractmethod
    def step(self, actions: Union[np.ndarray, EnergyAction], **kwargs) -> None:
        """
        Perform the given action and return the new state and reward.

        Parameters:
        action (EnergyAction): The action to perform.

       """
        
        pass
    
    # TODO: Define the predict method
    @abstractmethod
    def predict(self, action: EnergyAction, state: State):
        """
        Predict the outcome of performing the given action on the given state.

        Parameters:
        action (EnergyAction): The action to perform.
        state (State): The current state.

        Returns:
        list: The predicted new state and reward after performing the action.
        """
        pass

    @abstractmethod
    def update_system_state(self):
        pass

    @abstractmethod
    def get_state(self) -> State:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_observation_space(self) -> Bounds:
        pass

    @abstractmethod
    def get_action_space(self) -> Bounds:
        pass


class ElementaryGridEntity(GridEntity):
    """
    This class is an elementary network entity that is composed of other network entities. It provides an interface for stepping through actions,
    predicting the outcome of actions, getting the current state, updating the state, and getting the reward.
    """

    def __init__(self, name, energy_dynamics: EnergyDynamics , init_state:State):
        super().__init__(name)
        # if the state is none - this is a stateless entity
        self.state = init_state
        self.init_state = init_state
        self.energy_dynamics = energy_dynamics

    def step(self, action: Union[np.ndarray, EnergyAction], **kwargs) -> None:
        new_state = self.energy_dynamics.do(action=action, state=self.state, **kwargs)
        self.state = new_state

            

    # TODO: implement predict
    def predict(self, action: EnergyAction, state: State):
        predicted_state = self.energy_dynamics.predict(action=action, state=state)
        return predicted_state


    def get_state(self) -> State:
        """
        Get the current state of the network entity.

        Returns:
        State: The current state.
        """
        return self.state
    
  
         
    
    def reset(self) -> None:
        self.state = self.init_state

    @abstractmethod
    def get_observation_space(self) -> Bounds:
        pass

    @abstractmethod
    def get_action_space(self) -> Bounds:
        pass

        
class CompositeGridEntity(GridEntity):
    """ 
    This class is a composite network entity that is composed of other network entities. It provides an interface for stepping through actions,
    predicting the outcome of actions, getting the current state, updating the state, and getting the reward.
    """

    def __init__(self, name: str, sub_entities: dict[str, GridEntity] = None, agg_func: AggFunc = None):
        super().__init__(name)
        self.sub_entities = sub_entities
        self.agg_func = agg_func

    def step(self, actions: dict[str, Union[np.ndarray, EnergyAction]], **kwargs) -> None:
        for entity_name, action in actions.items():
            if type(action) is np.ndarray:
                action = self.sub_entities[entity_name].action_type.from_numpy(action)
            
            self.sub_entities[entity_name].step(action, **kwargs)
    # TODO: implement predict
    def predict(self, actions: Union[np.ndarray, dict[str, EnergyAction]]):

        predicted_states = {}
        if type(actions) is np.ndarray:
            # we convert the entity dict to a list and match action to entities by index
            sub_entities = list(self.sub_entities.values())
            for entity_index, action in enumerate(actions):
                predicted_states[sub_entities[entity_index].name] = sub_entities[entity_index].predict(np.array([action]))

        else:
            for entity_name, action in actions.items():
                predicted_states[entity_name] = self.sub_entities[entity_name].predict(action)

        if self.agg_func:
            agg_value = self.agg_func(predicted_states)
            return agg_value
        else:
            return predicted_states


    def get_state(self, numpy_arr = False) -> dict[str, State]:
        state = {}
        for entity in self.sub_entities.values():
            state[entity.name] = entity.get_state()
        
        if self.agg_func:
            state = self.agg_func(state)
            
        if numpy_arr:
            state = np.concatenate([s.to_numpy() for s in state.values()])
            

        return state
    
    def reset(self) -> None:
        for entity in self.sub_entities.values():
            entity.reset()


    def get_observation_space(self) -> dict[str, Bounds]:
        obs_space = {}
        for name, entity in self.sub_entities.items():
            obs_space[name] = entity.get_observation_space()
        
        return obs_space
    

    def get_action_space(self) -> dict[str, Bounds]:
        action_space = {}
        for name, entity in self.sub_entities.items():
            action_space[name] = entity.get_action_space()
            
        return action_space
