import gymnasium as gym
from gymnasium import spaces, ObservationWrapper, RewardWrapper, ActionWrapper
import numpy as np
from typing import List
from energy_net.env import EnergyNetEnv

class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = self._flatten_action_space(env.action_space)

    def _flatten_action_space(self, action_space):
        if isinstance(action_space, spaces.Dict):
            flat_action_space = spaces.Box(
                low=np.concatenate([space.low for space in action_space.values()]),
                high=np.concatenate([space.high for space in action_space.values()]),
                dtype=np.float32,
            )
            return flat_action_space
        else:
            return action_space
        

    def action(self, action):
        if isinstance(self.env.action_space, spaces.Dict):
            split_actions = np.split(action, len(self.env.action_space.spaces))
            return {key: split_action for key, split_action in zip(self.env.action_space.spaces.keys(), split_actions)}
        else:
            return action   
        
        
        

        
    
    
    
    

        
        
        
