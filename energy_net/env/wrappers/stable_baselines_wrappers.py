from gymnasium import spaces, ObservationWrapper, RewardWrapper, ActionWrapper, Wrapper
from energy_net.env import EnergyNetEnv
from typing import List
import numpy as np

class StableBaselines3ObservationWrapper(ObservationWrapper):
    """Observation wrapper for :code:`stable-baselines3` algorithms.
    Parameters
    ----------
    env: EnergyNetEnv
    """

    def __init__(self, env: EnergyNetEnv):
                
        super().__init__(env)
        self.env: EnergyNetEnv
        
    @property
    def observation_space(self) -> spaces.Box:
        """Returns single spaces Box object."""

        return self.env.observation_space[0]
    
    def observation(self, observations: List[List[float]]) -> np.ndarray:
        """Returns observations as 1-dimensional numpy array."""

        return np.array(observations[0], dtype='float32')
    
    
    
    
class StableBaselines3ActionWrapper(ActionWrapper):
    """Action wrapper for :code:`stable-baselines3` algorithms.

    Parameters
    ----------
    env: EnergyNetEnv
    """

    def __init__(self, env: EnergyNetEnv):   
        super().__init__(env)
        self.env: EnergyNetEnv

    @property
    def action_space(self) -> spaces.Box:
        """Returns single spaces Box object."""

        return self.env.action_space[0]

    def action(self, actions: List[float]) -> List[List[float]]:
        """Returns actions as 1-dimensional numpy array."""

        return [actions]
    
    
class StableBaselines3RewardWrapper(RewardWrapper):
    """Reward wrapper for :code:`stable-baselines3` algorithms.
    
    Parameters
    ----------
    env: EnergyNetEnv
    """

    def __init__(self, env: EnergyNetEnv):  
        super().__init__(env)
        self.env: EnergyNetEnv

    def reward(self, reward: List[float]) -> float:
        """Returns reward as float value."""

        return reward[0]
    
    
class StableBaselines3Wrapper(Wrapper):
    """Wrapper for :code:`stable-baselines3` algorithms.

    Wraps observations so that they are returned in a 1-dimensional numpy array.
    Wraps actions so that they are returned in a 1-dimensional numpy array.
    Wraps rewards so that it is returned as float value.
    
    Parameters
    ----------
    env: EnergyNetEnv
    """

    def __init__(self, env: EnergyNetEnv):
        env = StableBaselines3ActionWrapper(env)
        env = StableBaselines3RewardWrapper(env)
        env = StableBaselines3ObservationWrapper(env)
        super().__init__(env)
        self.env: EnergyNetEnv