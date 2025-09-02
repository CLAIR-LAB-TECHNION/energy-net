from gymnasium import spaces, ObservationWrapper, RewardWrapper, ActionWrapper, Wrapper
from energy_net.envs.energy_net_v0 import EnergyNetV0
from typing import List, Dict, Any
import numpy as np

class StableBaselines3ObservationWrapper(ObservationWrapper):
    """Observation wrapper for :code:`stable-baselines3` algorithms.
    Parameters
    ----------
    env: EnergyNetV0
    """

    def __init__(self, env: EnergyNetV0):
        super().__init__(env)
        self.env: EnergyNetV0
        
    @property
    def observation_space(self) -> spaces.Box:
        """Returns flattened Box observation space."""
        if isinstance(self.env.observation_space, spaces.Dict):
            # Concatenate all observation spaces
            low = np.concatenate([space.low for space in self.env.observation_space.values()])
            high = np.concatenate([space.high for space in self.env.observation_space.values()])
            return spaces.Box(low=low, high=high, dtype=np.float32)
        return self.env.observation_space
    
    def observation(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Returns observations as 1-dimensional numpy array."""
        if isinstance(observations, dict):
            return np.concatenate(list(observations.values())).astype(np.float32)
        return observations
    
class StableBaselines3ActionWrapper(ActionWrapper):
    """Action wrapper for :code:`stable-baselines3` algorithms.

    Parameters
    ----------
    env: EnergyNetV0
    """

    def __init__(self, env: EnergyNetV0):   
        super().__init__(env)
        self.env: EnergyNetV0

    @property
    def action_space(self) -> spaces.Box:
        """Returns flattened Box action space."""
        if isinstance(self.env.action_space, spaces.Dict):
            # Concatenate all action spaces
            low = np.concatenate([space.low for space in self.env.action_space.values()])
            high = np.concatenate([space.high for space in self.env.action_space.values()])
            return spaces.Box(low=low, high=high, dtype=np.float32)
        return self.env.action_space

    def action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Splits flattened action into dictionary."""
        if isinstance(self.env.action_space, spaces.Dict):
            # Split the flattened action back into a dictionary
            dims = [space.shape[0] for space in self.env.action_space.values()]
            split_actions = np.split(action, np.cumsum(dims)[:-1])
            return dict(zip(self.env.action_space.keys(), split_actions))
        return action
    
class StableBaselines3RewardWrapper(RewardWrapper):
    """Reward wrapper for :code:`stable-baselines3` algorithms.
    
    Parameters
    ----------
    env: EnergyNetV0
    """

    def __init__(self, env: EnergyNetV0):  
        super().__init__(env)
        self.env: EnergyNetV0

    def reward(self, reward: Dict[str, float]) -> float:
        """Returns average reward across all agents."""
        if isinstance(reward, dict):
            return np.mean(list(reward.values()))
        return reward
    
class StableBaselines3Wrapper(Wrapper):
    """Wrapper for :code:`stable-baselines3` algorithms.

    Wraps observations so that they are returned in a 1-dimensional numpy array.
    Wraps actions so that they are returned in a 1-dimensional numpy array.
    Wraps rewards so that it is returned as float value.
    
    Parameters
    ----------
    env: EnergyNetV0
    """

    def __init__(self, env: EnergyNetV0):
        env = StableBaselines3ActionWrapper(env)
        env = StableBaselines3RewardWrapper(env)
        env = StableBaselines3ObservationWrapper(env)
        super().__init__(env)
        self.env: EnergyNetV0