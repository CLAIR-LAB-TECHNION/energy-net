import gymnasium as gym
from gymnasium import spaces, ObservationWrapper, RewardWrapper, ActionWrapper
import numpy as np
from typing import List
from energy_net.env import EnergyNetEnv

class ClippedObservationWrapper(ObservationWrapper):
    """Wrapper for observations min-max and periodic normalization.
    
    Observations are clipped to be within the observation space limits.

    Parameters
    ----------
    env: EnergyNetEnv
    """

    def __init__(self, env: EnergyNetEnv) -> None:
        super().__init__(env)
        self.env: EnergyNetEnv

    def observation(self, observations: List[List[float]]) -> List[List[float]]:
        """Returns normalized observations."""

        for i, (o, s) in enumerate(zip(observations, self.observation_space)):
            for j, (o_, l, u) in enumerate(zip(o, s.low, s.high)):
                observations[i][j] = min(max(o_, l), u)

        return observations