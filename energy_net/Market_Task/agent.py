from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    @abstractmethod
    def act(self, obs) -> np.ndarray:
        """Return an action given an observation."""
        pass

    def learn(self, transition):
        """Optional for learning agents."""
        pass

    def reset(self):
        """Called at the start of an episode."""
        pass