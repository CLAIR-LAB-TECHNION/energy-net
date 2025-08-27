# energy_net_env/rewards/base_reward.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseReward(ABC):
    """
    Abstract base class for reward functions.
    All custom reward functions should inherit from this class and implement the compute_reward method.
    """

    @abstractmethod
    def compute_reward(self, info: Dict[str, Any]) -> float:
        """
        Computes the reward based on the current state and actions.

        Args:
            info (Dict[str, Any]): A dictionary containing relevant information from the environment.

        Returns:
            float: The calculated reward.
        """
        pass
