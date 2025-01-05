# energy_net_env/rewards/cost_reward.py

from energy_net.rewards.base_reward import BaseReward
from typing import Dict, Any

class CostReward(BaseReward):
    """
    Reward function based on minimizing the net cost of energy transactions.
    """

    def compute_reward(self, info: Dict[str, Any]) -> float:
        """
        Computes the reward as the negative net cost.

        Args:
            info (Dict[str, Any]): Contains:
                - net_exchange: Amount of energy exchanged
                - pricing_function: Function that returns price based on quantity

        Returns:
            float: Negative net cost.
        """
        net_exchange = info.get('net_exchange', 0.0)
        pricing_function = info.get('pricing_function')
        
        reward = -1 * pricing_function(net_exchange)
        
        return reward