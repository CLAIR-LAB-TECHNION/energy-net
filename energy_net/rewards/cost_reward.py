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
            info (Dict[str, Any]): Dictionary containing:
                - net_exchange (float): Amount of energy exchanged with grid.
                - iso_buy_price (float): ISO buying price for energy.
                - iso_sell_price (float): ISO selling price for energy.
        
        Returns:
            float: The reward value, which is the negative cost.
        """
        net_exchange = info.get('net_exchange', 0.0)

        if net_exchange > 0:
            cost = net_exchange * info['iso_buy_price']
        else:
            cost = net_exchange * info['iso_sell_price']
        
        return -cost