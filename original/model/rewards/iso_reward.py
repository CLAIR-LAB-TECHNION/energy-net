from typing import Dict, Any
from original.model.rewards.base_reward import BaseReward
import numpy as np

class ISOReward(BaseReward):
    """
    Reward function for the ISO in a scenario with uncertain (stochastic) demand,
    reflecting the cost of reserve activation (shortfall penalty).
    """
    
    def __init__(self, error_penalty: float = 5.0):
        """
        Initialize ISOReward with an error penalty factor.

        Args:
            error_penalty: Weight for the squared dispatch error regularization.
        """
        super().__init__()
        self.error_penalty = error_penalty
        
    def compute_reward(self, info: Dict[str, Any]) -> float:
        """
        Calculate ISO's reward combining cost-based reward and regularization on dispatch error.
        """
        # Base cost-based reward (negative total cost)
        reserve_cost = info.get('reserve_cost', 0.0)
        dispatch_cost = info.get('dispatch_cost', 0.0)
        pcs_demand = info.get('pcs_demand', 0.0)
        # Determine price direction
        if pcs_demand > 0:
            price = info.get('iso_sell_price', 0.0)
        else:
            price = info.get('iso_buy_price', 0.0)
        cost_reward = -(reserve_cost + dispatch_cost - pcs_demand * price)

        # Regularization: squared error between dispatch and realized demand
        realized = info.get('realized_demand', info.get('net_demand', 0.0))
        dispatch = info.get('dispatch', 0.0)
        error = dispatch - realized
        reg_penalty = self.error_penalty * (error ** 2)

        # Combine rewards
        total_reward = cost_reward - reg_penalty
        print(f"Cost reward: {cost_reward:.3f}, Error: {error:.3f}, Reg penalty: {reg_penalty:.3f}, Total reward: {total_reward:.3f}")
        return float(total_reward)
