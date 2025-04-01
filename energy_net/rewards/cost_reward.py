# energy_net_env/rewards/cost_reward.py

from energy_net.rewards.base_reward import BaseReward
from typing import Dict, Any

class CostReward(BaseReward):
    """
    Reward function based on minimizing the net cost of energy transactions.
    """
    def __init__(self):
        super().__init__()
        self.episode_reward = 0.0
        self.step_count = 0

    def reset(self):
        """Reset internal state for new episode"""
        self.episode_reward = 0.0
        self.step_count = 0

    def compute_reward(self, info: Dict[str, Any]) -> float:
        """
        Computes the reward as the negative net cost.
        
        Args:
            info (Dict[str, Any]): Dictionary containing:
                - net_exchange (float): Amount of energy exchanged with grid.
                                      Positive means PCS is buying from grid.
                                      Negative means PCS is selling to grid.
                - iso_buy_price (float): Price at which ISO buys energy.
                - iso_sell_price (float): Price at which ISO sells energy.
        
        Returns:
            float: The reward value, which is the negative cost.
                  Negative reward means PCS paid money (cost).
                  Positive reward means PCS received money (revenue).
        """
        net_exchange = info.get('net_exchange', 0.0)

        if net_exchange > 0:  # PCS is buying energy from grid
            cost = net_exchange * info['iso_sell_price']  # Pay ISO's sell price
        else:  # PCS is selling energy to grid
            # Use absolute value of net_exchange to ensure proper sign
            cost = net_exchange * info['iso_buy_price']  # Revenue (negative cost)
        
        reward = -cost
        self.episode_reward += reward
        self.step_count += 1
        
        #print(f"[REWARD] Step {self.step_count}: net_exchange={net_exchange:.4f}, cost={cost:.4f}, reward={reward:.4f}, episode_total={self.episode_reward:.4f}")
        
        return reward