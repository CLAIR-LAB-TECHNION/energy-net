from typing import Dict, Any
from energy_net.rewards.base_reward import BaseReward
import numpy as np

class ISOReward(BaseReward):
    """
    Reward function for the ISO in a scenario with uncertain (stochastic) demand,
    reflecting the cost of reserve activation (shortfall penalty).
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_reward(self, info: Dict[str, Any]) -> float:
        """
        Calculate ISO's reward for a single timestep in the 6.3 context.
        
        Args:
            info: Dictionary containing:
                - shortfall (float): The amount by which realized demand (minus PCS battery response) 
                                     exceeds the dispatch (predicted demand).
                - reserve_cost (float): The cost to cover that shortfall ( shortfall * reserve_price ).
                - (Optionally) other cost/revenue terms if you want to include them.
                
        Returns:
            float: The negative of the total cost the ISO faces (here it's primarily reserve_cost).
        """
        shortfall = info.get('shortfall', 0.0)  # how much demand exceeds dispatch    
        reserve_cost = info.get('reserve_cost', 0.0) # cost to cover that shortfall
        pcs_demand = info.get('pcs_demand', 0.0) # how much the PCS is buying/selling
      
        if pcs_demand>0: 
            price = info.get('buy_price', 0.0)
        else:
            price = info.get('sell_price', 0.0)

        reward = -(reserve_cost + pcs_demand*price)
        
        return float(reward)
