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
            info (Dict[str, Any]): Dictionary containing:
                - shortfall (float): The amount by which realized demand (minus PCS battery response) 
                                     exceeds the dispatch (predicted demand).
                - reserve_cost (float): The cost to cover that shortfall ( shortfall * reserve_price ).
                - pcs_demand (float): How much the PCS is buying/selling.
                - dispatch_cost (float): Cost to cover the predicted demand.
                - iso_sell_price (float): ISO selling price.
                - iso_buy_price (float): ISO buying price.
                
        Returns:
            float: The negative of the total cost the ISO faces (here it's primarily reserve_cost).
        """
        reserve_cost = info.get('reserve_cost', 0.0) # cost to cover that shortfall, shortfall*reserve_price
        pcs_demand = info.get('pcs_demand', 0.0) # how much the PCS is buying/selling
        dispatch_cost = info.get('dispatch_cost', 0.0) # cost to cover that shortfall, shortfall*reserve_price
      
        if pcs_demand>0: 
            price = info.get('iso_sell_price', 0.0)
        else:
            price = info.get('iso_buy_price', 0.0)

        reward = -(reserve_cost + dispatch_cost - pcs_demand*price)
        
        return float(reward)
