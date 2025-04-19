from typing import Dict, Any
from energy_net.model.rewards.base_reward import BaseReward
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
        # Encourage dispatch to match realized demand, penalize any mismatch
        dispatch = info.get('dispatch', 0.0)
        realized = info.get('realized_demand', 0.0)
        mismatch = abs(dispatch - realized)
        print(f"Dispatch: {dispatch}, Realized: {realized}, Mismatch: {mismatch}")
        # Negative of mismatch: lower mismatch yields higher reward
        return float(-mismatch)
