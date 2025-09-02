from typing import Dict, Any
from energy_net.model.rewards.base_reward import BaseReward
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

        shortage_penalty = 50.0
        curtailment_penalty = 5.0

        if error > 0:
            # Curtailment case
            error_penalty = curtailment_penalty
        elif error < 0:
            # Shortage case
            error_penalty = shortage_penalty

        #reg_penalty = self.error_penalty * (error ** 2)
        reg_penalty = error_penalty * (error ** 2)



        # Combine rewards
        total_reward = cost_reward - reg_penalty
        print(f"Cost reward: {cost_reward:.3f}, Error: {error:.3f}, Reg penalty: {reg_penalty:.3f}, Total reward: {total_reward:.3f}")
        return float(total_reward)

    def compute_reward_smp(self, info: Dict[str, Any]) -> float:
        # Regularization: squared error between dispatch and realized demand
        realized = info.get('realized_demand', info.get('net_demand', 0.0))
        dispatch = info.get('dispatch', 0.0)
        error = dispatch - realized

        # smp reward
        reward = dispatch + 0 + 0.2 * error + 0.0005 * error ** 2 + 0.00000199 * error ** 3
        return float(reward)

    def compute_reward_normailized(self, info: Dict[str, Any]) -> float:
        """
        Cost-based reward + asymmetric, price-aware regularization on dispatch error.
        """
        # --- Cost reward (negative total cost) ---
        reserve_cost  = float(info.get('reserve_cost', 0.0))
        dispatch_cost = float(info.get('dispatch_cost', 0.0))
        pcs_demand    = float(info.get('pcs_demand', 0.0))
        iso_sell_price = float(info.get('iso_sell_price', 0.0))
        iso_buy_price  = float(info.get('iso_buy_price', 0.0))

        # If net demand positive, you're selling; otherwise you're buying
        price = iso_sell_price if pcs_demand > 0 else iso_buy_price
        cost_reward = -(reserve_cost + dispatch_cost - pcs_demand * price)

        # --- Error terms ---
        realized = float(info.get('realized_demand', info.get('net_demand', 0.0)))
        dispatch = float(info.get('dispatch', 0.0))
        error = dispatch - realized  # >0 over-dispatch (curtail), <0 under (shortage)

        # --- Asymmetric price-aware weights ---
        # Tunable scalars; set once in __init__ (shown here with sane defaults)
        k_over   = getattr(self, "k_over", 0.5)    # scales curtail penalty
        k_under  = getattr(self, "k_under", 2.0)   # scales shortage penalty (typically >> k_over)
        deadband = getattr(self, "deadband", 0.0)  # MW tolerance where we ignore tiny errors
        delta    = getattr(self, "huber_delta", 5.0)  # MW: Huber transition from L2→L1 to avoid blow-ups
        # Optional estimate of "value of lost load"; use high default if not provided
        voll     = float(getattr(self, "voll", 3000.0))  # $/MWh

        # Economic intuition:
        # - Over-dispatch cost ~ lost revenue (curtailment): ~ iso_sell_price
        # - Under-dispatch cost ~ buying power + reserves + risk of unserved load: ~ iso_buy_price + reserve_cost (proxy for scarcity), capped by VOLL
        # Weight them and feed into a (tilted) Huber penalty.
        over_rate  = k_over  * max(iso_sell_price, 0.0)
        under_rate = k_under * min(max(iso_buy_price + reserve_cost, 0.0), voll)

        # Apply deadband
        e = error
        if abs(e) <= deadband:
            reg_penalty = 0.0
            self.error_penalty = 0.0
        else:
            e = e - deadband if e > 0 else e + deadband  # shrink toward 0 outside deadband

            # Huber-style asymmetric penalty:
            #   0.5 * rate * e^2        if |e| <= delta
            #   rate * (delta*|e| - 0.5*delta^2)  otherwise  (linear tails)
            rate = over_rate if e > 0 else under_rate
            ae = abs(e)
            if ae <= delta:
                self.error_penalty = rate  # for transparency/logging
                reg_penalty = 0.5 * rate * (e ** 2)
            else:
                self.error_penalty = rate
                reg_penalty = rate * (delta * ae - 0.5 * (delta ** 2))

        total_reward = cost_reward - reg_penalty
        print(f"Cost reward: {cost_reward:.3f}, Error: {error:.3f}, Reg penalty: {reg_penalty:.3f}, Total reward: {total_reward:.3f}")
        return float(total_reward)
