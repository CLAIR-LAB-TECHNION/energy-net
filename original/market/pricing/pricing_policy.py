"""
Pricing Policy Module

This module defines the pricing policies that can be used by the ISO agent.
"""

from enum import Enum

class PricingPolicy(Enum):
    """
    Enum for different pricing policies that can be used by ISO.
    
    Values:
        QUADRATIC: Uses a quadratic function to determine prices
        ONLINE: Uses online learning for price determination
        CONSTANT: Uses constant prices
        INTERVALS: Uses intervals for price determination
        QUADRATIC_INTERVALS: Uses quadratic intervals for price determination
    """
    QUADRATIC = "quadratic"
    ONLINE = "online"
    CONSTANT = "constant"
    INTERVALS = "intervals"
    QUADRATIC_INTERVALS = "quadratic_intervals"
