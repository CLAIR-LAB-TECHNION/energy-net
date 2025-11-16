# iso_base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable


class ISOBase(ABC):
    """
    Base class for Independent System Operator (ISO) implementations.
    
    All ISO implementations must inherit from this class and implement
    the get_pricing_function method.
    """

    @abstractmethod
    def get_pricing_function(self, state: Dict[str, Any]) -> Callable[[float], float]:
        """
        Returns a pricing function based on the current state.
        
        Args:
            state (Dict[str, Any]): Current state of the system
            
        Returns:
            Callable[[float], float]: Function that takes demand and returns price
        """
        pass


