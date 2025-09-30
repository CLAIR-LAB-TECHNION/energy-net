from dataclasses import dataclass
from enum import Enum
import numpy as np
from energy_net.defs import (
    ACTION_TYPE_STORAGE,
    ACTION_TYPE_PRODUCTION,
    ACTION_TYPE_CONSUMPTION,
    ACTION_TYPE_TRADE
)


class ActionType(Enum):
    """
    Enumeration of action types for grid entities.
    """
    STORAGE = ACTION_TYPE_STORAGE
    PRODUCTION = ACTION_TYPE_PRODUCTION
    CONSUMPTION = ACTION_TYPE_CONSUMPTION
    TRADE = ACTION_TYPE_TRADE


@dataclass(frozen=True)
class Action:
    """
    Polymorphic action dataclass representing any energy grid action.
    
    Attributes
    ----------
    id : ActionType
        The type of action being performed.
    amount : float
        The amount of energy involved in the action, in kilowatts (kW).
        - For storage: positive values indicate charging, negative indicate discharging
        - For production: positive values indicate production amount
        - For consumption: positive values indicate consumption amount  
        - For trade: positive values indicate selling, negative indicate buying
    """
    id: ActionType
    amount: float = 0.0

    @classmethod
    def from_numpy(cls, action_type: ActionType, arr: np.ndarray) -> 'Action':
        """
        Create an Action instance from a NumPy array.
        
        Parameters
        ----------
        action_type : ActionType
            The type of action to create.
        arr : np.ndarray
            A NumPy array with a single float element representing the action amount.
        
        Returns
        -------
        Action
            An instance of Action with the specified type and amount.
        
        Raises
        ------
        ValueError
            If the input array does not contain exactly one element.
        """
        if arr.size != 1:
            raise ValueError(f"Input array must have exactly one element, got {arr.size}.")
        amount_value = float(arr[0])
        return cls(id=action_type, amount=amount_value)

    @classmethod
    def storage(cls, amount: float = 0.0) -> 'Action':
        """
        Create a storage action.
        
        Parameters
        ----------
        amount : float
            Charge amount (positive for charging, negative for discharging).
            
        Returns
        -------
        Action
            Storage action instance.
        """
        return cls(id=ActionType.STORAGE, amount=amount)

    @classmethod
    def production(cls, amount: float = 0.0) -> 'Action':
        """
        Create a production action.
        
        Parameters
        ----------
        amount : float
            Production amount (positive values).
            
        Returns
        -------
        Action
            Production action instance.
        """
        return cls(id=ActionType.PRODUCTION, amount=amount)

    @classmethod
    def consumption(cls, amount: float = 0.0) -> 'Action':
        """
        Create a consumption action.
        
        Parameters
        ----------
        amount : float
            Consumption amount (positive values).
            
        Returns
        -------
        Action
            Consumption action instance.
        """
        return cls(id=ActionType.CONSUMPTION, amount=amount)

    @classmethod
    def trade(cls, amount: float = 0.0) -> 'Action':
        """
        Create a trade action.
        
        Parameters
        ----------
        amount : float
            Trade amount (positive for selling, negative for buying).
            
        Returns
        -------
        Action
            Trade action instance.
        """
        return cls(id=ActionType.TRADE, amount=amount)
