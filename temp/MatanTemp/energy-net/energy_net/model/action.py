from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Type
import numpy as np


class EnergyAction(ABC):
    """
    Abstract base class for all energy actions.
    
    Serves as a marker for different types of energy actions within the smart grid simulation.
    """
    pass


@dataclass(frozen=True)
class StorageAction(EnergyAction):
    """
    Action representing the charging behavior of a storage device.
    
    Attributes
    ----------
    charge : float
        The amount of energy to charge, in kilowatts (kW). Positive values indicate charging.
    """
    charge: float = 0.0

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'StorageAction':
        """
        Create a StorageAction instance from a NumPy array.
        
        Parameters
        ----------
        arr : np.ndarray
            A NumPy array with a single float element representing the charge value.
        
        Returns
        -------
        StorageAction
            An instance of StorageAction with the specified charge.
        
        Raises
        ------
        ValueError
            If the input array does not contain exactly one element.
        """
        if arr.size != 1:
            raise ValueError(f"Input array must have exactly one element, got {arr.size}.")
        charge_value = float(arr[0])
        return cls(charge=charge_value)


@dataclass(frozen=True)
class ProduceAction(EnergyAction):
    """
    Action representing the production behavior of a generation unit.
    
    Attributes
    ----------
    production : float
        The amount of energy to produce, in kilowatts (kW). Positive values indicate production.
    """
    production: float = 0.0

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'ProduceAction':
        """
        Create a ProduceAction instance from a NumPy array.
        
        Parameters
        ----------
        arr : np.ndarray
            A NumPy array with a single float element representing the production value.
        
        Returns
        -------
        ProduceAction
            An instance of ProduceAction with the specified production.
        
        Raises
        ------
        ValueError
            If the input array does not contain exactly one element.
        """
        if arr.size != 1:
            raise ValueError(f"Input array must have exactly one element, got {arr.size}.")
        production_value = float(arr[0])
        return cls(production=production_value)


@dataclass(frozen=True)
class ConsumeAction(EnergyAction):
    """
    Action representing the consumption behavior of a load.
    
    Attributes
    ----------
    consumption : float
        The amount of energy to consume, in kilowatts (kW). Positive values indicate consumption.
    """
    consumption: float = 0.0

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'ConsumeAction':
        """
        Create a ConsumeAction instance from a NumPy array.
        
        Parameters
        ----------
        arr : np.ndarray
            A NumPy array with a single float element representing the consumption value.
        
        Returns
        -------
        ConsumeAction
            An instance of ConsumeAction with the specified consumption.
        
        Raises
        ------
        ValueError
            If the input array does not contain exactly one element.
        """
        if arr.size != 1:
            raise ValueError(f"Input array must have exactly one element, got {arr.size}.")
        consumption_value = float(arr[0])
        return cls(consumption=consumption_value)


@dataclass(frozen=True)
class TradeAction(EnergyAction):
    """
    Action representing the trading behavior within the energy market.
    
    Attributes
    ----------
    amount : float
        The amount of energy to trade, in kilowatts (kW). Positive values indicate selling,
        and negative values indicate buying.
    """
    amount: float = 0.0

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'TradeAction':
        """
        Create a TradeAction instance from a NumPy array.
        
        Parameters
        ----------
        arr : np.ndarray
            A NumPy array with a single float element representing the trade amount.
        
        Returns
        -------
        TradeAction
            An instance of TradeAction with the specified amount.
        
        Raises
        ------
        ValueError
            If the input array does not contain exactly one element.
        """
        if arr.size != 1:
            raise ValueError(f"Input array must have exactly one element, got {arr.size}.")
        amount_value = float(arr[0])
        return cls(amount=amount_value)
