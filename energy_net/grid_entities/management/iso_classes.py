import numpy as np
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any
from energy_net.foundation.model import State, Action
from energy_net.management.utils import _validate_array_fields_same_shape_no_nans


@dataclass
class ISOState(State):
    prev_day_realized_demand:   np.ndarray = field(metadata={"validate_array": True})
    
    prev_day_forecast:          np.ndarray = field(metadata={"validate_array": True})
    prev_day_dispatch:          np.ndarray = field(metadata={"validate_array": True})
    prev_day_buy_price:         np.ndarray = field(metadata={"validate_array": True})
    prev_day_sell_price:        np.ndarray = field(metadata={"validate_array": True})

    day_ahead_forecast:         np.ndarray = field(metadata={"validate_array": True})
    day_ahead_dispatch:         np.ndarray = field(metadata={"validate_array": True})
    day_ahead_buy_price:        np.ndarray = field(metadata={"validate_array": True})
    day_ahead_sell_price:       np.ndarray = field(metadata={"validate_array": True})    

    def __post_init__(self) -> None:
        _validate_array_fields_same_shape_no_nans(self) 


@dataclass
class ISOAction(Action):
    day_ahead_dispatch:    np.ndarray = field(metadata={"validate_array": True})      
    day_ahead_buy_price:   np.ndarray = field(metadata={"validate_array": True})
    day_ahead_sell_price:  np.ndarray = field(metadata={"validate_array": True})

    def __post_init__(self) -> None:
        _validate_array_fields_same_shape_no_nans(self)
