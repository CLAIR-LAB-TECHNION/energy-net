import numpy as np
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


def _ensure_same_shape(arr1: np.ndarray, arr2: np.ndarray) -> None:
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        raise TypeError("both arrays must be numpy arrays!")
    if arr1.shape != arr2.shape:
        raise ValueError(f"shape mismatch: {arr1.shape} vs {arr2.shape}")


def _ensure_no_nans(arr: np.ndarray) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError("array must be a numpy array!")
    if not np.isfinite(arr).all():  
        raise ValueError("array contains NaN or inf")


def _validate_array_fields_same_shape_no_nans(
    obj: Any,
    meta_key: str = "validate_array",
) -> None:    
    if not is_dataclass(obj):
        raise TypeError(f"_validate_array_fields_same_shape_no_nans expects a dataclass instance, "
                        f"got {type(obj)!r}")
    
    array_fields: list[tuple[str, np.ndarray]] = []
    for f in fields(obj):
        if not f.metadata.get(meta_key, False):
            continue

        value = getattr(obj, f.name)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Field '{f.name}' is marked {meta_key}=True but is not a numpy array, "
                f"got {type(value)!r}."
            )
        array_fields.append((f.name, value))

    if not array_fields:
        return
        
    reference_name, reference_arr = array_fields[0]
    for name, arr in array_fields[1:]:
        try:
            _ensure_same_shape(reference_arr, arr)
        except Exception as exc:
            raise type(exc)(
                f"Shape validation failed between '{reference_name}' and '{name}': {exc}") from exc 

    for name, arr in array_fields:
        try:
            _ensure_no_nans(arr)
        except Exception as exc:
            raise type(exc)(f"NaN/inf validation failed for field '{name}': {exc}") from exc           



@dataclass
class ISOState:
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
class ISOAction:
    day_ahead_dispatch:    np.ndarray = field(metadata={"validate_array": True})      
    day_ahead_buy_price:   np.ndarray = field(metadata={"validate_array": True})
    day_ahead_sell_price:  np.ndarray = field(metadata={"validate_array": True})

    def __post_init__(self) -> None:
        _validate_array_fields_same_shape_no_nans(self)
