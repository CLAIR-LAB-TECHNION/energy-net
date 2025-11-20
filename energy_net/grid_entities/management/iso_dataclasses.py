from dataclasses import dataclass, fields, is_dataclass
from typing import Iterable, Any, Tuple, TypeAlias
import numpy as np
from numpy.typing import NDArray, DTypeLike


# ---------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------
Array: TypeAlias = NDArray[np.float64]


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def convert_fields_to_arrays(instance: Any, dtype: DTypeLike = float) -> None:
    """
    Convert all dataclass fields into numpy arrays with the given dtype.
    Mutates the instance in place.

    This function assumes *all* fields in the instance are intended to be arrays.
    """
    if not is_dataclass(instance):
        raise TypeError(
            f"convert_fields_to_arrays expects a dataclass instance, "
            f"got {type(instance).__name__!r}."
        )

    for f in fields(instance):
        value = getattr(instance, f.name)
        try:
            arr = np.asarray(value, dtype=dtype)
        except Exception as e:
            raise TypeError(
                f"Field '{f.name}' could not be converted to an ndarray: {e}"
            ) from e

        setattr(instance, f.name, arr)


def iter_array_fields(instance: Any) -> Iterable[Tuple[str, Array]]:
    """
    Yield (field_name, array) for each field in a dataclass instance.

    Raises informative errors when fields are not arrays.
    """
    if not is_dataclass(instance):
        raise TypeError(
            f"iter_array_fields expects a dataclass instance, "
            f"got {type(instance).__name__!r}."
        )

    for f in fields(instance):
        value = getattr(instance, f.name)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Field '{f.name}' on {instance.__class__.__name__} "
                f"is not an ndarray. Ensure convert_fields_to_arrays(...) "
                f"is called first."
            )
        yield f.name, value


def validate_no_nans(instance: Any) -> None:
    """
    Ensure that no field arrays contain NaN values.
    """
    for name, arr in iter_array_fields(instance):
        if np.isnan(arr).any():
            raise ValueError(
                f"Field '{name}' on {instance.__class__.__name__} contains NaN values."
            )


def validate_same_shape(instance: Any) -> None:
    """
    Ensure that all field arrays have identical shapes.
    """
    arrays = list(iter_array_fields(instance))
    if not arrays:  # No fields -> nothing to validate
        return

    ref_name, ref_arr = arrays[0]
    ref_shape = ref_arr.shape

    for name, arr in arrays[1:]:
        if arr.shape != ref_shape:
            raise ValueError(
                f"All arrays in {instance.__class__.__name__} must share the same shape.\n"
                f"  '{ref_name}': {ref_shape}\n"
                f"  '{name}':    {arr.shape}"
            )


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@dataclass
class ISOState:
    """
    Full ISO-level daily state:
    - previous day's realized data
    - new day-ahead forecasts and plan
    """

    prev_day_realized:      Array
    prev_day_dispatch:      Array
    prev_day_pricing_plan:  Array
    day_ahead_forecast:     Array
    day_ahead_dispatch:     Array
    day_ahead_pricing_plan: Array

    def __post_init__(self) -> None:
        convert_fields_to_arrays(self, dtype=float)
        validate_no_nans(self)
        validate_same_shape(self)


@dataclass
class ISOAction:
    """
    ISO action for a single day-ahead market:
    - dispatch vector
    - pricing plan vector
    """

    day_ahead_dispatch:     Array
    day_ahead_pricing_plan: Array

    def __post_init__(self) -> None:
        convert_fields_to_arrays(self, dtype=float)
        validate_no_nans(self)
        validate_same_shape(self)
