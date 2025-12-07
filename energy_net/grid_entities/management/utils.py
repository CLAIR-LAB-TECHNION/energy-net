import numpy as np
from typing import Mapping


def _ensure_same_shape(arr1: np.ndarray, arr2: np.ndarray) -> None:
    """Raise if arr1 and arr2 do not have identical shapes."""
    if arr1.shape != arr2.shape:
        raise ValueError(f"shape mismatch: {arr1.shape} vs {arr2.shape}")


def _ensure_no_nans(arr: np.ndarray) -> None:
    """Raise if arr contains NaN or inf values."""
    if not np.isfinite(arr).all():
        raise ValueError("array contains NaN or inf")


def validate_named_arrays_same_shape_no_nans(arrays: Mapping[str, object]) -> None:
    """
    Validate only the fields that are NumPy arrays.

    For each value in `arrays`:
        - If it is a numpy array, it is validated.
        - Non-array values are ignored entirely.

    Validation performed:
        - All numpy arrays must have identical shapes.
        - Arrays must not contain NaN or inf.
    """

    # Extract only numpy array fields
    array_items = {
        name: arr for name, arr in arrays.items()
        if isinstance(arr, np.ndarray)
    }

    if not array_items:
        return

    ref_name, ref_arr = next(iter(array_items.items()))

    for name, arr in array_items.items():
        _ensure_same_shape(ref_arr, arr)

    for name, arr in array_items.items():
        _ensure_no_nans(arr)
