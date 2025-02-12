from typing import Callable, Any, TypedDict
import numpy as np

# used to represent production price per produced unit
AmountPricePair = tuple[float, float]
Bid = list[AmountPricePair]
#ProductionPredFn = Callable[[Any, ...], Bid]
#ProductionFn = Callable[[Any, ...], AmountPricePair]


class Bounds:
    """
    Represents the bounds for observations or actions in the simulation environment.

    Attributes
    ----------
    low : Union[np.ndarray, List[float]]
        The lower bound of the space.
    high : Union[np.ndarray, List[float]]
        The upper bound of the space.
    dtype : type
        The data type of the bounds (e.g., `float`, `int`).
    shape : tuple
        The shape of the bound space.
    """
    def __init__(self, low: Any, high: Any, dtype: Any, shape: Any):
        self.low = low
        self.high = high
        self.dtype = dtype
        self.shape = shape
        
    def remove_first_dim(self):
        """
        Remove the first dimension from both `low` and `high`, and update `shape`.
        """
        if isinstance(self.low, np.ndarray) and isinstance(self.high, np.ndarray):
            self.low = self.low[1:]
            self.high = self.high[1:]
        elif isinstance(self.low, list) and isinstance(self.high, list):
            self.low = self.low[1:]
            self.high = self.high[1:]
        else:
            raise TypeError("Unsupported type for `low` and `high`. Must be list or np.ndarray.")
        
        if isinstance(self.shape, tuple):
            self.shape = self.shape[1:]
        else:
            raise TypeError("Unsupported type for `shape`. Must be tuple.")

      
