# dynamics/energy_dynamics.py

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd


class EnergyDynamics(ABC):
    """
    Abstract Base Class for defining energy dynamics.
    Represents the behavior of a component in the PCSUnit.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state of the dynamics.
        Called at the beginning of each new episode.
        """
        pass

    @abstractmethod
    def get_value(self, **kwargs) -> Any:
        """
        Retrieves the current value based on provided arguments.

        Args:
            **kwargs: Arbitrary keyword arguments specific to the dynamic.

        Returns:
            Any: The current value based on the dynamics.
        """
        pass


class GroundTruthUnavailableError(ValueError):
    """Raised when ground truth is requested for a time it does not cover."""


def as_datetime_index(times) -> pd.DatetimeIndex:
    """Normalize a timestamp, a string, or an iterable of them to an index."""
    if isinstance(times, pd.DatetimeIndex):
        return times
    if isinstance(times, (str, pd.Timestamp)) or not hasattr(times, "__iter__"):
        return pd.DatetimeIndex([pd.Timestamp(times)])
    return pd.DatetimeIndex(times)


class GroundTruthDynamics(EnergyDynamics):
    """
    Dynamics of a quantity that no agent controls, such as household demand
    or PV output. It answers what *actually* happens; it is not a prediction.

    It has two roles:

    1. ``ground_truth(times)``: the value the simulation will realize at the
       requested times. The answer must be deterministic and idempotent for
       the lifetime of the instance ("given it was calculated"): data-backed
       values are fixed, and computed values must be memoized or derived
       from seeded counters. A value queried ahead of time is therefore
       exactly the value the simulator later realizes.
    2. ``sample(times, rng, noise)``: ground truth plus noise, for agents.

    Predictions are a separate concern; see ``foundation.forecaster``.
    """

    def __init__(self, quantity: str = "value"):
        self.quantity = quantity

    @abstractmethod
    def ground_truth(self, times) -> np.ndarray:
        """
        Return the realized values at ``times``.

        Raises:
            GroundTruthUnavailableError: for times without a valid value.
        """

    def availability_error(self, times, context: Optional[str] = None) -> Optional[str]:
        """
        Describe why ground truth at ``times`` cannot be used, or return None.

        Values must exist, be finite and be nonnegative. ``context`` is
        inserted before the quantity name in messages, for example
        ``"preceding/episode"``.
        """
        label = f"{context} {self.quantity}" if context else self.quantity
        try:
            values = self.ground_truth(as_datetime_index(times))
        except GroundTruthUnavailableError as exc:
            return str(exc)
        if not np.isfinite(values).all() or (values < 0).any():
            return f"Non-finite or negative {label}"
        return None

    def support(self) -> Optional[pd.DatetimeIndex]:
        """Timestamps where ground truth is defined, or None if unbounded."""
        return None

    def sample(self, times, rng: np.random.Generator, noise,
               lead_hours: Optional[np.ndarray] = None) -> np.ndarray:
        """Return ground truth plus noise drawn from ``noise`` using ``rng``."""
        truth = self.ground_truth(as_datetime_index(times))
        return noise.apply(truth, rng=rng, lead_hours=lead_hours)

    def get_value(self, **kwargs) -> float:
        """Bridge to the entity API: requires a ``timestamp`` keyword."""
        timestamp = kwargs.get("timestamp")
        if timestamp is None:
            raise ValueError(
                f"{type(self).__name__} requires a 'timestamp' to "
                "provide ground truth"
            )
        return float(self.ground_truth(as_datetime_index(timestamp))[0])

    def reset(self) -> None:
        """Ground truth does not depend on episodes; nothing to reset."""
