"""
Prediction interface for grid entities.

A forecaster is a belief about future values of a quantity. Any algorithm
can sit behind it: a precomputed table, a trained ML predictor, the
entity's own model-based or model-free RL, or noisy samples of the ground
truth. Only ``forecast`` is required; the other hooks are optional and are
called through the helpers below, so plain objects work without
subclassing.
"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Forecaster(Protocol):
    """Structural interface: anything with this ``forecast`` method works."""

    def forecast(self, issue_time, target_times) -> np.ndarray:
        """
        Predict values at ``target_times`` using only information available
        at ``issue_time``.
        """


class BaseForecaster(ABC):
    """Optional base class documenting every hook with a default."""

    @abstractmethod
    def forecast(self, issue_time, target_times) -> np.ndarray:
        """Predict values at ``target_times`` as of ``issue_time``."""

    def availability_error(self, issue_time, target_times) -> Optional[str]:
        """Describe why a forecast cannot be produced, or return None."""
        return None

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Called at the start of every episode."""

    def observe(self, times, realized: np.ndarray) -> None:
        """
        Online-learning hook. After each step the simulator passes only the
        values that were just realized, never future ones.
        """


def validate_forecaster(forecaster) -> None:
    """Raise TypeError unless ``forecaster`` has a callable ``forecast``."""
    if not callable(getattr(forecaster, "forecast", None)):
        raise TypeError(
            "A forecaster must provide a callable "
            "forecast(issue_time, target_times)"
        )


def forecast_availability_error(forecaster, issue_time, target_times) -> Optional[str]:
    """Call the optional ``availability_error`` hook if it exists."""
    hook = getattr(forecaster, "availability_error", None)
    return hook(issue_time, target_times) if callable(hook) else None


def reset_forecaster(forecaster, rng: Optional[np.random.Generator]) -> None:
    """Call the optional ``reset`` hook if it exists."""
    hook = getattr(forecaster, "reset", None)
    if callable(hook):
        hook(rng)


def notify_forecaster(forecaster, times, realized) -> None:
    """Call the optional ``observe`` hook if it exists."""
    hook = getattr(forecaster, "observe", None)
    if callable(hook):
        hook(times, realized)
