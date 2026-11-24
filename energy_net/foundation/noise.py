"""Noise models used to turn ground truth into noisy values for agents."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class NoiseModel(ABC):
    """Maps exact values to noisy values using a supplied random generator."""

    @abstractmethod
    def apply(self, values: np.ndarray, rng: np.random.Generator,
              lead_hours: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return a noisy copy of ``values``.

        ``lead_hours`` gives, per value, how far ahead of the issue time it
        lies. Models may ignore it; it exists so lead-dependent noise can be
        added without changing this interface.
        """


class NoNoise(NoiseModel):
    """Identity noise: returns an exact copy."""

    def apply(self, values, rng=None, lead_hours=None):
        return np.array(values, dtype=float, copy=True)


class GaussianNoise(NoiseModel):
    """
    Additive Gaussian noise.

    Args:
        std: Standard deviation, in value units, or as a fraction of each
            value's magnitude when ``relative`` is True.
        relative: Scale the deviation by ``abs(value)``.
        clip_min: Lower bound applied after adding noise (None disables it).
            Defaults to 0 because load and generation are nonnegative.
    """

    def __init__(self, std: float, relative: bool = False,
                 clip_min: Optional[float] = 0.0):
        if not np.isfinite(std) or std < 0:
            raise ValueError("std must be finite and nonnegative")
        self.std = float(std)
        self.relative = bool(relative)
        self.clip_min = clip_min

    def apply(self, values, rng, lead_hours=None):
        values = np.asarray(values, dtype=float)
        scale = self.std * np.abs(values) if self.relative else self.std
        noisy = values + rng.standard_normal(values.shape) * scale
        if self.clip_min is not None:
            noisy = np.maximum(noisy, self.clip_min)
        return noisy
