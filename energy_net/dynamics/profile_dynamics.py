"""Synthetic ground truth: daily profiles and constants."""

import numpy as np
import pandas as pd

from energy_net.foundation.dynamics import GroundTruthDynamics, as_datetime_index


class ProfileDynamics(GroundTruthDynamics):
    """
    Adapter giving timestamp-based ground truth from a daily profile.

    ``profile`` is any object with ``get_value(time=fraction_of_day)``, such
    as the existing Constant, DoublePeak, LinearGrowth or GMM consumption and
    production dynamics. Each timestamp is converted to its fraction of the
    day (00:00 -> 0.0, 12:00 -> 0.5) before the profile is evaluated.
    """

    def __init__(self, profile, quantity: str = "value"):
        super().__init__(quantity)
        if not callable(getattr(profile, "get_value", None)):
            raise TypeError("profile must provide get_value(time=...)")
        self.profile = profile

    def ground_truth(self, times) -> np.ndarray:
        index = as_datetime_index(times)
        fractions = (index - index.normalize()) / pd.Timedelta(days=1)
        return np.array(
            [float(self.profile.get_value(time=float(f))) for f in fractions],
            dtype=float,
        )


class ConstantDynamics(GroundTruthDynamics):
    """The same value at every time, e.g. a zero-output generator."""

    def __init__(self, value: float = 0.0, quantity: str = "value"):
        super().__init__(quantity)
        self.value = float(value)

    def ground_truth(self, times) -> np.ndarray:
        return np.full(len(as_datetime_index(times)), self.value, dtype=float)
