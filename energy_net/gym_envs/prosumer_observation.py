"""Encode a ProsumerObservation as a flat float32 vector with a named layout."""

from typing import Dict

import numpy as np
from gymnasium import spaces

from energy_net.simulation.contracts import ProsumerObservation

# Name -> width (H = forecast horizon). Order is the vector layout.
_FIELDS = (
    ("storage", 1),
    ("previous_load", 1),
    ("previous_generation", 1),
    ("buy_tariff", 1),
    ("sell_tariff", 1),
    ("load_forecast", "H"),
    ("generation_forecast", "H"),
    ("forecast_mask", "H"),
    ("available_charge", 1),
    ("available_discharge", 1),
    ("calendar", 7),
)


class ProsumerObservationEncoder:
    """
    Flat observation with a fixed, named layout (``layout[name]`` is a
    slice), so no code needs magic indices.

    Forecast positions beyond the valid horizon are zero and their mask is
    0. Storage is divided by ``storage_scale`` (the battery capacity); all
    other energies by ``energy_scale``.
    """

    def __init__(self, forecast_horizon: int, storage_scale: float,
                 energy_scale: float = 10.0):
        if not np.isfinite(storage_scale) or storage_scale <= 0:
            raise ValueError("storage_scale must be finite and positive")
        if energy_scale <= 0:
            raise ValueError("energy_scale must be positive")
        self.forecast_horizon = int(forecast_horizon)
        self.storage_scale = float(storage_scale)
        self.energy_scale = float(energy_scale)

        self.layout: Dict[str, slice] = {}
        position = 0
        for name, width in _FIELDS:
            width = self.forecast_horizon if width == "H" else width
            self.layout[name] = slice(position, position + width)
            position += width
        self.size = position
        self.space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.size,), dtype=np.float32)

    def encode(self, observation: ProsumerObservation) -> np.ndarray:
        # Computed in float64, cast once at the end (as the legacy env did).
        vector = np.zeros(self.size, dtype=float)
        energy = self.energy_scale
        valid = len(observation.load_forecast)
        if valid > self.forecast_horizon:
            raise ValueError("observation has more forecasts than the horizon")

        def put(name, value):
            vector[self.layout[name]] = value

        put("storage", observation.storage / self.storage_scale)
        put("previous_load", observation.previous_load / energy)
        put("previous_generation", observation.previous_generation / energy)
        put("buy_tariff", observation.buy_tariff)
        put("sell_tariff", observation.sell_tariff)
        start = self.layout["load_forecast"].start
        vector[start:start + valid] = observation.load_forecast / energy
        start = self.layout["generation_forecast"].start
        vector[start:start + valid] = observation.generation_forecast / energy
        start = self.layout["forecast_mask"].start
        vector[start:start + valid] = 1.0
        put("available_charge", observation.available_charge / energy)
        put("available_discharge", observation.available_discharge / energy)
        put("calendar", observation.calendar)
        return vector.astype(np.float32)

    def terminal(self) -> np.ndarray:
        """There is no decision after termination: an all-zero vector."""
        return np.zeros(self.size, dtype=np.float32)

    def field(self, vector: np.ndarray, name: str) -> np.ndarray:
        """The slice of ``vector`` holding ``name``."""
        return vector[self.layout[name]]
