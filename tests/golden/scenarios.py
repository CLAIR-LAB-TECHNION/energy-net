"""Shared, implementation-independent definitions of the golden scenarios.

A scenario is plain data: which data files to use, which pricing strategy,
the env options, and how episodes are started. The recorder replays these
against the legacy PCSEnv; the golden test replays the *recorded* actions
against the refactored ProsumerEnv and compares every transition.
"""

import os

import numpy as np

from energy_net.grid_entities.management.price_curve import (
    ActionBasedPriceStrategy,
    FixedPriceStrategy,
    PriceCurveStrategy,
    SineWavePriceStrategy,
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gym",
    "data_for_tests",
)

DEFAULT_LOAD = os.path.join(DATA_DIR, "synthetic_household_consumption_test.csv")
DEFAULT_FORECAST = os.path.join(DATA_DIR, "consumption_predictions.csv")
ZERO_LOAD = os.path.join(DATA_DIR, "zero_consumption_test.csv")
ZERO_FORECAST = os.path.join(DATA_DIR, "zero_consumption_predictions.csv")

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_JSON = os.path.join(FIXTURE_DIR, "legacy_pcs_trajectories.json.gz")
FIXTURE_NPZ = os.path.join(FIXTURE_DIR, "legacy_pcs_trajectories.npz")


class RecordingDataDrivenStrategy(PriceCurveStrategy):
    """Deterministic data-driven strategy that records every observation.

    Prices depend on the observation it receives, so any change in how the
    daily observation is built is visible in the tariffs (and therefore in
    rewards), not only in the recorded inputs.
    """

    requires_daily_observation = True
    price_max = 0.30

    def __init__(self):
        self.received = []

    def calculate_price(self, observation=None):
        observation = np.asarray(observation, dtype=float)
        self.received.append(observation.copy())
        demand = np.clip(observation[:48], 0.0, 10.0)
        return (0.05 + 0.02 * demand).astype(np.float32)


def _asymmetric_strategy():
    rng = np.random.default_rng(1234)
    iso_buy = rng.uniform(0.0, 0.5, 48)        # becomes the PCS sell tariff
    iso_sell = 0.5 + rng.uniform(0.0, 0.5, 48)  # becomes the PCS buy tariff
    return ActionBasedPriceStrategy(
        action=np.r_[iso_buy, iso_sell].astype(np.float32),
        price_min=0.0,
        price_max=0.20,
        use_asymmetric_pricing=True,
    )


def _saturation_actions(_rng, steps):
    pattern = [10.0] * 12 + [-10.0] * 12 + [0.0] * 12 + [10.0] * 12
    return np.array(pattern[:steps], dtype=np.float32)


def _uniform_actions(rng, steps):
    return rng.uniform(-10.0, 10.0, steps).astype(np.float32)


# Each scenario: env kwargs (shared vocabulary), strategy factory,
# list of episode starts (None = automatic reset), action generator, seed.
SCENARIOS = {
    "default_auto_reset": dict(
        load=DEFAULT_LOAD, forecast=DEFAULT_FORECAST,
        strategy=SineWavePriceStrategy, starts=[None, None, None],
        actions=_uniform_actions, seed=0, env_kwargs={},
    ),
    "saturation": dict(
        load=DEFAULT_LOAD, forecast=DEFAULT_FORECAST,
        strategy=SineWavePriceStrategy, starts=["2025-07-10 00:00:00"],
        actions=_saturation_actions, seed=1, env_kwargs={},
    ),
    "penalized_reward": dict(
        load=DEFAULT_LOAD, forecast=DEFAULT_FORECAST,
        strategy=SineWavePriceStrategy, starts=["2025-07-12 00:00:00"],
        actions=_uniform_actions, seed=2,
        env_kwargs=dict(include_shortage_penalty=True, shortage_penalty=1.0),
    ),
    "asymmetric_action_based": dict(
        load=DEFAULT_LOAD, forecast=DEFAULT_FORECAST,
        strategy=_asymmetric_strategy, starts=["2025-07-11 00:00:00"],
        actions=_uniform_actions, seed=3, env_kwargs={},
    ),
    "fixed_price": dict(
        load=DEFAULT_LOAD, forecast=DEFAULT_FORECAST,
        strategy=lambda: FixedPriceStrategy(fixed_price=0.12),
        starts=["2025-07-16 00:00:00", None],
        actions=_uniform_actions, seed=4, env_kwargs={},
    ),
    "data_driven_cross_midnight": dict(
        load=DEFAULT_LOAD, forecast=DEFAULT_FORECAST,
        strategy=RecordingDataDrivenStrategy,
        starts=["2025-07-11 12:00:00"],
        actions=_uniform_actions, seed=5, env_kwargs={},
    ),
    "zero_consumption": dict(
        load=ZERO_LOAD, forecast=ZERO_FORECAST,
        strategy=SineWavePriceStrategy, starts=[None, None],
        actions=_uniform_actions, seed=6, env_kwargs={},
    ),
    "short_horizon": dict(
        load=DEFAULT_LOAD, forecast=DEFAULT_FORECAST,
        strategy=SineWavePriceStrategy, starts=["2025-07-15 06:00:00"],
        actions=_uniform_actions, seed=7,
        env_kwargs=dict(prediction_horizon=12),
    ),
}


def legacy_layout(horizon):
    """Observation slices of the legacy 4 + 2H + 2 + 7 PCS observation."""
    return {
        "storage": (0, 1),
        "previous_load": (1, 2),
        "buy_tariff": (2, 3),
        "sell_tariff": (3, 4),
        "load_forecast": (4, 4 + horizon),
        "forecast_mask": (4 + horizon, 4 + 2 * horizon),
        "available_charge": (4 + 2 * horizon, 5 + 2 * horizon),
        "available_discharge": (5 + 2 * horizon, 6 + 2 * horizon),
        "calendar": (6 + 2 * horizon, 13 + 2 * horizon),
    }
