import os

import pandas as pd
import numpy as np
import pytest
from stable_baselines3 import PPO

from energy_net.gym_envs.alternating_env import AlternatingISOEnv
from energy_net.gym_envs.iso_env import ISOEnv
from energy_net.gym_envs.prosumer_env import ProsumerEnv


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEST_DATA_FILE = os.path.join(
    PROJECT_ROOT,
    'tests',
    'gym',
    'data_for_tests',
    'synthetic_household_consumption_test.csv',
)
PREDICTIONS_FILE = os.path.join(
    PROJECT_ROOT,
    'tests',
    'gym',
    'data_for_tests',
    'consumption_predictions.csv',
)


class ConstantProsumerModel:
    def predict(self, observation, deterministic=True):
        return np.zeros(1, dtype=np.float32), None


def create_model_linked_environment(use_asymmetric_pricing=False):
    prosumer_env = ProsumerEnv(
        load_data_file=TEST_DATA_FILE,
        load_forecast_file=PREDICTIONS_FILE,
        verbosity=0,
    )
    iso_env_kwargs = dict(
        actual_csv=TEST_DATA_FILE,
        predicted_csv=PREDICTIONS_FILE,
        prosumer_env=prosumer_env,
        prosumer_model=ConstantProsumerModel(),
        iso_verbosity=0,
        prosumer_verbosity=0,
    )
    if use_asymmetric_pricing:
        iso_env_kwargs["use_asymmetric_pricing"] = True
    iso_env = AlternatingISOEnv(**iso_env_kwargs)
    iso_model = PPO(
        "MlpPolicy",
        iso_env,
        verbose=0,
        n_steps=2,
        batch_size=2,
    )
    iso_env.iso_model = iso_model
    return iso_env


def test_model_linked_default_action_stays_symmetric():
    iso_env = create_model_linked_environment()
    assert iso_env.action_space.shape == (2 * iso_env.T,)

    price_values = np.full(iso_env.T, 0.1, dtype=np.float32)
    dispatch_values = np.full(iso_env.T, 0.9, dtype=np.float32)
    action = np.concatenate([price_values, dispatch_values])

    iso_env.reset()
    _, _, _, _, info = iso_env.step(action)

    strategy = iso_env.prosumer_env.simulator.price_strategy
    assert strategy.use_asymmetric_pricing is False
    np.testing.assert_allclose(
        strategy.calculate_buy_price(),
        np.full(iso_env.T, 0.02, dtype=np.float32),
    )
    np.testing.assert_allclose(
        strategy.calculate_sell_price(),
        np.full(iso_env.T, 0.02, dtype=np.float32),
    )
    np.testing.assert_allclose(info["dispatch"], dispatch_values)


def test_explicit_asymmetric_action_slices_all_three_blocks():
    iso_env = create_model_linked_environment(use_asymmetric_pricing=True)
    assert iso_env.action_space.shape == (3 * iso_env.T,)

    buy_values = np.full(iso_env.T, 0.1, dtype=np.float32)
    sell_values = np.full(iso_env.T, 0.4, dtype=np.float32)
    dispatch_values = np.full(iso_env.T, 0.9, dtype=np.float32)
    action = np.concatenate([buy_values, sell_values, dispatch_values])

    iso_env.reset()
    _, _, _, _, info = iso_env.step(action)

    strategy = iso_env.prosumer_env.simulator.price_strategy
    assert strategy.use_asymmetric_pricing is True
    np.testing.assert_allclose(
        strategy.calculate_buy_price(),
        np.full(iso_env.T, 0.02, dtype=np.float32),
    )
    np.testing.assert_allclose(
        strategy.calculate_sell_price(),
        np.full(iso_env.T, 0.08, dtype=np.float32),
    )
    np.testing.assert_allclose(info["dispatch"], dispatch_values)


def test_iso_env_asymmetric_dispatch_uses_final_block():
    iso_env = ISOEnv(
        actual_csv=TEST_DATA_FILE,
        predicted_csv=PREDICTIONS_FILE,
        use_asymmetric_pricing=True,
        dispatch_scale=2.0,
        verbosity=0,
    )
    iso_env.reset()

    buy_values = np.full(iso_env.T, 0.1, dtype=np.float32)
    sell_values = np.full(iso_env.T, 0.4, dtype=np.float32)
    dispatch_values = np.full(iso_env.T, 0.9, dtype=np.float32)
    action = np.concatenate([buy_values, sell_values, dispatch_values])

    _, _, _, _, info = iso_env.step(action)

    np.testing.assert_allclose(info["prices_raw"], buy_values)
    np.testing.assert_allclose(info["dispatch_raw"], dispatch_values)
    np.testing.assert_allclose(info["dispatch"], dispatch_values * 2.0)


@pytest.mark.parametrize("use_asymmetric_pricing", [False, True])
def test_action_length_must_match_configured_contract(use_asymmetric_pricing):
    iso_env = create_model_linked_environment(
        use_asymmetric_pricing=use_asymmetric_pricing
    )
    iso_env.reset()

    with pytest.raises(ValueError, match="action length"):
        iso_env.step(np.zeros(iso_env.action_space.shape[0] - 1, dtype=np.float32))


def test_alternating_reset_skips_missing_preceding_demand():
    iso_env = create_model_linked_environment()

    try:
        obs, reset_info = iso_env.reset()

        # First dataset day is not eligible for prosumer reset.
        assert reset_info["start_idx"] == 48
        assert reset_info["timestamp"] == pd.Timestamp(
            "2025-07-10 00:00:00"
        )

        # ISO observation must correspond to selected day.
        assert obs[0] == pytest.approx(
            iso_env.pred_data[48]
        )

        # The ISO's action must control that same day.
        action = np.zeros(
            iso_env.action_space.shape,
            dtype=np.float32,
        )

        _, _, terminated, truncated, info = iso_env.step(action)

        assert terminated
        assert not truncated

        assert info["realized_consumption"][0] == pytest.approx(
            iso_env.actual_data[48]
        )

        assert iso_env.prosumer_env.current_step == 48

    finally:
        iso_env.close()


def test_alternating_reset_skips_invalid_consumption_day():
    iso_env = create_model_linked_environment()

    try:
        # July 14 contains an invalid demand measurement.
        iso_env._next_start_idx = 5 * iso_env.T

        obs, info = iso_env.reset()

        # Skip the invalid day and use July 15 instead.
        assert info["start_idx"] == 6 * iso_env.T
        assert info["timestamp"] == pd.Timestamp(
            "2025-07-15 00:00:00"
        )

        assert obs[0] == pytest.approx(
            iso_env.pred_data[6 * iso_env.T]
        )

    finally:
        iso_env.close()
