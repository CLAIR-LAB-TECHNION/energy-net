import os

import numpy as np
import pytest
from stable_baselines3 import PPO

from energy_net.gym_envs.alternating_env import AlternatingISOEnv
from energy_net.gym_envs.iso_env import ISOEnv
from energy_net.gym_envs.pcs_env import PCSEnv


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


class ConstantPCSModel:
    def predict(self, observation, deterministic=True):
        return np.zeros(1, dtype=np.float32), None


def create_model_linked_environment(use_asymmetric_pricing=False):
    pcs_env = PCSEnv(
        test_data_file=TEST_DATA_FILE,
        predictions_file=PREDICTIONS_FILE,
        verbosity=0,
    )
    iso_env_kwargs = dict(
        actual_csv=TEST_DATA_FILE,
        predicted_csv=PREDICTIONS_FILE,
        pcs_env=pcs_env,
        pcs_model=ConstantPCSModel(),
        iso_verbosity=0,
        pcs_verbosity=0,
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

    strategy = iso_env.pcs_env.price_strategy
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

    strategy = iso_env.pcs_env.price_strategy
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
