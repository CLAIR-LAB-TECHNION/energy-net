"""
End-to-end equivalence: ProsumerEnv (with a zero-output generator) must
reproduce the legacy PCSEnv trajectories recorded in tests/golden.
"""

import numpy as np
import pytest

from energy_net.gym_envs.prosumer_env import ProsumerEnv
from tests.golden.fixture import load_fixture
from tests.golden.scenarios import (
    SCENARIOS,
    RecordingDataDrivenStrategy,
    use_legacy_daily_observation,
)

# Info keys that echoed reward *configuration* rather than step data; the
# reward function itself is available as env.reward_fn.
REMOVED_KEYS = {"include_shortage_penalty", "configured_shortage_penalty"}
# Values the legacy env decoded back from its float32 observation; the new
# values are exact, so they differ only by float32 rounding.
DECODED_KEYS = {"forecast_at_decision_units", "calendar_features_at_decision"}


def _env(spec, strategy):
    options = dict(spec["env_kwargs"])
    if "prediction_horizon" in options:
        options["forecast_horizon"] = options.pop("prediction_horizon")
    return ProsumerEnv(
        load_data_file=spec["load"],
        load_forecast_file=spec["forecast"],
        price_strategy=strategy,
        verbosity=0,
        **options,
    )


def _assert_legacy_channels(env, new_obs, legacy_obs, layout):
    for name, (start, stop) in layout.items():
        np.testing.assert_array_equal(
            env.encoder.field(new_obs, name), legacy_obs[start:stop], err_msg=name
        )
    # Channels the legacy env did not have: a zero generator.
    assert not env.encoder.field(new_obs, "previous_generation").any()
    assert not env.encoder.field(new_obs, "generation_forecast").any()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_prosumer_env_reproduces_legacy_pcs_env(name):
    metadata, arrays = load_fixture()
    spec = SCENARIOS[name]
    recorded = metadata["scenarios"][name]
    layout = recorded["layout"]
    strategy = spec["strategy"]()
    env = _env(spec, strategy)
    use_legacy_daily_observation(env.simulator)
    assert env.max_steps == recorded["max_steps"]

    for index, (start, episode) in enumerate(zip(spec["starts"], recorded["episodes"])):
        legacy_obs = arrays[f"{name}/{index}"]
        options = None if start is None else {"start_date": start}
        obs, reset_info = env.reset(options=options)
        assert reset_info == {}
        assert env.current_datetime.isoformat() == episode["start"]
        _assert_legacy_channels(env, obs, legacy_obs[0], layout)

        for k, step in enumerate(episode["steps"]):
            decision = obs
            obs, reward, terminated, truncated, info = env.step(
                np.array([step["action"]], dtype=np.float32)
            )
            assert reward == step["reward"]
            assert (terminated, truncated) == (step["terminated"], step["truncated"])
            if terminated:
                assert not obs.any() and obs.shape == env.observation_space.shape
            else:
                _assert_legacy_channels(env, obs, legacy_obs[k + 1], layout)
            assert info["observation_at_decision"] == decision.tolist()

            missing = set(step["info"]) - set(info) - REMOVED_KEYS
            assert not missing, missing
            for key, expected in step["info"].items():
                if key in REMOVED_KEYS:
                    continue
                if key in DECODED_KEYS:
                    if isinstance(expected, dict):
                        for column, value in expected.items():
                            assert info[key][column] == pytest.approx(value, abs=1e-6)
                    else:
                        assert info[key] == pytest.approx(expected, rel=1e-6)
                elif isinstance(expected, dict):
                    assert {k2: info[key][k2] for k2 in expected} == expected, key
                else:
                    assert info[key] == expected, key

    if isinstance(strategy, RecordingDataDrivenStrategy):
        np.testing.assert_array_equal(
            np.stack(strategy.received), arrays[f"{name}/strategy_inputs"]
        )
    env.close()
