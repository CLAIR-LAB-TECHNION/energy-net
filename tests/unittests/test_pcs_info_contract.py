
"""Tests for complete PCS transition diagnostics."""

import numpy as np
import pytest

from tests.unittests.test_pcs_pomdp_rewards import _env


@pytest.mark.parametrize("penalized", [False, True])
def test_info_exposes_reward_and_energy_decomposition(
    tmp_path, penalized
):
    env = _env(tmp_path, penalized=penalized)

    try:
        initial_obs, _ = env.reset(
            start_date="2025-01-02 00:00"
        )

        obs, reward, done, _, info = env.step(
            np.array([3.0], dtype=np.float32)
        )

        assert not done
        assert info["step_index"] == 0
        assert info["interval_start"] == (
            "2025-01-02T00:00:00"
        )

        assert info["storage_before_units"] == pytest.approx(0)
        assert info["storage_after_command_units"] == (
            pytest.approx(3)
        )
        assert info["storage_after_units"] == pytest.approx(0)

        assert info["consumption_units"] == pytest.approx(5)
        assert info[
            "battery_energy_used_for_consumption_units"
        ] == pytest.approx(3)
        assert info["uncovered_consumption_units"] == (
            pytest.approx(2)
        )

        assert info["grid_import_units"] == pytest.approx(5)
        assert info["grid_export_units"] == pytest.approx(0)

        assert info["electricity_cost"] == pytest.approx(1)
        assert info["step_money"] == pytest.approx(-1)
        assert info["step_reward"] == pytest.approx(reward)

        expected_penalty = 7.0 if penalized else 0.0

        assert info["shortage_penalty"] == pytest.approx(
            expected_penalty
        )
        assert info["total_reward_so_far"] == pytest.approx(
            -1.0 - expected_penalty
        )
        assert info["total_shortage_penalty_so_far"] == (
            pytest.approx(expected_penalty)
        )
        assert info["shortage_count_so_far"] == 1

        assert info["total_consumption_units"] == (
            pytest.approx(5)
        )
        assert info["total_battery_served_units"] == (
            pytest.approx(3)
        )
        assert info["total_grid_supplied_demand_units"] == (
            pytest.approx(2)
        )

        assert info["terminal_state"] is None

        assert info["observation_at_decision"] == pytest.approx(
            initial_obs.tolist()
        )

    finally:
        env.close()


@pytest.mark.parametrize("penalized", [False, True])
def test_final_info_preserves_terminal_state(
    tmp_path, penalized
):
    env = _env(tmp_path, penalized=penalized)

    try:
        obs, _ = env.reset(
            start_date="2025-01-02 00:00"
        )

        rewards = []
        money = []

        for i in range(48):
            action = 7.0 if i == 47 else 0.0
            if i == 47:
                final_decision_observation = obs.copy()
            obs, reward, done, truncated, info = env.step(
                np.array([action], dtype=np.float32)
            )

            rewards.append(reward)
            money.append(info["step_money"])
        # The terminal observation is zero-filled, but the
        # final decision observation remains available.
        assert np.all(obs == 0.0)

        assert info["observation_at_decision"] == pytest.approx(
            final_decision_observation.tolist()
        )

        assert info["terminal_state"]["storage_units"] == pytest.approx(
            env.pcs.get_total_storage()
        )
        assert done
        assert not truncated
        assert np.all(obs == 0.0)

        assert info["step_index"] == 47
        assert info["interval_start"] == (
            "2025-01-02T23:30:00"
        )
        assert info["interval_end"] == (
            "2025-01-03T00:00:00"
        )

        assert info["energy_sold_or_bought_units"] == (
            pytest.approx(7.0)
        )
        assert info["storage_after_units"] == pytest.approx(
            2.0
        )

        final_state = info["terminal_state"]
        assert final_state == info["state_after"]

        assert final_state["storage_units"] == pytest.approx(2)
        assert final_state["previous_load_units"] == (
            pytest.approx(5)
        )
        assert final_state["episode_step"] == 48
        assert final_state["available_discharge_units"] == (
            pytest.approx(2)
        )

        assert info["total_reward_so_far"] == pytest.approx(
            sum(rewards)
        )
        assert info["total_money_so_far"] == pytest.approx(
            sum(money)
        )

        assert info["total_consumption_units"] == (
            pytest.approx(48 * 5.0)
        )
        assert info["total_grid_import_units"] == (
            pytest.approx(47 * 5.0 + 7.0)
        )
        assert info["shortage_count_so_far"] == 47

        expected_penalty = 47 * 7.0 if penalized else 0.0

        assert info["total_shortage_penalty_so_far"] == (
            pytest.approx(expected_penalty)
        )

    finally:
        env.close()
