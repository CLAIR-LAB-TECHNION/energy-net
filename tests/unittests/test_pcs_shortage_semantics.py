"""Shortage means uncovered load, not a post-consumption reserve requirement."""

import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.pcs_env import PCSEnv


def _make_env(tmp_path, demand):
    timestamps = pd.date_range("2025-01-01", periods=3 * 48, freq="30min")
    data_file = tmp_path / "consumption.csv"
    predictions_file = tmp_path / "predictions.csv"
    pd.DataFrame({
        "Datetime": timestamps,
        "Consumption": demand,
    }).to_csv(data_file, index=False)
    pd.DataFrame({
        "timestamp": timestamps,
        "predicted_consumption": demand,
    }).to_csv(predictions_file, index=False)
    return PCSEnv(
        test_data_file=str(data_file),
        predictions_file=str(predictions_file),
        shortage_penalty=7.0,
        include_shortage_penalty=True,
        verbosity=0,
    )


@pytest.mark.parametrize(
    "initial_storage, demand, action, uncovered",
    [
        pytest.param(8.0, 3.19168, 0.0, 0.0, id="ample-reserve-control"),
        pytest.param(5.0, 3.19168, 0.0, 0.0, id="report-counterexample"),
        pytest.param(3.19168, 3.19168, 0.0, 0.0, id="exactly-served"),
        pytest.param(2.0, 3.19168, 0.0, 1.19168, id="partial-shortfall"),
        pytest.param(0.0, 3.19168, 0.0, 3.19168, id="empty"),
        pytest.param(5.0, 3.0, 0.0, 0.0, id="five-units-serve-three"),
        pytest.param(3.0, 3.0, 0.0, 0.0, id="three-units-serve-three"),
        pytest.param(6.0, 3.0, 0.0, 0.0, id="remaining-equals-demand"),
        pytest.param(0.0, 0.0, 0.0, 0.0, id="no-demand-empty-battery"),
        pytest.param(0.0, 3.0, 3.0, 0.0, id="charging-covers-demand"),
        pytest.param(0.0, 3.0, 2.0, 1.0, id="charging-leaves-shortfall"),
        pytest.param(8.0, 3.0, -4.0, 0.0, id="selling-leaves-enough"),
        pytest.param(5.0, 3.0, -4.0, 2.0, id="selling-creates-shortfall"),
        pytest.param(0.5, 0.3, 0.0, 0.0, id="fractional-demand-fully-served"),
        pytest.param(2.999, 3.0, 0.0, 0.001, id="small-real-shortfall"),
        pytest.param(0.0, 0.00000001, 0.0, 0.00000001, id="any-positive-shortfall"),
    ],
)
def test_shortage_penalizes_only_uncovered_consumption(
    tmp_path, initial_storage, demand, action, uncovered
):
    env = _make_env(tmp_path, demand)
    penalty = env.shortage_penalty
    try:
        env.reset()
        # Reset initializes an empty battery and services demand; seed only afterward.
        battery = env.pcs.storage_units[0]
        battery.reset(initial_storage)

        obs, reward, _, _, info = env.step(np.array([action], dtype=np.float32))

        served = demand - uncovered
        expected_storage = initial_storage + action - served
        expected_shortage = uncovered > 0.0
        expected_penalty = penalty if expected_shortage else 0.0
        expected_money = (
            -info["buy_tariff"] * max(action + uncovered, 0.0)
            + info["sell_tariff"] * max(-(action + uncovered), 0.0)
        )
        assert env.pcs.get_production() == 0.0
        assert info["consumption_units"] == pytest.approx(demand)
        assert info["uncovered_consumption_units"] == pytest.approx(uncovered)
        assert info["battery_energy_used_for_consumption_units"] == pytest.approx(served)
        assert info["storage_before_units"] == pytest.approx(initial_storage)
        assert info["storage_after_units"] == pytest.approx(expected_storage)
        assert info["available_discharge_units"] == pytest.approx(expected_storage)
        assert info["energy_sold_or_bought_units"] == pytest.approx(action)
        assert info["step_money"] == pytest.approx(expected_money)
        assert np.isfinite(obs).all()
        assert (
            info["shortage"],
            reward,
            env.shortage_count,
            env.total_shortage_penalty,
        ) == (
            expected_shortage,
            pytest.approx(expected_money - expected_penalty),
            int(expected_shortage),
            expected_penalty,
        ), f"Served={served}; uncovered={uncovered}; post-service capacity={info['available_discharge_units']}"
        assert env.total_reward == pytest.approx(reward)
        assert env.total_money_earned == pytest.approx(expected_money)
    finally:
        env.close()


@pytest.mark.parametrize(
    "initial_storage, action, actual_action, expected_storage, uncovered",
    [
        pytest.param(99.0, 5.0, 1.0, 97.0, 0.0, id="charging-at-capacity"),
        pytest.param(1.0, -5.0, -1.0, 0.0, 3.0, id="selling-more-than-stored"),
    ],
)
def test_shortage_uses_actual_not_requested_battery_action(
    tmp_path, initial_storage, action, actual_action, expected_storage, uncovered
):
    env = _make_env(tmp_path, demand=3.0)
    try:
        env.reset()
        env.pcs.storage_units[0].reset(initial_storage)

        _, reward, _, _, info = env.step(np.array([action], dtype=np.float32))

        assert info["energy_sold_or_bought_units"] == pytest.approx(actual_action)
        assert info["storage_after_units"] == pytest.approx(expected_storage)
        assert info["uncovered_consumption_units"] == pytest.approx(uncovered)
        assert info["shortage"] == (uncovered > 0.0)
        penalty = env.shortage_penalty if uncovered > 0.0 else 0.0
        assert reward == pytest.approx(info["step_money"] - penalty)
    finally:
        env.close()


@pytest.mark.parametrize(
    "discharge_limit, expected_storage, expected_shortage",
    [
        pytest.param(2.0, 6.0, True, id="rate-too-low-despite-ample-storage"),
        pytest.param(3.0, 5.0, False, id="rate-exactly-covers-demand"),
    ],
)
def test_shortage_respects_discharge_rate(
    tmp_path, discharge_limit, expected_storage, expected_shortage
):
    env = _make_env(tmp_path, demand=3.0)
    try:
        env.reset()
        battery = env.pcs.storage_units[0]
        battery.reset(8.0)
        battery.discharge_rate_max = discharge_limit

        _, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))

        assert info["storage_after_units"] == pytest.approx(expected_storage)
        assert info["shortage"] == expected_shortage
        expected_penalty = env.shortage_penalty if expected_shortage else 0.0
        assert reward == pytest.approx(info["step_money"] - expected_penalty)
        assert env.total_shortage_penalty == pytest.approx(expected_penalty)
        assert env.shortage_count == int(expected_shortage)
    finally:
        env.close()


def test_only_actual_shortfalls_accumulate_penalties(tmp_path):
    env = _make_env(tmp_path, demand=3.0)
    try:
        env.reset()
        env.pcs.storage_units[0].reset(6.0)

        shortages = []
        rewards = []
        money = []
        storage_levels = []
        for _ in range(3):
            _, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
            shortages.append(info["shortage"])
            rewards.append(reward)
            money.append(info["step_money"])
            storage_levels.append(info["storage_after_units"])

        np.testing.assert_allclose(storage_levels, [3.0, 0.0, 0.0])
        assert (
            shortages,
            rewards,
            env.shortage_count,
            env.total_shortage_penalty,
            env.total_reward,
        ) == (
            [False, False, True],
            [money[0], money[1], money[2] - env.shortage_penalty],
            1,
            env.shortage_penalty,
            sum(money) - env.shortage_penalty,
        )

        env.reset()
        assert env.shortage_count == 0
        assert env.total_shortage_penalty == 0.0
        assert env.total_reward == 0.0
    finally:
        env.close()


@pytest.mark.parametrize(
    "initial_storage, demand",
    [
        pytest.param(5.0, 0.1, id="small-fractional-demand"),
        pytest.param(5.0, 1.95219, id="real-dataset-counterexample"),
        pytest.param(5.0, 0.3, id="decimal-demand"),
    ],
)
def test_floating_point_residual_does_not_trigger_shortage(
    tmp_path, initial_storage, demand
):
    """Numerical roundoff must not trigger a shortage penalty.

    When the battery fully serves demand, reconstructing the
    energy used from the storage difference may produce a tiny
    positive residual due to floating-point arithmetic.
    """
    env = _make_env(tmp_path, demand)
    try:
        env.reset()
        env.pcs.storage_units[0].reset(initial_storage)

        _, reward, _, _, info = env.step(
            np.array([0.0], dtype=np.float32)
        )

        # The battery has sufficient energy to serve all demand.
        assert info["storage_before_units"] == pytest.approx(
            initial_storage
        )
        assert info["battery_energy_used_for_consumption_units"] == (
            pytest.approx(demand)
        )
        assert info["storage_after_units"] == pytest.approx(
            initial_storage - demand
        )

        # Any remaining uncovered demand is numerical noise.
        assert 0.0 <= info["uncovered_consumption_units"] < 1e-12

        # Numerical noise must not cause an actual penalty.
        assert not info["shortage"]
        assert env.shortage_count == 0
        assert env.total_shortage_penalty == pytest.approx(0.0)

        # No battery transaction and no shortage means zero reward.
        assert info["step_money"] == pytest.approx(0.0)
        assert reward == pytest.approx(0.0)
        assert env.total_reward == pytest.approx(0.0)

    finally:
        env.close()
