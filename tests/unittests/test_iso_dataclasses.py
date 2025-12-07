import numpy as np
import pytest

from energy_net.grid_entities.management.iso_classes import ISOState, ISOAction


def test_iso_state_accepts_valid_arrays():
    arr = np.ones(24)
    state = ISOState(
        prev_day_realized_demand=arr,
        prev_day_forecast=arr,
        prev_day_dispatch=arr,
        prev_day_buy_price=arr,
        prev_day_sell_price=arr,
        day_ahead_forecast=arr,
        day_ahead_dispatch=arr,
        day_ahead_buy_price=arr,
        day_ahead_sell_price=arr,
    )

    assert isinstance(state, ISOState)
    assert state.prev_day_realized_demand.shape == (24,)


def test_iso_state_rejects_shape_mismatch():
    arr24 = np.ones(24)
    arr12 = np.ones(12)

    with pytest.raises(ValueError):
        ISOState(
            prev_day_realized_demand=arr24,
            prev_day_forecast=arr12,  # mismatch
            prev_day_dispatch=arr24,
            prev_day_buy_price=arr24,
            prev_day_sell_price=arr24,
            day_ahead_forecast=arr24,
            day_ahead_dispatch=arr24,
            day_ahead_buy_price=arr24,
            day_ahead_sell_price=arr24,
        )


def test_iso_state_rejects_nans():
    clean = np.ones(24)
    bad = clean.copy()
    bad[0] = np.nan

    with pytest.raises(ValueError):
        ISOState(
            prev_day_realized_demand=bad,  # NaN here
            prev_day_forecast=clean,
            prev_day_dispatch=clean,
            prev_day_buy_price=clean,
            prev_day_sell_price=clean,
            day_ahead_forecast=clean,
            day_ahead_dispatch=clean,
            day_ahead_buy_price=clean,
            day_ahead_sell_price=clean,
        )


def test_iso_action_accepts_valid_arrays():
    arr = np.linspace(0, 1, 24)
    action = ISOAction(
        day_ahead_dispatch=arr,
        day_ahead_buy_price=arr,
        day_ahead_sell_price=arr,
    )

    assert isinstance(action, ISOAction)
    assert action.day_ahead_dispatch.shape == (24,)


def test_iso_action_rejects_shape_mismatch():
    arr24 = np.ones(24)
    arr12 = np.ones(12)

    with pytest.raises(ValueError):
        ISOAction(
            day_ahead_dispatch=arr24,
            day_ahead_buy_price=arr12,  # mismatch
            day_ahead_sell_price=arr24,
        )


def test_iso_action_rejects_nans():
    clean = np.ones(24)
    bad = clean.copy()
    bad[-1] = np.inf

    with pytest.raises(ValueError):
        ISOAction(
            day_ahead_dispatch=clean,
            day_ahead_buy_price=bad,  # inf here
            day_ahead_sell_price=clean,
        )
