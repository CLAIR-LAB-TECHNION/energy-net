import numpy as np
import pytest

from energy_net.grid_entities.management.iso_classes import ISOState, ISOAction
from energy_net.grid_entities.management.iso_dynamics import iso_day_to_day_transition


def _make_state_and_action(n: int = 3):
    state = ISOState(
        prev_day_realized_demand=np.zeros(n),
        prev_day_forecast=np.full(n, 10.0),
        prev_day_dispatch=np.full(n, 11.0),
        prev_day_buy_price=np.full(n, 1.0),
        prev_day_sell_price=np.full(n, 2.0),
        day_ahead_forecast=np.full(n, 20.0),
        day_ahead_dispatch=np.full(n, 21.0),
        day_ahead_buy_price=np.full(n, 3.0),
        day_ahead_sell_price=np.full(n, 4.0),
    )

    action = ISOAction(
        day_ahead_dispatch=np.full(n, 31.0),
        day_ahead_buy_price=np.full(n, 5.0),
        day_ahead_sell_price=np.full(n, 6.0),
    )

    return state, action


def test_transition_updates_state_fields_correctly():
    state, action = _make_state_and_action(3)

    new_forecast = np.array([100.0, 101.0, 102.0])
    new_realized = np.array([7.0, 8.0, 9.0])

    new_state, info = iso_day_to_day_transition(
        state=state,
        action=action,
        day_ahead_forecast=new_forecast,
        prev_day_realized_demand=new_realized,
    )

    # prev_day_* in new_state
    np.testing.assert_array_equal(
        new_state.get_attribute("prev_day_realized_demand"),
        new_realized,
    )
    np.testing.assert_array_equal(
        new_state.get_attribute("prev_day_forecast"),
        state.get_attribute("day_ahead_forecast"),
    )
    np.testing.assert_array_equal(
        new_state.get_attribute("prev_day_dispatch"),
        state.get_attribute("day_ahead_dispatch"),
    )
    np.testing.assert_array_equal(
        new_state.get_attribute("prev_day_buy_price"),
        state.get_attribute("day_ahead_buy_price"),
    )
    np.testing.assert_array_equal(
        new_state.get_attribute("prev_day_sell_price"),
        state.get_attribute("day_ahead_sell_price"),
    )

    # day_ahead_* in new_state
    np.testing.assert_array_equal(
        new_state.get_attribute("day_ahead_forecast"),
        new_forecast,
    )
    np.testing.assert_array_equal(
        new_state.get_attribute("day_ahead_dispatch"),
        action.get_action("day_ahead_dispatch"),
    )
    np.testing.assert_array_equal(
        new_state.get_attribute("day_ahead_buy_price"),
        action.get_action("day_ahead_buy_price"),
    )
    np.testing.assert_array_equal(
        new_state.get_attribute("day_ahead_sell_price"),
        action.get_action("day_ahead_sell_price"),
    )

    # original state unchanged
    np.testing.assert_array_equal(
        state.get_attribute("prev_day_realized_demand"),
        np.zeros(3),
    )


def test_transition_info_dict_mismatches():
    state, action = _make_state_and_action(3)

    prev_realized = np.array([1.0, 2.0, 3.0])
    forecast = np.array([50.0, 51.0, 52.0])

    _, info = iso_day_to_day_transition(
        state=state,
        action=action,
        day_ahead_forecast=forecast,
        prev_day_realized_demand=prev_realized,
    )

    # prev_day_forecast inside iso_day_to_day_transition is taken from
    # state.get_attribute("day_ahead_forecast")
    expected_forecast = state.get_attribute("day_ahead_forecast")
    expected_dispatch = state.get_attribute("day_ahead_dispatch")

    np.testing.assert_array_equal(
        info["forecast_mismatch"],
        expected_forecast - prev_realized,
    )
    np.testing.assert_array_equal(
        info["dispatch_mismatch"],
        expected_dispatch - prev_realized,
    )


def test_transition_fails_on_bad_prev_realized_shape():
    state, action = _make_state_and_action(3)

    bad_prev_realized = np.ones(5)  # wrong shape

    with pytest.raises(ValueError):
        iso_day_to_day_transition(
            state=state,
            action=action,
            day_ahead_forecast=np.array([10.0, 11.0, 12.0]),
            prev_day_realized_demand=bad_prev_realized,
        )
