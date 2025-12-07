import numpy as np
from .iso_classes import ISOState, ISOAction


def iso_day_to_day_transition(
    state: ISOState,
    action: ISOAction,
    day_ahead_forecast: np.ndarray,
    prev_day_realized_demand: np.ndarray,
) -> tuple[ISOState, dict[str, np.ndarray]]:
    """
    Advance the ISO state forward by one day using the previous state's day-ahead
    information and the new day-ahead action.
    """

    prev_day_forecast = state.get_attribute("day_ahead_forecast")
    prev_day_dispatch = state.get_attribute("day_ahead_dispatch")
    prev_day_buy_price = state.get_attribute("day_ahead_buy_price")
    prev_day_sell_price = state.get_attribute("day_ahead_sell_price")

    day_ahead_dispatch = action.get_action("day_ahead_dispatch")
    day_ahead_buy_price = action.get_action("day_ahead_buy_price")
    day_ahead_sell_price = action.get_action("day_ahead_sell_price")

    new_state = ISOState(
        prev_day_realized_demand=prev_day_realized_demand,
        prev_day_forecast=prev_day_forecast,
        prev_day_dispatch=prev_day_dispatch,
        prev_day_buy_price=prev_day_buy_price,
        prev_day_sell_price=prev_day_sell_price,
        day_ahead_forecast=day_ahead_forecast,
        day_ahead_dispatch=day_ahead_dispatch,
        day_ahead_buy_price=day_ahead_buy_price,
        day_ahead_sell_price=day_ahead_sell_price,
    )

    info: dict[str, np.ndarray] = {
        "forecast_mismatch": prev_day_forecast - prev_day_realized_demand,
        "dispatch_mismatch": prev_day_dispatch - prev_day_realized_demand,
    }

    return new_state, info
