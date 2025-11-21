import numpy as np
from iso_dataclasses import ISOState, ISOAction


def iso_day_to_day_transition(
    state: ISOState,
    action: ISOAction,
    day_ahead_forecast: np.ndarray,
    prev_day_realized_demand: np.ndarray,
) -> tuple[ISOState, dict[str, np.ndarray]]:
    """

    """

    prev_day_forecast = state.day_ahead_forecast
    prev_day_dispatch = state.day_ahead_dispatch
    prev_day_buy_price = state.day_ahead_buy_price
    prev_day_sell_price = state.day_ahead_sell_price
    
    day_ahead_dispatch = action.day_ahead_dispatch
    day_ahead_buy_price = action.day_ahead_buy_price
    day_ahead_sell_price = action.day_ahead_sell_price

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
        # can add more diagnostics later
    }
    
    return new_state, info
