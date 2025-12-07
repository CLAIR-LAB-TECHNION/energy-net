import numpy as np
from .iso_classes import ISOState, ISOAction


def iso_day_to_day_transition(
    state: ISOState,
    action: ISOAction,
    day_ahead_forecast: np.ndarray,
    prev_day_realized_demand: np.ndarray,
) -> tuple[ISOState, dict[str, np.ndarray]]:
    """
    Advance the ISO state forward by one day using the previous state's
    day-ahead values and the new day-ahead action.

    The function constructs a new `ISOState` where:
        - The previous state's day-ahead forecast/dispatch/prices become the
        new state's prev-day values.
        - The provided `day_ahead_forecast` and the fields from `action`
        become the new state's day-ahead values.
        - Validation is performed by the ISOState constructor to ensure all
        arrays have matching shapes and no NaN/inf values.

    It also returns diagnostic information capturing differences between
    the previous day's forecasts/dispatch and the realized demand.

    Args:
        state (ISOState): The current day's state.
        action (ISOAction): The ISO day-ahead action for the next day.
        day_ahead_forecast (np.ndarray): Forecasted demand for the upcoming day.
        prev_day_realized_demand (np.ndarray): Actual realized demand for the previous day.

    Returns:
        tuple[ISOState, dict[str, np.ndarray]]:
            A pair `(new_state, info)` where:

            - `new_state` (ISOState): The advanced state representing the next day.
            - `info` (dict[str, np.ndarray]): Diagnostics containing:
                - `"forecast_mismatch"`: previous day forecast - realized demand
                - `"dispatch_mismatch"`: previous day dispatch - realized demand

    Raises:
        ValueError: If array shapes do not match or contain NaN/inf values
            (via ISOState validation).
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
