import numpy as np

def iso_day_to_day_transition(state, action, realized, forecast):
    """
    Updates the ISO's day-to-day state after a day ends.

    This transition records the realized demand of the previous day
    and previous day's dispatch, and stores the newly chosen day-ahead
    dispatch and forecast for the next day.
    It also returns diagnostic information such as the previous day's mismatch.

    Args:
        state (dict): The current ISO state containing at least 
        "day_ahead_dispatch" from the previous day.
        action (array-like): The dispatch plan chosen for the upcoming day.
        realized (array-like): The realized demand for the day that just ended.
        forecast (array-like): The forecasted demand for the next day.

    Returns:
        tuple:
            next_state (dict): Updated state with fields:
                - "prev_realized": realized demand from the ended day.
                - "prev_dispatch": dispatch used on the ended day.
                - "day_ahead_dispatch": dispatch plan for the next day.
                - "day_ahead_forecast": forecast for the next day.
            info (dict): Diagnostic values, including:
                - "prev_mismatch": realized minus dispatch for the ended day.
    """

    assert "day_ahead_dispatch" in state, "State missing 'day_ahead_dispatch'."
    action = np.asarray(action)
    realized = np.asarray(realized)
    forecast = np.asarray(forecast)
    prev_dispatch = np.asarray(state["day_ahead_dispatch"])

    assert realized.shape == prev_dispatch.shape, "Realized and prev_dispatch mismatch."
    assert action.shape == forecast.shape, "Action and forecast size mismatch."

    assert not np.isnan(realized).any(), "Realized contains NaN."
    assert not np.isnan(action).any(), "Action contains NaN."
    assert not np.isnan(forecast).any(), "Forecast contains NaN."


    next_state = {
        "prev_realized": realized,
        "prev_dispatch": prev_dispatch,
        "day_ahead_dispatch": action,
        "day_ahead_forecast": forecast,
    }

    info = {
        "prev_mismatch": realized - prev_dispatch
    }

    return next_state, info