import numpy as np
from iso_dataclasses import ISOState, ISOAction


def iso_day_to_day_transition(
        state: ISOState,
        action: ISOAction,
        day_ahead_forecast: np.ndarray,
        prev_day_realized: np.ndarray
):

    new_state = ISOState(
        prev_day_realized=prev_day_realized,
        prev_day_dispatch=state.day_ahead_dispatch,
        prev_day_pricing_plan=state.day_ahead_pricing_plan,
        day_ahead_forecast=day_ahead_forecast,
        day_ahead_dispatch=action.day_ahead_dispatch,
        day_ahead_pricing_plan=action.day_ahead_pricing_plan,
    )

    info = {
        "forecast_error": day_ahead_forecast - prev_day_realized,
        "dispatch_error": state.day_ahead_dispatch - prev_day_realized,
    }

    return new_state, info