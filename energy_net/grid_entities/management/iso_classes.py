import numpy as np
from energy_net.foundation.model import State, Action
from .utils import validate_named_arrays_same_shape_no_nans


class ISOState(State):
    def __init__(
        self,
        prev_day_realized_demand: np.ndarray,
        prev_day_forecast: np.ndarray,
        prev_day_dispatch: np.ndarray,
        prev_day_buy_price: np.ndarray,
        prev_day_sell_price: np.ndarray,
        day_ahead_forecast: np.ndarray,
        day_ahead_dispatch: np.ndarray,
        day_ahead_buy_price: np.ndarray,
        day_ahead_sell_price: np.ndarray,
    ) -> None:
        fields = {
            "prev_day_realized_demand": prev_day_realized_demand,
            "prev_day_forecast": prev_day_forecast,
            "prev_day_dispatch": prev_day_dispatch,
            "prev_day_buy_price": prev_day_buy_price,
            "prev_day_sell_price": prev_day_sell_price,
            "day_ahead_forecast": day_ahead_forecast,
            "day_ahead_dispatch": day_ahead_dispatch,
            "day_ahead_buy_price": day_ahead_buy_price,
            "day_ahead_sell_price": day_ahead_sell_price,
        }

        validate_named_arrays_same_shape_no_nans(fields)

        super().__init__(fields)


class ISOAction(Action):
    def __init__(
        self,
        day_ahead_dispatch: np.ndarray,
        day_ahead_buy_price: np.ndarray,
        day_ahead_sell_price: np.ndarray,
    ) -> None:
        fields = {
            "day_ahead_dispatch": day_ahead_dispatch,
            "day_ahead_buy_price": day_ahead_buy_price,
            "day_ahead_sell_price": day_ahead_sell_price,
        }

        validate_named_arrays_same_shape_no_nans(fields)
        super().__init__(fields)