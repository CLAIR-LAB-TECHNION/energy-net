"""PCS reward variants; both use the same physical net-meter settlement.

The optional penalty treats grid-supplied residual demand as an experimental
reward-shaping signal, *not* as unserved household demand (the grid supplies it).
"""

SHORTAGE_TOLERANCE = 1e-12


def net_meter_reward(grid_exchange: float, buy_tariff: float, sell_tariff: float) -> float:
    """Negative electricity bill under the PCS POMDP's net-metering rule.

    A positive grid exchange is an import paid at the PCS buy tariff.
    A negative exchange is an export remunerated at the PCS sell tariff.
    """
    return -(
        buy_tariff * max(grid_exchange, 0.0)
        - sell_tariff * max(-grid_exchange, 0.0)
    )


def net_meter_reward_with_shortage_penalty(
    grid_exchange: float,
    buy_tariff: float,
    sell_tariff: float,
    uncovered_demand: float,
    shortage_penalty: float,
) -> float:
    """Experimental variant: net-meter reward minus a fixed grid-reliance penalty.

    The penalty is charged once per interval with positive residual load, not
    per unit of demand. It is additional to the normal cost of grid imports.
    """
    reward = net_meter_reward(grid_exchange, buy_tariff, sell_tariff)
    if uncovered_demand > SHORTAGE_TOLERANCE:
        reward -= shortage_penalty
    return reward
