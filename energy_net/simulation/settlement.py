"""Financial settlement of the prosumer's grid exchange (net metering)."""

# Residual load at or below this is floating-point noise, not grid supply.
SHORTAGE_TOLERANCE = 1e-12


def net_meter_money(grid_exchange: float, buy_tariff: float, sell_tariff: float) -> float:
    """
    Money earned in one interval (negative when paying).

    A positive grid exchange is an import paid at the buy tariff; a
    negative exchange is an export credited at the sell tariff.
    """
    return -(
        buy_tariff * max(grid_exchange, 0.0)
        - sell_tariff * max(-grid_exchange, 0.0)
    )


def is_grid_supplied(uncovered_load: float) -> bool:
    """True when the grid had to supply household load in the interval."""
    return uncovered_load > SHORTAGE_TOLERANCE
