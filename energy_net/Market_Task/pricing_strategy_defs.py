"""
Default configuration values for pricing strategies.

This module centralizes all default values used by the pricing strategy implementations,
making them easier to maintain and modify.
"""

# Dispatch configuration defaults
DEFAULT_DISPATCH_MIN = 0.0
DEFAULT_DISPATCH_MAX = 500.0

# Quadratic pricing defaults
QUADRATIC_POLY_MIN = -100.0
QUADRATIC_POLY_MAX = 100.0
QUADRATIC_COEF_SIZE = 3  # Number of polynomial coefficients (a, b, c)

# Constant pricing defaults
# (uses min_price and max_price from constructor)

# Online pricing defaults
# (uses min_price and max_price from constructor)

# Action space dimensions
QUADRATIC_NUM_COEFFICIENTS = 6  # Buy coefficients (3) + Sell coefficients (3)
CONSTANT_NUM_PRICES = 2  # Buy price + Sell price
ONLINE_NUM_PRICES = 2  # Buy price + Sell price