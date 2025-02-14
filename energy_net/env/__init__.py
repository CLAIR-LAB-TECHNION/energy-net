# energy_net/env/__init__.py

# Import the registration script to register the environments
from .register_envs import *

from enum import Enum

class PricingPolicy(Enum):
    QUADRATIC = "quadratic"
    ONLINE = "online"
    CONSTANT = "constant"
