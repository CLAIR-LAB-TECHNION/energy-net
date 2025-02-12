import gymnasium as gym
from energy_net.env.pcs_unit_v0 import PCSUnitEnv

gym.register(
    id="PCSUnitEnv-v0",
    entry_point="energy_net.env.pcs_unit_v0:PCSUnitEnv"
)
