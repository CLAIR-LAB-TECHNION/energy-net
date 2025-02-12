import gymnasium as gym
from energy_net.env.iso_v0 import ISOEnv

gym.register(
    id="ISOEnv-v0",
    entry_point="energy_net.env.iso_v0:ISOEnv"
)
