# energy_net/env/register_envs.py

from gymnasium.envs.registration import register

print("Registering PCSUnitEnv-v0")
register(
    id='PCSUnitEnv-v0',
    entry_point='energy_net.env.pcs_unit_v0:PCSUnitEnv',
    # Optional parameters:
    # max_episode_steps=1000,
    # reward_threshold=100.0,
    # nondeterministic=False,
)

print("Registering ISOEnv-v0")
register(
    id='ISOEnv-v0',
    entry_point='energy_net.env.iso_v0:ISOEnv',
    # Optional parameters:
    # max_episode_steps=1000,   
    # reward_threshold=100.0,
    # nondeterministic=False,
)