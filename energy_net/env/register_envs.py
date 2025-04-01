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

print("Registering EnergyNetEnv-v0")
register(
    id='EnergyNetEnv-v0',
    entry_point='energy_net.env.energy_net_v0:EnergyNetV0',
    # Optional parameters:
    # max_episode_steps=1000,   
    # reward_threshold=100.0,
    # nondeterministic=False,
)

# Register additional environments for RL Zoo integration

print("Registering ISO-RLZoo-v0")
register(
    id='ISO-RLZoo-v0',
    entry_point='energy_net.env.iso_env:make_iso_env_zoo',
    max_episode_steps=48,  # Based on your config
)

print("Registering PCS-RLZoo-v0")
register(
    id='PCS-RLZoo-v0', 
    entry_point='energy_net.env.pcs_env:make_pcs_env_zoo',
    max_episode_steps=48,  # Based on your config
)