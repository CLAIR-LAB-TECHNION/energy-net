# energy_net/envs/register_envs.py

from gymnasium.envs.registration import register

# Register additional environments for RL Zoo integration

print("Registering ISO-RLZoo-v0")
register(
    id='ISO-RLZoo-v0',
    entry_point='energy_net.envs.iso_env:make_iso_env_zoo',
    max_episode_steps=48,  # Based on your config
)

print("Registering PCS-RLZoo-v0")
register(
    id='PCS-RLZoo-v0',
    entry_point='energy_net.envs.pcs_env:make_pcs_env_zoo',
    max_episode_steps=48,  # Based on your config
)