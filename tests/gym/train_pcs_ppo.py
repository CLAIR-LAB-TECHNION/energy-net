import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from RL_Test import PCSGymEnv  # adjust if needed


def make_env():
    """Create one environment for fast, silent testing"""
    return PCSGymEnv(
        data_file='SystemDemand_30min_2023-2025.csv',
        dt=0.5/24,
        episode_length_days=1,   # 1-day episodes
        prediction_horizon=12,   # smaller horizon
        shortage_penalty=1000.0,
        base_price=0.10,
        price_volatility=0.05
        # No render_mode, so no printing
    )


def main():
    # Single environment for fast silent test
    env = DummyVecEnv([make_env])

    # PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,       # silent, no tables
        learning_rate=3e-4,
        n_steps=256,     # small rollout
        batch_size=64,
        gamma=0.99
    )

    # --- Short training ---
    model.learn(total_timesteps=5000)

    # --- Quick evaluation ---
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward.item()  # safer, extracts the single value

    print(f"\nSilent test episode total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
