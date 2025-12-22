import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from stable_baselines3 import PPO
from typing import Optional, Tuple, Dict
import gymnasium as gym
from PCSGymEnv import PCSGymEnv  # Import your existing env


class ISOPricingWrapper:
    """
    Wrapper that uses a trained ISO agent to generate price curves for the PCS environment.
    """

    def __init__(self, iso_model_path: Optional[str] = None, fallback_base_price: float = 0.10):
        self.iso_model = None
        self.fallback_base_price = fallback_base_price

        if iso_model_path and Path(iso_model_path).exists():
            print(f"Loading ISO model from {iso_model_path}")
            self.iso_model = PPO.load(iso_model_path)
        else:
            print("No ISO model loaded - using fallback pricing")

    def generate_price_curve(self, predicted_consumption: np.ndarray,
                             base_price: float = 0.10,
                             price_scale: float = 1.0) -> np.ndarray:
        """
        Generate a price curve for a day given predicted consumption.

        Args:
            predicted_consumption: Array of predicted consumption for 48 timesteps
            base_price: Base price to add to normalized ISO prices
            price_scale: Scaling factor for ISO prices

        Returns:
            Array of prices for each timestep
        """
        if self.iso_model is None:
            # Fallback: simple sinusoidal pricing
            prices = []
            for i in range(len(predicted_consumption)):
                time_of_day = (i * 0.5) % 24  # 30-min steps
                variation = np.sin(2 * np.pi * time_of_day / 24) * 0.15
                prices.append(base_price + variation)
            return np.array(prices)

        # Use ISO agent to generate prices
        # ISO expects (T,) shape for observation
        obs = predicted_consumption.astype(np.float32)
        action, _ = self.iso_model.predict(obs, deterministic=True)

        # Extract prices from action (first T elements)
        T = len(predicted_consumption)
        iso_prices = action[:T]

        # Scale and shift ISO prices to reasonable range
        # ISO outputs in range [min_price, max_price], typically [-500, 500]
        # We need to map this to realistic electricity prices
        normalized_prices = (iso_prices - iso_prices.min()) / (iso_prices.max() - iso_prices.min() + 1e-8)
        scaled_prices = base_price + normalized_prices * price_scale

        return scaled_prices


class PCSGymEnvWithISO(gym.Env):
    """
    Enhanced PCS environment that gets prices from ISO agent.
    Collects data for ISO retraining.
    """

    def __init__(self,
                 test_data_file='synthetic_household_consumption_test.csv',
                 predictions_file='consumption_predictions.csv',
                 iso_model_path: Optional[str] = None,
                 dt=0.5 / 24,
                 episode_length_days=1,
                 prediction_horizon=48,
                 shortage_penalty=5.0,
                 base_price=0.10,
                 price_scale=0.20,  # Max variation from base price
                 **kwargs):

        # Initialize base PCS environment
        self.base_env = PCSGymEnv(
            test_data_file=test_data_file,
            predictions_file=predictions_file,
            dt=dt,
            episode_length_days=episode_length_days,
            prediction_horizon=prediction_horizon,
            shortage_penalty=shortage_penalty,
            base_price=base_price,
            **kwargs
        )

        # Copy spaces
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

        # ISO pricing wrapper
        self.iso_wrapper = ISOPricingWrapper(iso_model_path, base_price)
        self.base_price = base_price
        self.price_scale = price_scale

        # Data collection for ISO training
        self.episode_data = {
            'predicted_consumption': [],
            'realized_consumption': [],
            'prices_used': [],
            'storage_levels': [],
            'actions_taken': []
        }

    def reset(self, start_date=None, seed=None, options=None):
        """Reset and generate new price curve from ISO agent."""
        obs, info = self.base_env.reset(start_date=start_date, seed=seed, options=options)

        # Get predicted consumption for the full episode
        predicted_consumption = self.base_env._get_predicted_consumption(self.base_env.max_steps)

        # Generate price curve using ISO agent
        self.base_env.price_curve = self.iso_wrapper.generate_price_curve(
            predicted_consumption,
            self.base_price,
            self.price_scale
        )

        # Reset episode data collection
        self.episode_data = {
            'predicted_consumption': predicted_consumption.copy(),
            'realized_consumption': [],
            'prices_used': self.base_env.price_curve.copy(),
            'storage_levels': [],
            'actions_taken': []
        }

        return obs, info

    def step(self, action):
        """Step with data collection."""
        # Store pre-step data
        self.episode_data['storage_levels'].append(self.base_env.pcs.get_total_storage())
        self.episode_data['actions_taken'].append(float(action[0]))

        # Take step
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        # Store realized consumption
        self.episode_data['realized_consumption'].append(info['consumption_units'])

        return obs, reward, terminated, truncated, info

    def get_episode_data(self) -> Dict[str, np.ndarray]:
        """Get collected episode data for ISO training."""
        return {
            'predicted_consumption': np.array(self.episode_data['predicted_consumption']),
            'realized_consumption': np.array(self.episode_data['realized_consumption']),
            'prices_used': np.array(self.episode_data['prices_used']),
            'storage_levels': np.array(self.episode_data['storage_levels']),
            'actions_taken': np.array(self.episode_data['actions_taken'])
        }

    def render(self, mode='human'):
        return self.base_env.render(mode=mode)


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Example 1: Testing PCS with Fixed ISO Policy")
    print("=" * 60 + "\n")

    # Create environment with pre-trained ISO agent
    env = PCSGymEnvWithISO(
        test_data_file='synthetic_household_consumption_test.csv',
        predictions_file='consumption_predictions.csv',
        iso_model_path='ppo_isoenv.zip',  # Your friend's trained ISO
        episode_length_days=1,
        prediction_horizon=48,
        shortage_penalty=5.0,
        base_price=0.10,
        price_scale=0.20,
    )

    # Run a few episodes with random actions to see ISO pricing
    print("\nTesting with random actions:")
    for ep in range(3):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        shortages = 0

        print(f"\nEpisode {ep + 1}:")
        print(f"  ISO-generated prices (first 10): {env.base_env.price_curve[:10]}")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if info.get('shortage', False):
                shortages += 1
            done = terminated or truncated

        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Shortages: {shortages}")
