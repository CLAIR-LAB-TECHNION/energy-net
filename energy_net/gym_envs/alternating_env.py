from datetime import timedelta

from energy_net.gym_envs.pcs_env import PCSEnv
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
from iso_env import ISOEnv


class ISOPricingWrapper:
    """Direct bridge between an ISO RL model and the PCS environment.
    Converts a 48-step predicted consumption window into a realistic
    price curve (in $/unit) using the ISO policy's first 48 action values.
    """

    def __init__(self, iso_model, base_price=0.10, price_scale=0.20):
        self.iso_model = iso_model
        self.base_price = base_price
        self.price_scale = price_scale

    def generate_price_curve(self, predicted_consumption: np.ndarray) -> np.ndarray:
        # ISO expects a float32 (48,) forecast observation
        obs = predicted_consumption.astype(np.float32)

        # Query the ISO policy deterministically
        action, _ = self.iso_model.predict(obs, deterministic=True)

        # First 48 entries represent an unscaled price signal
        raw_prices = action[:48]

        # Robust min-max normalization
        min_p, max_p = raw_prices.min(), raw_prices.max()
        denom = (max_p - min_p) + 1e-8
        normalized = (raw_prices - min_p) / denom

        # Center and scale into a realistic price range
        scaled_prices = (self.base_price - self.price_scale / 2) + (normalized * self.price_scale)
        return scaled_prices.astype(np.float32)


class AlternatingISOEnv(ISOEnv):
    """An ISO environment that internally coordinates a PCS environment.

    Behavior differences compared to a simple nested reset:
    - The ISO queries the PCS env for its global half-hour index and advances
      the ISO pointer forward to that index (prevents rewinding).
    - The ISO injects its price curve into the PCS and runs the PCS policy for
      T=48 steps to produce an economic reward which becomes the ISO reward.
    """

    def __init__(self, actual_csv, predicted_csv, pcs_env: PCSEnv, pcs_model, steps_per_day=48):
        super().__init__(actual_csv, predicted_csv, steps_per_day)
        self.pcs_env = pcs_env
        self.pcs_model = pcs_model

    def sync_to_pcs(self, pcs_step_index: int):
        """Move the ISO's internal pointer to match the PCS half-hour index."""
        self._next_start_idx = int(pcs_step_index)

    def _get_iso_timestamp_from_pcs_index(self, pcs_index: int):
        return self.base_timestamp + timedelta(minutes=30 * int(pcs_index))

    def step(self, action):
        """Map ISO action to prices, sync to PCS, run PCS for a day, then advance ISO.

        Returns: (obs, reward, done, truncated, info) where reward is PCS earnings.
        """
        # --- 1) Map ISO action -> realistic price curve ($/unit) ---
        raw_prices = action[:self.T]
        p_min, p_max = raw_prices.min(), raw_prices.max()
        denom = (p_max - p_min) + 1e-8
        normalized_prices = (raw_prices - p_min) / denom
        price_curve = 0.05 + (normalized_prices * 0.20)

        # --- 2) Query PCS for current index and sync ISO pointer forward ---
        pcs_index = self.pcs_env.get_step_index()
        self.sync_to_pcs(pcs_index)

        # Timestamp corresponding to the PCS index (used for PCS.reset)
        current_iso_timestamp = self._get_iso_timestamp_from_pcs_index(pcs_index)

        # --- 3) Inject prices into PCS and align its index ---
        self.pcs_env.set_price_curve(price_curve.astype(np.float32))
        # Align PCS internal pointers so it starts the episode at the same index
        self.pcs_env.set_step_index(pcs_index)

        # Reset PCS to start the episode (this resets physical state but does not rewind the dataset)
        obs, _ = self.pcs_env.reset(start_date=current_iso_timestamp)

        # Run the PCS policy for a full day, accumulate money
        day_money = 0
        for _ in range(self.T):
            pcs_action, _ = self.pcs_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.pcs_env.step(pcs_action)
            day_money += info.get('step_money', 0)

        # --- 4) Advance the ISO pointer and collect ISO internal info ---
        _, _, _, _, iso_info = super().step(action)

        # Make the ISO reward equal to the PCS money earned for that day
        iso_reward = day_money
        iso_info["money_earned"] = day_money

        # Return observation expected by ISOEnv (self._expected), reward and info
        return self._expected, iso_reward, True, False, iso_info


def run_alternating_training(cycle_days: int = 7, total_iterations: int = 50):
    """Main training loop for tandem ISO <-> PCS training.

    Args:
        cycle_days: number of days in each ISO/PCS cycle (replaces the hardcoded 7).
        total_iterations: how many ISO<->PCS iterations to run.
    """
    print("--- INITIALIZING TANDEM TRAINING ---")

    # 1) Create PCS environment directly (assume PCSEnv now implements
    #    get_step_index() and set_step_index())
    base_pcs_env = PCSEnv(
        test_data_file='data_for_tests/synthetic_household_consumption_test.csv',
        predictions_file='data_for_tests/consumption_predictions.csv'
    )

    # 2) Create PCS policy on the PCS env
    pcs_model = PPO("MlpPolicy", base_pcs_env, verbose=0, n_steps=48, batch_size=48)

    # 3) Create the Alternating ISO environment which will coordinate the PCS
    iso_env = AlternatingISOEnv(
        actual_csv='data_for_tests/synthetic_household_consumption_test.csv',
        predicted_csv='data_for_tests/consumption_predictions.csv',
        pcs_env=base_pcs_env,
        pcs_model=pcs_model
    )

    # 4) ISO agent (acts once per day; set n_steps/batch_size to cycle_days)
    iso_model = PPO("MlpPolicy", iso_env, verbose=0, n_steps=cycle_days, batch_size=cycle_days)

    print(f"Simulation Started. ISO pointer at Row: {iso_env._next_start_idx + 2}")
    print("-" * 50)

    # Preload predicted values once (they wrap naturally as you said)
    predicted_vals = pd.read_csv('data_for_tests/consumption_predictions.csv', header=None).to_numpy().flatten()
    pricing = ISOPricingWrapper(iso_model)
    steps_per_day = iso_env.T

    # --- Main alternating training loop ---
    for iteration in range(1, total_iterations + 1):
        # PHASE 1: ISO Learning Phase (ISO updates across cycle_days day-steps)
        print(f"[Iteration {iteration}] ISO Learning Phase ({cycle_days} Days)...")
        iso_model.learn(total_timesteps=cycle_days, reset_num_timesteps=False)

        # PHASE 2: PCS Learning Phase - one fixed-day episode per day
        print(f"[Iteration {iteration}] PCS Learning Phase...")
        base_idx = iso_env._next_start_idx

        for day in range(cycle_days):
            # compute global half-hour start index for this day
            start = base_idx + day * steps_per_day

            # slice the predicted window (prediction file wraps naturally)
            pred_window = predicted_vals[start:start + steps_per_day].astype(np.float32)

            # get the ISO's price curve for that day (ISO policy static during PCS updates)
            price_curve = pricing.generate_price_curve(pred_window)

            # inject curve and align PCS to the intended start
            base_pcs_env.set_price_curve(price_curve.astype(np.float32))
            base_pcs_env.set_step_index(start)

            # train PCS for exactly one day (steps_per_day steps) under this fixed curve
            pcs_model.learn(total_timesteps=steps_per_day, reset_num_timesteps=False)

        # After PCS completes the cycle of day-episodes, sync ISO pointer forward
        pcs_idx = base_pcs_env.get_step_index()
        iso_env.sync_to_pcs(pcs_idx)

        # Logging
        current_row = iso_env._next_start_idx + 2
        print(f">>> End of Iteration {iteration}. Next ISO start row: {current_row}")
        print(f">>> Cumulative PCS Money: ${base_pcs_env.get_money():.2f}")


if __name__ == "__main__":
    try:
        run_alternating_training()
    except Exception as e:
        print(f"FATAL ERROR during training: {e}")
        import traceback
        traceback.print_exc()
