# alternating_env.py
from datetime import timedelta
from typing import Sequence

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
        # 1) Price mapping logic
        raw_prices = action[:self.T]
        p_min, p_max = raw_prices.min(), raw_prices.max()
        denom = (p_max - p_min) + 1e-8
        normalized_prices = (raw_prices - p_min) / denom
        price_curve = 0.05 + (normalized_prices * 0.20)

        # 2) Anchor the index to the ISO's current position
        current_idx = self._next_start_idx
        current_iso_timestamp = self._get_iso_timestamp_from_pcs_index(current_idx)

        # 3) Setup PCS
        self.pcs_env.set_price_curve(price_curve.astype(np.float32))
        self.pcs_env.set_step_index(current_idx)
        obs, _ = self.pcs_env.reset(start_date=current_iso_timestamp)

        day_money = 0
        for _ in range(self.T):
            pcs_action, _ = self.pcs_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.pcs_env.step(pcs_action)
            day_money += info.get('step_money', 0)

        # 4) Advance ISO pointer via super()
        # This moves _next_start_idx forward by exactly self.T (48)
        _, _, _, _, iso_info = super().step(action)

        iso_info["money_earned"] = day_money
        return self._expected, day_money, True, False, iso_info
def get_pred_window(preds: Sequence[float], start: int, length: int) -> np.ndarray:
    """Return a length-sized window starting at `start` from preds, wrapping if necessary."""
    n = len(preds)
    if n == 0:
        raise ValueError("Predictions array is empty.")
    start = int(start) % n
    if start + length <= n:
        return preds[start:start + length]
    # wrap-around case
    first = preds[start:]
    remaining = length - len(first)
    # take from the beginning as needed
    second = preds[:remaining]
    return np.concatenate([first, second], axis=0)


# -----------------------
# Main training function
# -----------------------
def run_alternating_training(
        cycle_days: int = 7,
        total_iterations: int = 50,
        *,
        pcs_algo_cls=PPO,
        iso_algo_cls=PPO,
        pcs_policy: str = "MlpPolicy",
        iso_policy: str = "MlpPolicy",
        pcs_algo_kwargs: dict | None = None,
        iso_algo_kwargs: dict | None = None,
        pcs_steps_per_day: int | None = None,
        verbose: int = 0,
        # if True prints per-day debug eval money
        per_day_debug: bool = True,
):
    """
    Tandem ISO <-> PCS training loop (configurable).
    ... (comments preserved) ...
    """

    # -------------------------
    # Basic validation & kwargs
    # -------------------------
    if cycle_days < 1:
        raise ValueError("cycle_days must be >= 1")
    if total_iterations < 1:
        raise ValueError("total_iterations must be >= 1")

    # FIX: Force BOTH models to respect the step limits.
    # Without this, PPO runs 2048 steps by default, causing the jump.
    pcs_algo_kwargs = {} if pcs_algo_kwargs is None else dict(pcs_algo_kwargs)

    # determine how many steps constitute a day (default to env.max_steps)
    # We need this early to set the buffer size
    temp_env = PCSEnv(test_data_file='data_for_tests/synthetic_household_consumption_test.csv',
                      predictions_file='data_for_tests/consumption_predictions.csv')
    steps_per_day = int(pcs_steps_per_day or temp_env.max_steps)

    pcs_algo_kwargs.setdefault('n_steps', steps_per_day)
    pcs_algo_kwargs.setdefault('batch_size', steps_per_day)

    iso_algo_kwargs = {} if iso_algo_kwargs is None else dict(iso_algo_kwargs)
    iso_algo_kwargs.setdefault('n_steps', cycle_days)
    iso_algo_kwargs.setdefault('batch_size', cycle_days)

    print("\n--- INITIALIZING TANDEM TRAINING ---")

    # 1) Create PCS environment
    base_pcs_env = temp_env  # Use the one we already created

    # 2) Create PCS RL agent
    print(f"Creating PCS model: {pcs_algo_cls.__name__} policy={pcs_policy} kwargs={pcs_algo_kwargs}")
    pcs_model = pcs_algo_cls(pcs_policy, base_pcs_env, verbose=verbose, **pcs_algo_kwargs)

    # 3) Create Alternating ISO env and ISO model
    iso_env = AlternatingISOEnv(
        actual_csv='data_for_tests/synthetic_household_consumption_test.csv',
        predicted_csv='data_for_tests/consumption_predictions.csv',
        pcs_env=base_pcs_env,
        pcs_model=pcs_model
    )

    print(f"Creating ISO model: {iso_algo_cls.__name__} policy={iso_policy} kwargs={iso_algo_kwargs}")
    iso_model = iso_algo_cls(iso_policy, iso_env, verbose=verbose, **iso_algo_kwargs)

    print(f"Simulation Started. ISO pointer at Row: {iso_env._next_start_idx + 2}")
    print("-" * 50)

    # Preload predicted values
    pred_df = pd.read_csv('data_for_tests/consumption_predictions.csv', parse_dates=['timestamp']).set_index(
        'timestamp').sort_index()
    predicted_vals = pred_df['predicted_consumption'].astype(float).to_numpy().flatten()
    pricing = ISOPricingWrapper(iso_model)

    # -------------------------
    # Main alternating loop
    # -------------------------
    for iteration in range(1, total_iterations + 1):
        # ------- PHASE 1: ISO Learning -------
        print(f"\n[Iteration {iteration}] ISO Learning Phase ({cycle_days} Days)...")
        iso_model.learn(total_timesteps=cycle_days, reset_num_timesteps=False)

        # ------- PHASE 2: PCS Learning (per-day eval + training) -------
        print(f"[Iteration {iteration}] PCS Learning Phase...")

        # Start PCS right after the ISO block
        base_idx = iso_env._next_start_idx

        day_money_list = []

        for day in range(cycle_days):
            start = base_idx + day * steps_per_day

            pred_window = get_pred_window(predicted_vals, start, steps_per_day).astype(np.float32)
            price_curve = pricing.generate_price_curve(pred_window)

            base_pcs_env.set_price_curve(price_curve.astype(np.float32))
            base_pcs_env.set_step_index(start)

            # 1) EVALUATE (deterministic)
            obs, _ = base_pcs_env.reset(start_date=iso_env._get_iso_timestamp_from_pcs_index(start))
            eval_day_money = 0.0
            for _step in range(steps_per_day):
                pcs_action, _ = pcs_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = base_pcs_env.step(pcs_action)
                eval_day_money += info.get('step_money', 0.0)
            day_money_list.append(eval_day_money)

            if per_day_debug:
                print(f"  [DEBUG] Day {day + 1}/{cycle_days} start_idx={start} eval_money=${eval_day_money:.2f}")

            # 2) TRAIN the PCS on that same day
            base_pcs_env.set_step_index(start)
            obs, _ = base_pcs_env.reset(start_date=iso_env._get_iso_timestamp_from_pcs_index(start))

            # This now stops exactly at steps_per_day because of pcs_algo_kwargs
            pcs_model.learn(total_timesteps=steps_per_day, reset_num_timesteps=False)

        # Sync ISO pointer forward to where the LAST day of training finished
        # If Day 7 started at 624 and ran 48 steps, this sets us to 672.
        final_idx = base_idx + (cycle_days * steps_per_day)
        iso_env.sync_to_pcs(final_idx)

        # ------- Logging -------
        avg_money = float(np.mean(day_money_list)) if day_money_list else 0.0
        print(f">>> End of Iteration {iteration}. Next ISO start row: {iso_env._next_start_idx + 2}")
        print(f">>> Iteration {iteration} average PCS money per day: ${avg_money:.2f}")

    print("\n--- TRAINING COMPLETE ---")
if __name__ == "__main__":
    try:
        run_alternating_training()
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        import traceback
        traceback.print_exc()
