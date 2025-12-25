from datetime import timedelta

from energy_net.gym_envs.pcs_env import PCSEnv
from stable_baselines3 import PPO
import numpy as np
from iso_env import ISOEnv

class ISOPricingWrapper:
    """
    Direct bridge between a live ISO model and the PCS environment.
    Converts RL actions into realistic price curves ($/unit).
    """

    def __init__(self, iso_model, base_price=0.10, price_scale=0.20):
        self.iso_model = iso_model
        self.base_price = base_price
        self.price_scale = price_scale

    def generate_price_curve(self, predicted_consumption: np.ndarray) -> np.ndarray:
        # ISO expects float32 forecast (48,)
        obs = predicted_consumption.astype(np.float32)

        # 1. Get raw action from the model
        action, _ = self.iso_model.predict(obs, deterministic=True)

        # 2. Slice the first 48 elements (prices)
        raw_prices = action[:48]

        # 3. Scale to actual currency range
        min_p, max_p = raw_prices.min(), raw_prices.max()
        denom = (max_p - min_p) + 1e-8
        normalized = (raw_prices - min_p) / denom

        # Center the price around base_price
        scaled_prices = (self.base_price - self.price_scale / 2) + (normalized * self.price_scale)

        return scaled_prices.astype(np.float32)


class AlternatingISOEnv(ISOEnv):
    def __init__(self, actual_csv, predicted_csv, pcs_env, pcs_model, steps_per_day=48):
        # Initialize the parent ISOEnv (handles data loading and the day pointer)
        super().__init__(actual_csv, predicted_csv, steps_per_day)

        self.pcs_env = pcs_env
        self.pcs_model = pcs_model

    def step(self, action):
        """
        1. Captures current index/timestamp.
        2. Injects ISO prices into PCS environment.
        3. Runs PCS agent for 48 sub-steps to get reward.
        4. Calls super().step() to advance the day pointer.
        """
        # --- 1. Map ISO Action to actual prices ($/unit) ---
        raw_prices = action[:self.T]
        # Robust normalization to avoid division by zero
        p_min, p_max = raw_prices.min(), raw_prices.max()
        denom = (p_max - p_min) + 1e-8
        normalized_prices = (raw_prices - p_min) / denom
        # Scale to a realistic range (e.g., $0.05 to $0.25)
        price_curve = 0.05 + (normalized_prices * 0.20)

        # --- 2. Sync PCS environment to the ISO's CURRENT pointer ---
        # We use self._next_start_idx BEFORE super().step() advances it.
        current_iso_timestamp = self.base_timestamp + timedelta(minutes=30 * self._next_start_idx)

        self.pcs_env.set_price_curve(price_curve.astype(np.float32))

        # --- 3. Run the PCS Agent for the full day ---
        # We force the PCS clock to match the ISO clock for this day
        obs, _ = self.pcs_env.reset(start_date=current_iso_timestamp)

        day_money = 0
        for _ in range(self.T):
            # PCS acts based on the ISO's price curve
            pcs_action, _ = self.pcs_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.pcs_env.step(pcs_action)
            day_money += info.get('step_money', 0)

        # --- 4. Advance the ISO pointer and get internal evaluation ---
        # This calls the parent ISOEnv.step, which performs the debug prints,
        # calculates the MAE, and then increments self._next_start_idx.
        _, _, _, _, iso_info = super().step(action)

        # --- 5. Return results ---
        # We override the reward to be the money earned by the PCS
        iso_reward = day_money

        # Add the money earned to the info dict for tracking
        iso_info["money_earned"] = day_money

        return self._expected, iso_reward, True, False, iso_info


def run_alternating_training():
    print("\n--- INITIALIZING TANDEM TRAINING ---")

    # 1. Initialize the internal PCS Environment
    # Note: Ensure the CSV filenames match your local files exactly
    base_pcs_env = PCSEnv(
        test_data_file='data_for_tests/synthetic_household_consumption_test.csv',
        predictions_file='data_for_tests/consumption_predictions.csv'
    )

    # 2. Initialize the PCS Agent
    # It learns in blocks of 48 steps (1 day)
    pcs_model = PPO("MlpPolicy", base_pcs_env, verbose=0, n_steps=48, batch_size=48)

    # 3. Initialize the Alternating ISO Environment
    # This environment now 'controls' the base_pcs_env internally
    iso_env = AlternatingISOEnv(
        actual_csv='data_for_tests/synthetic_household_consumption_test.csv',
        predicted_csv='data_for_tests/consumption_predictions.csv',
        pcs_env=base_pcs_env,
        pcs_model=pcs_model
    )

    # 4. Initialize the ISO Agent
    # It acts once per day, but we update every 7 days (n_steps=7)
    iso_model = PPO("MlpPolicy", iso_env, verbose=0, n_steps=7, batch_size=7)

    print(f"Simulation Started. ISO pointer at Row: {iso_env._next_start_idx + 2}")
    print("-" * 50)

    # 5. THE MAIN TRAINING LOOP
    # We loop through iterations. In each iteration, both agents get to learn.
    for iteration in range(1, 51):  # Run for 50 iterations

        # --- PHASE 1: ISO ACTION & PCS SIMULATION ---
        # We tell the ISO to learn for 7 steps (7 days).
        # Inside each step, AlternatingISOEnv.step() is called.
        # That step triggers 48 steps in the PCS env and advances the CSV pointer.
        print(f"\n[Iteration {iteration}] ISO Learning Phase (1 Days)...")
        iso_model.learn(total_timesteps=7, reset_num_timesteps=False)

        # --- PHASE 2: PCS POLICY UPDATE ---
        # The PCS has just experienced 7 days (7 * 48 = 336 steps) of data
        # while the ISO was 'learning'. Now we let the PCS update its own brain
        # based on the price curves the ISO just set.
        print(f"[Iteration {iteration}] PCS Learning Phase...")
        pcs_model.learn(total_timesteps=336, reset_num_timesteps=False)

        # --- LOGGING ---
        # The ISOEnv inherited pointer tells us exactly where we are in the CSV
        current_row = iso_env._next_start_idx + 2
        print(f">>> End of Iteration {iteration}. Next ISO start row: {current_row}")

        # Access profit from the base environment
        print(f">>> Cumulative PCS Money: ${base_pcs_env.get_money():.2f}")


if __name__ == "__main__":
    # Ensure you are in the correct directory where your CSVs are located
    try:
        run_alternating_training()
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        import traceback

        traceback.print_exc()