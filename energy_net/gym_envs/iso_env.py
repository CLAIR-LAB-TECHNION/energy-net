import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from datetime import datetime, timedelta


class ISOEnv(gym.Env):
    def __init__(self, actual_csv, predicted_csv, steps_per_day=48):
        super().__init__()

        # 1. Define these BEFORE loading data so they are guaranteed to exist
        self.T = steps_per_day

        # Observation: The 48 predicted consumption values for the day
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.T,),
            dtype=np.float32
        )

        # Action: 48 Prices + 48 Dispatch values (Total 96)
        self.action_space = spaces.Box(
            low=-500,
            high=500,
            shape=(self.T * 2,),
            dtype=np.float32
        )

        # 2. Load your data
        self.actual_df = pd.read_csv(actual_csv)
        self.pred_df = pd.read_csv(predicted_csv)

        # Get the first timestamp from the CSV to use as a clock base
        # This is required for the Tandem script to sync the PCS clock
        self.base_timestamp = pd.to_datetime(self.actual_df.iloc[0]['Datetime'])

        self.actual_data = self.actual_df['Consumption'].values.astype(np.float32)
        self.pred_data = self.pred_df['predicted_consumption'].values.astype(np.float32)

        self.total_rows = min(len(self.actual_data), len(self.pred_data))

        # This is the specific attribute your Tandem script was missing
        self._next_start_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = self._next_start_idx

        # Safety check for index out of bounds
        if idx + self.T > self.total_rows:
            idx = 0
            self._next_start_idx = 0

        self._expected = self.pred_data[idx: idx + self.T]
        self._realized = self.actual_data[idx: idx + self.T]

        # Calculate the current timestamp for this day
        current_time = self.base_timestamp + timedelta(minutes=30 * idx)

        # Return info with names that match your test script and Tandem script
        return self._expected, {
            "start_idx": idx,
            "timestamp": current_time
        }

    def step(self, action):
        # 1. EVALUATION (Before moving the pointer)
        prices = action[:self.T]
        dispatch = action[self.T:]
        cost = float(np.mean(np.abs(dispatch - self._realized)))

        # 2. PREPARE INFO DICT
        # Calculate current timestamp for the step info
        current_time = self.base_timestamp + timedelta(minutes=30 * self._next_start_idx)

        step_info = {
            "realized": self._realized,
            "predicted": self._expected,
            "dispatch": dispatch,
            "prices": prices,
            "mae": cost,
            "start_idx": self._next_start_idx,
            "timestamp": current_time
        }

        # 3. ADVANCE THE POINTER (Move to the next day)
        self._next_start_idx += self.T

        # Loop if we hit the end
        if self._next_start_idx + self.T > self.total_rows:
            print("!!! REACHED END OF CSV - RESTARTING FROM ROW 0 !!!")
            self._next_start_idx = 0

        # One step = One day. Always terminated=True.
        return self._expected, -cost, True, False, step_info


# ==========================================
# MAIN FUNCTION: THE CLI MONITOR
# ==========================================
def main():
    # Make sure these filenames match your local files!
    # These paths are relative to where you run the script
    try:
        env = ISOEnv(
            actual_csv='synthetic_household_consumption_test.csv',
            predicted_csv='consumption_predictions.csv'
        )
    except FileNotFoundError:
        # Fallback for the folder structure in your Tandem script
        env = ISOEnv(
            actual_csv='data_for_tests/synthetic_household_consumption_test.csv',
            predicted_csv='data_for_tests/consumption_predictions.csv'
        )

    print(f"\nSimulation Started. Total Records: {env.total_rows}")

    for day in range(1, 4):  # Look at 3 days
        obs, info = env.reset()

        # Dummy Action (Example: 48 prices, 48 dispatch values)
        # We concatenate dummy prices (0.1 to 0.5) with the forecast (obs)
        dummy_prices = np.linspace(0.1, 0.5, 48)
        action = np.concatenate([dummy_prices, obs])

        _, reward, _, _, step_info = env.step(action)

        print(f"--- [DAY {day}] ---")
        print(f"CSV Row Range: {info['start_idx'] + 2} to {info['start_idx'] + 49}")
        print(f"Calendar Date: {info['timestamp'].strftime('%Y-%m-%d')}")
        print(f"{'Time':<10} | {'CSV Row':<8} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}")
        print("-" * 70)

        for i in range(0, 48, 4):
            slot_time = info['timestamp'] + timedelta(minutes=30 * i)
            row_num = info['start_idx'] + i + 2

            p = step_info['predicted'][i]
            a = step_info['realized'][i]
            d = step_info['dispatch'][i]
            pr = step_info['prices'][i]

            print(f"{slot_time.strftime('%H:%M'):<10} | {row_num:<8} | {p:<8.3f} | {a:<8.3f} | {d:<10.3f} | {pr:<6.2f}")

        print("-" * 70)
        print(f"Day Summary: Avg MAE = {step_info['mae']:.4f}\n")


if __name__ == "__main__":
    main()