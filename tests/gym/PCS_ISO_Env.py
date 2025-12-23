import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from datetime import datetime, timedelta


class ISOEnv(gym.Env):
    def __init__(self, actual_csv, predicted_csv, steps_per_day=48):
        super().__init__()

        # Load data to get the starting timestamp
        actual_df = pd.read_csv(actual_csv)
        pred_df = pd.read_csv(predicted_csv)

        # Convert first index to a datetime object for tracking
        self.base_timestamp = pd.to_datetime(actual_df.iloc[0]['Datetime'])

        self.actual_data = actual_df['Consumption'].values.astype(np.float32)
        self.pred_data = pred_df['predicted_consumption'].values.astype(np.float32)

        self.total_rows = min(len(self.actual_data), len(self.pred_data))
        self.T = steps_per_day
        self._current_start_idx = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.T,), dtype=np.float32)
        self.action_space = spaces.Box(low=-500, high=500, shape=(self.T * 2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._current_start_idx + self.T > self.total_rows:
            print("\n" + "=" * 30 + " RESTARTING DATASET " + "=" * 30 + "\n")
            self._current_start_idx = 0

        # Grab windows
        self._expected = self.pred_data[self._current_start_idx: self._current_start_idx + self.T]
        self._realized = self.actual_data[self._current_start_idx: self._current_start_idx + self.T]

        # Current index and timestamp
        idx_info = self._current_start_idx
        current_time = self.base_timestamp + timedelta(minutes=30 * idx_info)

        self._current_start_idx += self.T

        return self._expected, {"start_idx": idx_info, "timestamp": current_time}

    def step(self, action):
        prices = action[:self.T]
        dispatch = action[self.T:]
        cost = float(np.mean(np.abs(dispatch - self._realized)))

        return self._expected, -cost, True, False, {
            "realized": self._realized,
            "predicted": self._expected,
            "dispatch": dispatch,
            "prices": prices,
            "mae": cost
        }


# ==========================================
# MAIN FUNCTION: THE CLI MONITOR
# ==========================================
def main():
    env = ISOEnv(
        actual_csv='synthetic_household_consumption_test.csv',
        predicted_csv='consumption_predictions.csv'
    )

    print(f"\nSimulation Started. Total Records: {env.total_rows}")
    print(f"Each Episode resets at 00:00:00 of the next day.\n")

    for day in range(1, 4):  # Look at 3 days
        obs, info = env.reset()

        # Dummy Action (Dispatch = Predicted)
        action = np.concatenate([np.linspace(0.1, 0.5, 48), obs])
        _, reward, _, _, step_info = env.step(action)

        print(f"--- [DAY {day}] ---")
        print(f"CSV Row Range: {info['start_idx']+2} to {info['start_idx'] +2 + 47}")
        print(f"Calendar Date: {info['timestamp'].strftime('%Y-%m-%d')}")
        print(f"{'Time':<10} | {'CSV Row':<8} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}")
        print("-" * 70)

        # Print every 4th step (every 2 hours) so the console isn't too cluttered
        for i in range(0, 48, 4):
            # Calculate time for this specific slot
            slot_time = info['timestamp'] + timedelta(minutes=30 * i)
            row_num = info['start_idx'] + i

            p = step_info['predicted'][i]+2
            a = step_info['realized'][i]+2
            d = step_info['dispatch'][i]
            pr = step_info['prices'][i]

            print(f"{slot_time.strftime('%H:%M'):<10} | {row_num:<8} | {p:<8.3f} | {a:<8.3f} | {d:<10.3f} | {pr:<6.2f}")

        print("-" * 70)
        print(f"Day Summary: Avg MAE = {step_info['mae']:.4f}\n")


if __name__ == "__main__":
    main()