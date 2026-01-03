import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from datetime import datetime, timedelta


class ISOEnv(gym.Env):
    """
    ISO Environment that automatically detects and includes feature columns
    from the predictions CSV in its observation space.
    """

    def __init__(self, actual_csv, predicted_csv, steps_per_day=48):
        super().__init__()

        # 1. Define these BEFORE loading data so they are guaranteed to exist
        self.T = steps_per_day

        # 2. Load your data
        self.actual_df = pd.read_csv(actual_csv)
        self.pred_df = pd.read_csv(predicted_csv)

        # Get the first timestamp from the CSV to use as a clock base
        self.base_timestamp = pd.to_datetime(self.actual_df.iloc[0]['Datetime'])

        self.actual_data = self.actual_df['Consumption'].values.astype(np.float32)
        self.pred_data = self.pred_df['predicted_consumption'].values.astype(np.float32)

        # 3. Automatically detect feature columns
        excluded_cols = {'timestamp', 'predicted_consumption'}
        self.feature_columns = [col for col in self.pred_df.columns if col not in excluded_cols]
        self.num_features = len(self.feature_columns)

        print(f"Detected {self.num_features} feature columns: {self.feature_columns}")

        # Cache feature values for fast lookup
        if self.num_features > 0:
            self.feature_cache = self.pred_df[self.feature_columns].values.astype(np.float32)
        else:
            self.feature_cache = None

        self.total_rows = min(len(self.actual_data), len(self.pred_data))

        # 4. Define observation and action spaces
        # Observation: predicted consumption (T values) + features per timestep (T * num_features)
        obs_dim = self.T + (self.T * self.num_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        print(f"Observation space dimension: {obs_dim}")
        print(f"  - Predicted consumption per day: {self.T}")
        print(f"  - Features per timestep: {self.num_features}")
        print(f"  - Total features per day: {self.T * self.num_features}")

        # Action: 48 Prices + 48 Dispatch values (Total 96)
        self.action_space = spaces.Box(
            low=-500,
            high=500,
            shape=(self.T * 2,),
            dtype=np.float32
        )

        # This is the specific attribute for Tandem script sync
        self._next_start_idx = 0

        # ---- bookkeeping for render() ----
        self._last_reset_info = None
        self._last_step_info = None
        self._last_reward = None

    def _get_features_for_range(self, start_idx, num_steps):
        """Get feature values for a range of timesteps."""
        if self.feature_cache is None or self.num_features == 0:
            return np.array([])

        end_idx = start_idx + num_steps

        if end_idx > len(self.feature_cache):
            # Handle edge case - pad with last available features
            available = self.feature_cache[start_idx:]
            padding_needed = num_steps - len(available)
            if padding_needed > 0:
                padding = np.tile(self.feature_cache[-1], (padding_needed, 1))
                features = np.vstack([available, padding])
            else:
                features = available
        else:
            features = self.feature_cache[start_idx:end_idx]

        # Flatten: [T, num_features] -> [T * num_features]
        return features.flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = self._next_start_idx

        # Safety check for index out of bounds
        if idx + self.T > self.total_rows:
            idx = 0
            self._next_start_idx = 0

        self._expected = self.pred_data[idx: idx + self.T]
        self._realized = self.actual_data[idx: idx + self.T]

        # Get features for this day
        self._current_features = self._get_features_for_range(idx, self.T)

        # Calculate the current timestamp for this day
        current_time = self.base_timestamp + timedelta(minutes=30 * idx)

        # Return info with names that match your test script and Tandem script
        info = {
            "start_idx": idx,
            "timestamp": current_time
        }

        # store for render()
        self._last_reset_info = info

        # Observation: predicted consumption + features
        obs = np.concatenate([self._expected, self._current_features])

        return obs, info

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

        # store for render()
        self._last_step_info = step_info
        self._last_reward = -cost

        # 3. ADVANCE THE POINTER (Move to the next day)
        self._next_start_idx += self.T

        # Loop if we hit the end
        if self._next_start_idx + self.T > self.total_rows:
            print("!!! REACHED END OF CSV - RESTARTING FROM ROW 0 !!!")
            self._next_start_idx = 0

        # Get the next observation (for the next day)
        next_idx = self._next_start_idx
        if next_idx + self.T > self.total_rows:
            next_idx = 0

        next_expected = self.pred_data[next_idx: next_idx + self.T]
        next_features = self._get_features_for_range(next_idx, self.T)
        next_obs = np.concatenate([next_expected, next_features])

        # One step = One day. Always terminated=True.
        return next_obs, -cost, True, False, step_info

    def render(self):
        """
        Render information for the most recent timestep (a 'day').

        This expects that reset() and step() have been called for the day,
        and uses the stored `_last_reset_info` and `_last_step_info` to
        reproduce the exact console output the old `main()` printed.
        It also returns a record dict containing the same structured data.
        """
        if self._last_reset_info is None or self._last_step_info is None:
            print("Nothing to render yet. Call reset() and step() first for the day.")
            return None

        info = self._last_reset_info
        step_info = self._last_step_info
        reward = self._last_reward

        # Print header for the day (matching original formatting)
        start_row = info['start_idx'] + 2
        end_row = info['start_idx'] + self.T + 1
        print(f"--- [DAY] ---")
        print(f"CSV Row Range: {start_row} to {end_row}")
        print(f"Calendar Date: {info['timestamp'].strftime('%Y-%m-%d')}")

        # Build header dynamically with feature columns
        header = f"{'Time':<10} | {'CSV Row':<8} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}"
        if self.num_features > 0:
            for col in self.feature_columns:
                header += f" | {col:<10}"
        print(header)
        print("-" * (70 + self.num_features * 13))

        rows = []
        for i in range(0, self.T, 4):
            slot_time = info['timestamp'] + timedelta(minutes=30 * i)
            row_num = info['start_idx'] + i + 2

            p = step_info['predicted'][i]
            a = step_info['realized'][i]
            d = step_info['dispatch'][i]
            pr = step_info['prices'][i]

            row_str = f"{slot_time.strftime('%H:%M'):<10} | {row_num:<8} | {p:<8.3f} | {a:<8.3f} | {d:<10.3f} | {pr:<6.2f}"

            row_dict = {
                "slot_time": slot_time,
                "row_num": int(row_num),
                "pred": float(p),
                "actual": float(a),
                "dispatch": float(d),
                "price": float(pr)
            }

            # Add feature values
            if self.num_features > 0:
                feature_start_idx = i * self.num_features
                feature_end_idx = feature_start_idx + self.num_features
                feature_values = self._current_features[feature_start_idx:feature_end_idx]

                for j, col in enumerate(self.feature_columns):
                    val = feature_values[j]
                    row_str += f" | {val:<10.4f}"
                    row_dict[col] = float(val)

            print(row_str)
            rows.append(row_dict)

        print("-" * (70 + self.num_features * 13))
        print(f"Day Summary: Avg MAE = {step_info['mae']:.4f}\n")

        # return a structured record identical in content to what was printed
        record = {
            "csv_range": (int(start_row), int(end_row)),
            "calendar_date": info['timestamp'],
            "rows": rows,
            "mae": float(step_info['mae']),
            "reward": float(reward)
        }
        return record


# ==========================================
# MAIN FUNCTION: THE CLI MONITOR
# ==========================================
if __name__ == "__main__":
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
            actual_csv='../../tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
            predicted_csv='../../tests/gym/data_for_tests/consumption_predictions.csv'
        )

    print(f"\nSimulation Started. Total Records: {env.total_rows}")

    # Run for 3 days, but delegate printing/recording to env.render()
    for day in range(1, 4):  # Look at 3 days
        obs, info = env.reset()

        # Dummy Action (Example: 48 prices, 48 dispatch values)
        # Note: obs now includes features, but we still use just the first 48 values (predictions)
        predictions_only = obs[:env.T]
        dummy_prices = np.linspace(0.1, 0.5, 48)
        action = np.concatenate([dummy_prices, predictions_only])

        _, reward, _, _, step_info = env.step(action)

        # Use the env's render() to print/record the day's info (render handles the formatting)
        env.render()