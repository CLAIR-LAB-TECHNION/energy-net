import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from datetime import datetime, timedelta


class ISOEnv(gym.Env):
    """
    ISO Environment that automatically detects and includes feature columns
    from the predictions CSV in its observation space.

    Changes in this updated version:
    - Action space normalized to [0, 1] (instead of [-1, 1])
    - Configurable scaling parameters: price_scale and dispatch_scale
    - First half of actions (prices) scaled by price_scale
    - Second half of actions (dispatch) scaled by dispatch_scale
    - Robust NaN/Inf guards for features, observations, actions, and rewards
    - Finite observation space bounds
    """

    def __init__(self, actual_csv, predicted_csv, steps_per_day=48,
                 price_scale=1.0, dispatch_scale=6.0):
        super().__init__()

        # 1. Store scaling parameters
        self.price_scale = price_scale
        self.dispatch_scale = dispatch_scale
        self.T = steps_per_day

        # 2. Load your data
        self.actual_df = pd.read_csv(actual_csv)
        self.pred_df = pd.read_csv(predicted_csv)

        # Defensive: fill NaNs in CSVs
        if self.actual_df.isnull().values.any() or self.pred_df.isnull().values.any():
            print("Warning: NaN values detected in CSVs. Filling with forward-fill and zeros.")
            self.pred_df = self.pred_df.fillna(method='ffill').fillna(0.0)
            self.actual_df = self.actual_df.fillna(method='ffill').fillna(0.0)

        # Get the first timestamp from the CSV to use as a clock base
        self.base_timestamp = pd.to_datetime(self.actual_df.iloc[0]['Datetime'])

        self.actual_data = self.actual_df['Consumption'].values.astype(np.float32)
        # Accept either column name 'predicted_consumption' or 'Prediction' etc if needed
        if 'predicted_consumption' in self.pred_df.columns:
            pred_col = 'predicted_consumption'
        else:
            # fallback to the second column excluding timestamp if available
            non_ts_cols = [c for c in self.pred_df.columns if c.lower() not in ('datetime', 'timestamp')]
            if len(non_ts_cols) > 0:
                pred_col = non_ts_cols[0]
            else:
                raise ValueError("No predicted consumption column found in predicted_csv")

        self.pred_data = self.pred_df[pred_col].values.astype(np.float32)

        # 3. Automatically detect feature columns (exclude timestamp and the predicted column)
        excluded_cols = {'timestamp', pred_col}
        # normalize exclusion to lowercase to be robust
        self.feature_columns = [col for col in self.pred_df.columns if col.lower() not in excluded_cols]
        self.num_features = len(self.feature_columns)

        # Cache feature values for fast lookup
        if self.num_features > 0:
            self.feature_cache = self.pred_df[self.feature_columns].values.astype(np.float32)
        else:
            self.feature_cache = None

        self.total_rows = min(len(self.actual_data), len(self.pred_data))

        # 4. Define observation and action spaces
        # Observation: predicted consumption (T values) + features per timestep (T * num_features)
        obs_dim = self.T + (self.T * self.num_features)
        big = 1e6
        self.observation_space = spaces.Box(
            low=-big,
            high=big,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action: normalized action space [0, 1] for both prices and dispatch
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
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
            return np.zeros(self.T * self.num_features, dtype=np.float32)

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
        features = features.flatten().astype(np.float32)
        # replace any NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        return features

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
        obs = np.concatenate([self._expected, self._current_features]).astype(np.float32)

        return obs, info

    def step(self, action):
        """
        Robust step() implementation:
        - accepts numpy/torch actions
        - sanitizes NaN/Inf
        - assumes incoming action is in [0, 1] and scales by price_scale/dispatch_scale
        - computes reward robustly and returns (next_obs, reward, terminated, truncated, info)
        """
        # ---- Convert & normalize incoming action ----
        action = np.asarray(action, dtype=np.float32).flatten()

        # ---- Scale action by configurable parameters ----
        # First half: prices (scaled by price_scale)
        # Second half: dispatch (scaled by dispatch_scale)
        prices_raw = action[:self.T]
        dispatch_raw = action[self.T:]

        prices = prices_raw * self.price_scale
        dispatch = dispatch_raw * self.dispatch_scale

        # ---- Compute robust cost ----
        # compute absolute diff and guard it
        diff = np.abs(dispatch - self._realized)
        cost = float(np.mean(diff))

        reward = -cost

        # ---- Build step info dict (store both raw & scaled for debugging) ----
        current_time = self.base_timestamp + timedelta(minutes=30 * self._next_start_idx)
        step_info = {
            "realized": self._realized.copy(),
            "predicted": self._expected.copy(),
            "dispatch": dispatch.copy(),
            "prices": prices.copy(),
            "mae": cost,
            "start_idx": self._next_start_idx,
            "timestamp": current_time,
            "action_raw": action.copy(),
            "prices_raw": prices_raw.copy(),
            "dispatch_raw": dispatch_raw.copy()
        }

        # store for render()
        self._last_step_info = step_info
        self._last_reward = float(reward)

        # ---- Advance pointer (move to next day) ----
        self._next_start_idx += self.T

        # Loop if we hit the end
        if self._next_start_idx + self.T > self.total_rows:
            print("!!! REACHED END OF CSV - RESTARTING FROM ROW 0 !!!")
            self._next_start_idx = 0

        # ---- Prepare next observation ----
        next_idx = self._next_start_idx
        if next_idx + self.T > self.total_rows:
            next_idx = 0

        next_expected = self.pred_data[next_idx: next_idx + self.T]
        next_features = self._get_features_for_range(next_idx, self.T)

        next_obs = np.concatenate([next_expected, next_features]).astype(np.float32)
        next_obs = np.nan_to_num(next_obs, nan=0.0, posinf=1e6, neginf=-1e6)

        terminated = True
        truncated = False

        return next_obs, float(reward), terminated, truncated, step_info

    def render(self):
        """
        Render information for the most recent timestep (a 'day').
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
# MINIMAL RL TESTING FUNCTION
# ==========================================

def test_with_rl():
    """
    Minimal test using Stable-Baselines3 PPO algorithm.
    Install with: pip install stable-baselines3
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("\n[ERROR] stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        return

    print("\n" + "=" * 60)
    print("TESTING WITH RL ALGORITHM (PPO)")
    print("=" * 60)

    # Create environment with custom scaling parameters
    try:
        env = ISOEnv(
            actual_csv='synthetic_household_consumption_test.csv',
            predicted_csv='consumption_predictions.csv',
            price_scale=1.0,
            dispatch_scale=6.0
        )
    except FileNotFoundError:
        env = ISOEnv(
            actual_csv='../../tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
            predicted_csv='../../tests/gym/data_for_tests/consumption_predictions.csv',
            price_scale=1.0,
            dispatch_scale=6.0
        )

    # Check if environment follows Gym API
    print("\n[1/3] Checking environment compatibility...")
    check_env(env, warn=True)
    print("✓ Environment check passed!")

    # Create PPO agent
    print("\n[2/3] Creating PPO agent...")
    # smaller LR and grad clipping for stability while debugging
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-4, max_grad_norm=0.5)
    print("✓ Agent created!")

    # Train for a few steps (quick test)
    print("\n[3/3] Training for 1000 timesteps...")
    model.learn(total_timesteps=1000)
    print("✓ Training complete!")

    # Test the trained agent for 3 days
    print("\n" + "-" * 60)
    print("TESTING TRAINED AGENT (3 days)")
    print("-" * 60)

    obs, info = env.reset()
    for day in range(3):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        record = env.render()
        print(f"Day {day + 1} Reward: {reward:.4f}")

        if terminated:
            obs, info = env.reset()

    print("\n✓ RL test completed successfully!")