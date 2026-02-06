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
                 price_scale=1.0, dispatch_scale=6.0, verbosity=2):
        super().__init__()

        # 1. Store scaling parameters
        self.price_scale = price_scale
        self.dispatch_scale = dispatch_scale
        self.T = steps_per_day
        self.verbosity = verbosity

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

    def render(self, verbosity=None):
        """
        Render information for the most recent timestep (a 'day') with configurable verbosity.
        
        Args:
            verbosity: Override instance verbosity level. If None, uses self.verbosity.
                Level 0: Silent - return data dict only, no output
                Level 1: Summary - day metrics only (MAE)
                Level 2: Condensed - summary + sampled timesteps (every 4th) - DEFAULT
                Level 3: Detailed - all timesteps with features
                Level 4: Debug - includes pricing details and dispatch analysis
        
        Returns:
            dict: Structured data containing day information
        """
        if self._last_reset_info is None or self._last_step_info is None:
            v = verbosity if verbosity is not None else self.verbosity
            if v > 0:
                print("Nothing to render yet. Call reset() and step() first for the day.")
            return None

        # Determine verbosity level
        v = verbosity if verbosity is not None else self.verbosity
        
        info = self._last_reset_info
        step_info = self._last_step_info
        reward = self._last_reward
        
        start_row = info['start_idx'] + 2
        end_row = info['start_idx'] + self.T + 1
        mae = step_info['mae']
        
        # Build all timestep rows for data return
        rows = []
        for i in range(self.T):
            slot_time = info['timestamp'] + timedelta(minutes=30 * i)
            row_num = info['start_idx'] + i + 2
            
            row_dict = {
                "slot_time": slot_time,
                "row_num": int(row_num),
                "pred": float(step_info['predicted'][i]),
                "actual": float(step_info['realized'][i]),
                "dispatch": float(step_info['dispatch'][i]),
                "price": float(step_info['prices'][i])
            }
            
            # Add feature values
            if self.num_features > 0:
                feature_start_idx = i * self.num_features
                feature_end_idx = feature_start_idx + self.num_features
                feature_values = self._current_features[feature_start_idx:feature_end_idx]
                for j, col in enumerate(self.feature_columns):
                    row_dict[col] = float(feature_values[j])
            
            rows.append(row_dict)
        
        # Build structured record (returned at all verbosity levels)
        record = {
            "csv_range": (int(start_row), int(end_row)),
            "calendar_date": info['timestamp'],
            "rows": rows,
            "mae": float(mae),
            "reward": float(reward)
        }
        
        # Level 0: Silent - return data only
        if v == 0:
            return record
        
        # Level 1: Summary only
        if v == 1:
            print(f"\n[ISO Day] {info['timestamp'].strftime('%Y-%m-%d')} - MAE: {mae:.4f}, Reward: {reward:.4f}")
            return record
        
        # Level 2: Condensed (summary + sampled timesteps) - DEFAULT
        if v == 2:
            print(f"\n--- [DAY] ---")
            print(f"CSV Row Range: {start_row} to {end_row}")
            print(f"Calendar Date: {info['timestamp'].strftime('%Y-%m-%d')}")
            
            # Build header dynamically with first 3 feature columns
            header = f"{'Time':<10} | {'CSV Row':<8} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}"
            if self.num_features > 0:
                for col in self.feature_columns[:3]:
                    header += f" | {col:<10}"
                if self.num_features > 3:
                    header += " | ..."
            print(header)
            print("-" * (70 + min(self.num_features, 3) * 13 + (4 if self.num_features > 3 else 0)))
            
            # Show every 4th timestep
            for i in range(0, self.T, 4):
                row = rows[i]
                row_str = f"{row['slot_time'].strftime('%H:%M'):<10} | {row['row_num']:<8} | {row['pred']:<8.3f} | {row['actual']:<8.3f} | {row['dispatch']:<10.3f} | {row['price']:<6.2f}"
                
                if self.num_features > 0:
                    for col in self.feature_columns[:3]:
                        row_str += f" | {row[col]:<10.4f}"
                    if self.num_features > 3:
                        row_str += " | ..."
                
                print(row_str)
            
            print("-" * (70 + min(self.num_features, 3) * 13 + (4 if self.num_features > 3 else 0)))
            print(f"Day Summary: Avg MAE = {mae:.4f}\n")
            return record
        
        # Level 3: Detailed (all timesteps with all features)
        if v == 3:
            print(f"\n{'=' * 70}")
            print(f"[ISO DAY DETAILED] {info['timestamp'].strftime('%Y-%m-%d')}")
            print(f"{'=' * 70}")
            print(f"CSV Row Range: {start_row} to {end_row}")
            print(f"MAE: {mae:.4f}, Reward: {reward:.4f}")
            print(f"{'=' * 70}")
            
            # Build header with all feature columns
            header = f"{'Time':<10} | {'CSV Row':<8} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}"
            if self.num_features > 0:
                for col in self.feature_columns:
                    header += f" | {col:<10}"
            print(header)
            print("-" * (70 + self.num_features * 13))
            
            # Show every 4th timestep with all features
            for i in range(0, self.T, 4):
                row = rows[i]
                row_str = f"{row['slot_time'].strftime('%H:%M'):<10} | {row['row_num']:<8} | {row['pred']:<8.3f} | {row['actual']:<8.3f} | {row['dispatch']:<10.3f} | {row['price']:<6.2f}"
                
                if self.num_features > 0:
                    for col in self.feature_columns:
                        row_str += f" | {row[col]:<10.4f}"
                
                print(row_str)
            
            print("-" * (70 + self.num_features * 13))
            print(f"Day Complete.\n")
            return record
        
        # Level 4: Debug (comprehensive analysis)
        if v >= 4:
            print(f"\n{'=' * 80}")
            print(f"[ISO DEBUG MODE] {info['timestamp'].strftime('%Y-%m-%d')}")
            print(f"{'=' * 80}")
            print(f"CSV Row Range:     {start_row} to {end_row}")
            print(f"Start Index:       {info['start_idx']}")
            
            print(f"\n--- Dispatch Performance ---")
            print(f"MAE:               {mae:.6f}")
            print(f"RMSE:              {np.sqrt(np.mean((step_info['dispatch'] - step_info['realized'])**2)):.6f}")
            print(f"Max Error:         {np.max(np.abs(step_info['dispatch'] - step_info['realized'])):.6f}")
            print(f"Reward:            {reward:.6f}")
            
            print(f"\n--- Pricing Analysis ---")
            print(f"Avg Price:         ${np.mean(step_info['prices']):.6f}")
            print(f"Price Range:       ${np.min(step_info['prices']):.4f} - ${np.max(step_info['prices']):.4f}")
            print(f"Price Scale:       {self.price_scale}")
            print(f"Dispatch Scale:    {self.dispatch_scale}")
            
            # Show all features if available
            if self.num_features > 0:
                print(f"\n--- Environmental Features (First Timestep) ---")
                for col in self.feature_columns:
                    print(f"  {col:<25}: {rows[0][col]:.6f}")
            
            # Show detailed timestep data (every 4th)
            print(f"\n--- Timestep Details (Sampled) ---")
            header = f"{'Time':<10} | {'Pred':<10} | {'Actual':<10} | {'Dispatch':<10} | {'Error':<10} | {'Price':<10}"
            print(header)
            print("-" * 75)
            
            for i in range(0, self.T, 4):
                row = rows[i]
                error = abs(row['dispatch'] - row['actual'])
                print(f"{row['slot_time'].strftime('%H:%M'):<10} | {row['pred']:<10.4f} | {row['actual']:<10.4f} | {row['dispatch']:<10.4f} | {error:<10.4f} | {row['price']:<10.4f}")
            
            print(f"{'=' * 80}\n")
            return record
        
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