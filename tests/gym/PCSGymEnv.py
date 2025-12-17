import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from energy_net.grid_entities.consumption.consumption_dynamics import CSV_DataConsumptionDynamics
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.consumption_prediction.predicting_consumption_model import (
    create_predictor,
    predict_consumption,
    generate_day_predictions
)
from energy_net.foundation.model import State, Action
from datetime import timedelta
import random
import numpy as np

class PCSGymEnv(gym.Env):
    """
    Simplified unitless Gym environment for PCSUnit.
    - Uses internal "units" everywhere (battery capacity 0..100, consumption ~2 units/step).
    - Actions are normalized in [-1, 1] and passed directly to PCS/Battery.
    - No unit conversions to kWh/kW — everything stays in the same internal units.
    """

    def __init__(self,
                 data_file='synthetic_household_consumption.csv',
                 dt=0.5 / 24,  # 30 minutes in days (kept for timestep logic but not used for unit conversions)
                 episode_length_days=1,
                 train_test_split=0.8,
                 prediction_horizon=48,
                 shortage_penalty=5.0,
                 base_price=0.10,
                 price_volatility=0.15):

        super().__init__()

        self.dt = dt
        self.episode_length_days = episode_length_days
        self.max_steps = int(episode_length_days / dt)
        self.prediction_horizon = prediction_horizon
        self.shortage_penalty = shortage_penalty
        self.base_price = base_price
        self.price_volatility = price_volatility

        # --- Load and split data ---
        print("Loading data and splitting into train/test sets...")
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if df.empty:
            raise ValueError(f"The data file {data_file} is empty or could not be loaded.")

        split_idx = int(len(df) * train_test_split)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        train_file = data_file.replace('.csv', '_train.csv')
        train_df.to_csv(train_file)
        print(f"Training data: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"Testing data: {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")

        test_file = data_file.replace('.csv', '_test.csv')
        test_df.to_csv(test_file)

        self.test_start_date = test_df.index[0]
        self.test_end_date = test_df.index[-1]

        # Predictor on training data only
        print("Creating consumption predictor on training data...")
        self.predictor = create_predictor(train_file)

        # Pre-generate predictions for test period
        print("Pre-generating all consumption predictions for test period...")
        self._pregenerate_all_predictions()
        print(f"Generated {len(self.prediction_cache)} predictions")

        # Battery (internal units: 0..100)
        battery_dynamics = DeterministicBattery(model_parameters={})
        battery_config = {
            "min": 0.0,
            "max": 100.0,       # battery capacity in internal units
            "charge_rate_max": 2e5,
            "discharge_rate_max": 2e5,
            "init": 0.0
        }
        battery = Battery(dynamics=battery_dynamics, config=battery_config)

        # Consumption unit using test data
        print("Creating consumption unit on testing data...")
        consumption_dynamics = CSV_DataConsumptionDynamics(params={'data_file': test_file})
        consumption_unit_config = {
            'data_file': test_file,
            'consumption_capacity': 6.0  # this stays as whatever your CSV provides; we treat it as internal units
        }
        consumption_unit = ConsumptionUnit(dynamics=consumption_dynamics, config=consumption_unit_config)

        self.pcs = PCSUnit(storage_units=[battery], consumption_units=[consumption_unit])

        # Action space: normalized [-1, 1]
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        # Observation: [current_storage, current_consumption, current_price, next N predicted consumptions]
        obs_dim = 3 + prediction_horizon
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Episode tracking
        self.current_step = 0
        self.start_date = self.test_start_date
        self.current_datetime = self.start_date
        self.total_profit = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        # Price curve
        self.price_curve = self._generate_price_curve(self.max_steps)

    def _pregenerate_all_predictions(self):
        total_days = (self.test_end_date - self.test_start_date).days + 1
        total_steps = int(total_days / self.dt) + self.prediction_horizon

        self.prediction_cache = []
        current_dt = self.test_start_date

        print(f"Generating predictions from {self.test_start_date} to {self.test_end_date}...")
        for step in range(total_steps):
            date_str = current_dt.strftime("%Y-%m-%d")
            time_str = current_dt.strftime("%H:%M")
            try:
                pred = predict_consumption(self.predictor, date_str, time_str)
            except Exception as e:
                pred = 0.0
                if step < 10:
                    print(f"Warning: Prediction failed for {date_str} {time_str}: {e}")
            self.prediction_cache.append(pred)
            current_dt += timedelta(days=self.dt)

        self.prediction_cache = np.array(self.prediction_cache)

    def _generate_price_curve(self, num_steps):
        t = np.arange(num_steps)
        daily_pattern = np.sin(2 * np.pi * t * self.dt) * 0.3
        weekly_pattern = np.sin(2 * np.pi * t * self.dt / 7) * 0.15
        noise = np.random.randn(num_steps) * self.price_volatility
        prices = self.base_price + daily_pattern + weekly_pattern + noise
        prices = np.maximum(prices, 0.0001)
        return prices

    def _get_current_price(self):
        if self.price_curve is None or self.current_step >= len(self.price_curve):
            return self.base_price
        return self.price_curve[self.current_step]

    def _get_predicted_consumption(self, num_steps):
        days_from_start = (self.current_datetime - self.test_start_date).total_seconds() / 86400
        current_idx = int(days_from_start / self.dt)
        end_idx = current_idx + num_steps

        if end_idx > len(self.prediction_cache):
            available = self.prediction_cache[current_idx:]
            padding = np.full(num_steps - len(available), self.prediction_cache[-1])
            return np.concatenate([available, padding])

        return self.prediction_cache[current_idx:end_idx]

    def reset(self, start_date=None, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.total_profit = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        if start_date is not None:
            self.start_date = start_date
        else:
            self.start_date = self.test_start_date

        self.current_datetime = self.start_date
        self.price_curve = self._generate_price_curve(self.max_steps)
        self.pcs.reset(initial_storage_unit_level=0)

        return self._get_obs(), {}

    def step(self, action):
        """
        action: normalized intent in [-1, 1]
        All returned values and checks use the same internal "units".
        """

        # Clip and interpret normalized action
        raw_action = float(action[0])

        # Storage before (internal units, 0..100)
        storage_before = self.pcs.get_total_storage()

        # Battery entity
        battery_entity = self.pcs.storage_units[0]

        # Current timestep state
        current_time = self.current_step * self.dt
        state = State()
        state.set_attribute('time', current_time)

        # Pass normalized intent directly to PCS/Battery
        actions = {"Battery_0": Action({'value': raw_action})}

        # Apply PCS update. energy_sold_or_bought is returned in the same internal units (storage_after - storage_before)
        energy_sold_or_bought = self.pcs.update(state, actions)

        storage_after = self.pcs.get_total_storage()

        # Total consumption reported by PCS — treated as internal units (power-like but we use as "units per step")
        consumption_units = float(self.pcs.get_consumption())

        # Available discharge reported by battery (assumed in same internal units)
        try:
            available_discharge_units = float(battery_entity.get_available_discharge_capacity())
        except Exception:
            # fallback if method not present
            available_discharge_units = getattr(battery_entity, 'available_discharge', 0.0)

        # Shortage check: compare consumption_units to available_discharge_units (same unit system)
        shortage = consumption_units > available_discharge_units

        # Compute reward using price per internal unit
        reward = 0.0
        if shortage:
            reward -= self.shortage_penalty
            self.shortage_count += 1
            self.total_shortage_penalty += self.shortage_penalty

        # Money for moved units (price is per internal unit)
        reward += self._get_current_price() * -energy_sold_or_bought
        self.total_profit += reward

        # Advance time
        self.current_step += 1
        self.current_datetime += timedelta(days=self.dt)

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()

        # Info in internal units for easy human-reading
        info = {
            'storage_before_units': storage_before,
            'storage_after_units': storage_after,
            'energy_sold_or_bought_units': energy_sold_or_bought,
            'consumption_units': consumption_units,
            'available_discharge_units': available_discharge_units,
            'shortage': shortage,
            'battery_action_normalized': raw_action,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Normalizations in observation are kept as before but operate on internal units:
        current_storage = self.pcs.get_total_storage() / 100.0  # normalize 0..100 -> 0..1
        current_consumption = self.pcs.get_consumption() / 10.0  # scale-down so the value is reasonable in obs
        current_price = self._get_current_price() / self.base_price

        predicted_consumption = self._get_predicted_consumption(self.prediction_horizon)
        predicted_consumption = predicted_consumption / 10.0

        obs = np.concatenate([
            [current_storage, current_consumption, current_price],
            predicted_consumption
        ])

        return obs.astype(np.float32)

    def render(self, mode='human'):
        if mode == 'human':
            print(f"\n{'=' * 60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Date: {self.current_datetime.strftime('%Y-%m-%d %H:%M')}")
            print(f"Storage: {self.pcs.get_total_storage():.2f} units (capacity=100)")
            print(f"Consumption: {self.pcs.get_consumption():.2f} units/step")
            print(f"Current Price: {self._get_current_price():.4f} $/unit")
            print(f"Total Profit: ${self.total_profit:.2f}")
            print(f"Total Penalties: ${self.total_shortage_penalty:.2f}")
            print(f"Shortages: {self.shortage_count}")
            print(f"{'=' * 60}")


if __name__ == "__main__":
    env = PCSGymEnv()
    obs, info = env.reset()

    accumulated_reward = 0.0
    num_steps = 20

    print("\n=== Debugging Random Actions in main() (uniform continuous) ===\n")
    for step in range(num_steps):
        # sample fresh each step from continuous uniform
        action_value = float(np.random.uniform(-10.0, 10.0))


        action = np.array([action_value], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        accumulated_reward += reward

        print(f"Step {step + 1}")
        print(f"  Battery action (normalized): {action[0]:.3f}")
        print(f"  Energy bought or sold (internal units): {info['energy_sold_or_bought_units']:.3f} units")
        print(f"  Consumption this step: {info['consumption_units']:.3f} units")
        print(f"  Available discharge now: {info['available_discharge_units']:.3f} units")
        print(f"  Storage before/after: {info['storage_before_units']:.2f} / {info['storage_after_units']:.2f}")
        print(f"  Step reward: {reward:.2f}")
        print(f"  Accumulated reward: {accumulated_reward:.2f}")
        print(f"  Shortage: {info['shortage']}")
        print("-" * 50)

        if terminated or truncated:
            print("Environment terminated early.")
            break
