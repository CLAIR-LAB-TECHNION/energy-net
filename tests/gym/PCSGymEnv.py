import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from energy_net.grid_entities.consumption.consumption_dynamics import CSV_DataConsumptionDynamics
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.foundation.model import State, Action
from datetime import timedelta
import numpy as np


class PCSGymEnv(gym.Env):
    """
    Simplified unit-less Gym environment for PCSUnit.
    - Uses internal "units" everywhere (battery capacity 0..100, consumption ~2 units/step).
    - Actions are normalized in [-1, 1] and passed directly to PCS/Battery.
    - No unit conversions to kWh/kW — everything stays in the same internal units.
    - Loads pre-computed predictions from CSV file.
    """

    def __init__(self,
                 test_data_file='synthetic_household_consumption_test.csv',
                 predictions_file='consumption_predictions.csv',
                 dt=0.5 / 24,  # 30 minutes in days
                 episode_length_days=1,
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

        # Load pre-computed predictions
        print(f"Loading pre-computed predictions from {predictions_file}...")
        predictions_df = pd.read_csv(predictions_file, parse_dates=['timestamp'])
        self.prediction_cache = predictions_df['predicted_consumption'].values
        self.prediction_timestamps = pd.to_datetime(predictions_df['timestamp'])
        print(f"Loaded {len(self.prediction_cache)} predictions")

        # Load test data info
        print(f"Loading test data from {test_data_file}...")
        test_df = pd.read_csv(test_data_file, index_col=0, parse_dates=True)

        # Round test start to beginning of day if needed
        initial_start = test_df.index[0]
        if initial_start.hour != 0 or initial_start.minute != 0 or initial_start.second != 0:
            self.test_start_date = (initial_start + timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                                               microsecond=0)
            print(f"Rounded test start from {initial_start} to {self.test_start_date}")
            # Filter test_df to only include data from the rounded start date
            test_df = test_df[test_df.index >= self.test_start_date]
        else:
            self.test_start_date = initial_start

        self.test_end_date = test_df.index[-1]
        print(f"Testing data: {len(test_df)} rows ({self.test_start_date} to {self.test_end_date})")

        # Battery (internal units: 0..100)
        battery_dynamics = DeterministicBattery(model_parameters={})
        battery_config = {
            "min": 0.0,
            "max": 100.0,  # battery capacity in internal units
            "charge_rate_max": 2e5,
            "discharge_rate_max": 2e5,
            "init": 0.0
        }
        battery = Battery(dynamics=battery_dynamics, config=battery_config)

        # Consumption unit using test data
        print("Creating consumption unit on testing data...")
        consumption_dynamics = CSV_DataConsumptionDynamics(params={'data_file': test_data_file})
        consumption_unit_config = {
            'data_file': test_data_file,
            'consumption_capacity': 6.0
        }
        consumption_unit = ConsumptionUnit(dynamics=consumption_dynamics, config=consumption_unit_config)

        self.pcs = PCSUnit(storage_units=[battery], consumption_units=[consumption_unit])

        # Action space: normalized [-10, 10]
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

    def _generate_price_curve(self, num_steps):
        """Generate a simple price curve with some variation."""
        prices = []
        for i in range(num_steps):
            # Simple sinusoidal price variation
            time_of_day = (i * self.dt * 24) % 24  # hour of day
            variation = np.sin(2 * np.pi * time_of_day / 24) * self.price_volatility
            price = self.base_price + variation
            prices.append(max(0.01, price))  # ensure positive
        return np.array(prices)

    def _get_current_price(self):
        if self.price_curve is None or self.current_step >= len(self.price_curve):
            return self.base_price
        return self.price_curve[self.current_step]

    def _get_predicted_consumption(self, num_steps):
        """Get predicted consumption from preloaded cache."""
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
        action: normalized intent in [-10, 10]
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

        # Apply PCS update
        energy_sold_or_bought = self.pcs.update(state, actions)

        storage_after = self.pcs.get_total_storage()

        # Total consumption reported by PCS
        consumption_units = float(self.pcs.get_consumption())

        # Available discharge reported by battery
        try:
            available_discharge_units = float(battery_entity.get_available_discharge_capacity())
        except Exception:
            available_discharge_units = getattr(battery_entity, 'available_discharge', 0.0)

        # Shortage check
        shortage = consumption_units > available_discharge_units

        # Compute reward
        reward = 0.0
        if shortage:
            reward -= self.shortage_penalty
            self.shortage_count += 1
            self.total_shortage_penalty += self.shortage_penalty

        # Money for moved units
        reward += self._get_current_price() * -energy_sold_or_bought
        self.total_profit += reward

        # Advance time
        self.current_step += 1
        self.current_datetime += timedelta(days=self.dt)

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()

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
        # Normalizations in observation
        current_storage = self.pcs.get_total_storage() / 100.0
        current_consumption = self.pcs.get_consumption() / 10.0
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
    # First, generate predictions if not already done
    # Uncomment and run generate_predictions.py first, or:
    # from generate_predictions import generate_and_save_predictions
    # generate_and_save_predictions()

    env = PCSGymEnv()
    obs, info = env.reset()

    accumulated_reward = 0.0
    num_steps = 20

    print("\n=== Debugging Random Actions in main() (uniform continuous) ===\n")
    for step in range(num_steps):
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