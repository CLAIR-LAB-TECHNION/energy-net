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
import os


class PCSEnv(gym.Env):
    """
    Simplified unit-less Gym environment for PCSUnit.
    - Uses internal "units" everywhere (battery capacity 0..100, consumption ~2 units/step).
    - Actions are normalized in [-1, 1] and passed directly to PCS/Battery.
    - No unit conversions to kWh/kW — everything stays in the same internal units.
    - Loads pre-computed predictions from CSV file.
    - Automatically detects and includes feature columns in observation space.
    """

    def __init__(self,
                 test_data_file='../../tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
                 predictions_file='../../tests/gym/data_for_tests/consumption_predictions.csv',
                 dt=0.5 / 24,  # 30 minutes in days
                 episode_length_days=1,
                 prediction_horizon=48,
                 shortage_penalty=100.0,
                 base_price=0.10,
                 price_volatility=0.15,
                 log_path='../../tests/gym/logs',
                 render_mode: str | None = None):

        super().__init__()

        # -------------------------
        # Directory setup
        # -------------------------
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # -------------------------
        # Basic parameters
        # -------------------------
        self.dt = dt
        self.episode_length_days = episode_length_days
        self.max_steps = int(episode_length_days / dt)
        self.prediction_horizon = prediction_horizon
        self.shortage_penalty = shortage_penalty
        self.base_price = base_price
        self.price_volatility = price_volatility
        self.log_path = log_path
        self.render_mode = render_mode
        self.last_action = None

        # -------------------------
        # Load test data
        # -------------------------
        self.test_df = pd.read_csv(
            test_data_file,
            index_col=0,
            parse_dates=True
        )

        self.test_start_date = self.test_df.index[0]
        self.test_end_date = self.test_df.index[-1]

        # -------------------------
        # Load predictions and automatically detect features
        # -------------------------
        pred_df = pd.read_csv(
            predictions_file,
            parse_dates=['timestamp']
        ).set_index('timestamp').sort_index()

        self.pred_df = pred_df
        self.prediction_cache = pred_df['predicted_consumption'].values
        self.prediction_timestamps = pred_df.index

        # Automatically detect feature columns (everything except timestamp and predicted_consumption)
        excluded_cols = {'predicted_consumption'}
        self.feature_columns = [col for col in pred_df.columns if col not in excluded_cols]
        self.num_features = len(self.feature_columns)

        print(f"Loaded {len(self.prediction_cache)} aligned predictions")
        print(f"Detected {self.num_features} feature columns: {self.feature_columns}")

        # Cache feature values for fast lookup
        if self.num_features > 0:
            self.feature_cache = pred_df[self.feature_columns].values
        else:
            self.feature_cache = None

        # ==============================================================
        # Battery
        # ==============================================================
        battery_dynamics = DeterministicBattery(model_parameters={})
        battery_config = {
            "min": 0.0,
            "max": 100.0,
            "init": 0.0
        }
        battery = Battery(
            dynamics=battery_dynamics,
            config=battery_config,
            log_file=os.path.join(log_path, "storage.log")
        )

        # ==============================================================
        # Consumption unit (test data)
        # ==============================================================
        print("Creating consumption unit on testing data...")
        consumption_dynamics = CSV_DataConsumptionDynamics(
            params={'data_file': test_data_file}
        )
        consumption_unit_config = {
            'data_file': test_data_file,
            'consumption_capacity': 6.0
        }
        consumption_unit = ConsumptionUnit(
            dynamics=consumption_dynamics,
            config=consumption_unit_config,
            log_file=os.path.join(log_path, "consumption_unit.log")
        )

        self.pcs = PCSUnit(
            storage_units=[battery],
            consumption_units=[consumption_unit],
            log_file=os.path.join(log_path, "pcs_unit.log")
        )

        # ==============================================================
        # Gym spaces
        # ==============================================================
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space:
        # [storage, consumption, price] + [predicted_consumption * horizon] + [features]
        obs_dim = 3 + prediction_horizon + self.num_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        print(f"Observation space dimension: {obs_dim}")
        print(f"  - Base features (storage, consumption, price): 3")
        print(f"  - Predicted consumption horizon: {prediction_horizon}")
        print(f"  - Additional time features: {self.num_features}")

        # ==============================================================
        # Episode state
        # ==============================================================
        self.current_step = 0
        self.start_date = self.test_start_date
        self.current_datetime = self.start_date

        self.total_reward = 0.0
        self.total_money_earned = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        # ==============================================================
        # Price curve
        # ==============================================================
        self.price_curve = self._generate_price_curve(self.max_steps)

    def get_money(self):
        """Returns the accumulated money made from transactions, ignoring shortage penalties."""
        return self.total_money_earned

    def _generate_price_curve(self, num_steps):
        """Generate a simple price curve with some variation."""
        prices = []
        for i in range(num_steps):
            time_of_day = (i * self.dt * 24) % 24
            variation = np.sin(2 * np.pi * time_of_day / 24) * self.price_volatility
            price = self.base_price + variation
            prices.append(max(0.01, price))
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

    def _get_current_features(self):
        """Get current feature values from preloaded cache."""
        if self.feature_cache is None or self.num_features == 0:
            return np.array([])

        days_from_start = (self.current_datetime - self.test_start_date).total_seconds() / 86400
        current_idx = int(days_from_start / self.dt)

        if current_idx >= len(self.feature_cache):
            # Use last available features if we're beyond the cache
            return self.feature_cache[-1]

        return self.feature_cache[current_idx]

    def set_price_curve(self, prices: np.ndarray):
        """External setter to inject prices from the ISO agent."""
        if len(prices) != self.max_steps:
            raise ValueError(f"Price curve must have {self.max_steps} steps.")
        self.price_curve = prices

    def reset(self, start_date=None, options=None, seed=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.total_reward = 0.0
        self.total_money_earned = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        if start_date is not None:
            self.current_datetime = start_date
        else:
            if self.current_datetime >= self.test_end_date:
                print("Reached end of dataset. Looping back to start.")
                self.current_datetime = self.test_start_date

        self.pcs.reset(initial_storage_unit_level=0)
        self.last_action = None

        return self._get_obs(), {}

    def step(self, action):
        """
        action: normalized intent in [-10, 10]
        All returned values and checks use the same internal "units".
        """
        raw_action = float(action[0])
        self.last_action = raw_action

        storage_before = self.pcs.get_total_storage()
        battery_entity = self.pcs.storage_units[0]

        current_time = self.current_step * self.dt
        state = State()
        state.set_attribute('time', current_time)

        actions = {"Battery_0": Action({'value': raw_action})}
        energy_sold_or_bought = self.pcs.update(state, actions)

        storage_after = self.pcs.get_total_storage()
        consumption_units = float(self.pcs.get_consumption())

        try:
            available_discharge_units = float(battery_entity.get_available_discharge_capacity())
        except Exception:
            available_discharge_units = getattr(battery_entity, 'available_discharge', 0.0)

        # Calculate reward
        step_money = self._get_current_price() * -energy_sold_or_bought

        shortage = consumption_units > available_discharge_units
        step_penalty = 0.0
        if shortage:
            step_penalty = self.shortage_penalty
            self.shortage_count += 1
            self.total_shortage_penalty += step_penalty

        self.total_money_earned += step_money
        reward = step_money - step_penalty
        self.total_reward += reward

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
            'battery_action': raw_action,
            'step_money': step_money,
            'total_money_so_far': self.total_money_earned
        }
        self._last_action = raw_action

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Build observation vector:
        [storage, consumption, price] + [predicted_consumption...] + [features...]
        """
        # Base observations
        current_storage = self.pcs.get_total_storage() / 100.0
        current_consumption = self.pcs.get_consumption() / 10.0
        current_price = self._get_current_price() / self.base_price

        # Predicted consumption
        predicted_consumption = self._get_predicted_consumption(self.prediction_horizon)
        predicted_consumption = predicted_consumption / 10.0

        # Current time features
        current_features = self._get_current_features()

        # Concatenate all parts
        obs = np.concatenate([
            [current_storage, current_consumption, current_price],
            predicted_consumption,
            current_features
        ])

        return obs.astype(np.float32)

    def render(self):
        if self.render_mode == 'human':
            print(f"\n{'=' * 60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Date: {self.current_datetime.strftime('%Y-%m-%d %H:%M')}")
            print(f"Storage: {self.pcs.get_total_storage():.2f} units (capacity=100)")
            print(f"Consumption (will affect next time step): {self.pcs.get_consumption():.2f} units/step")
            if self.last_action is not None:
                print(f"Last Action (Battery intent): {self.last_action:.2f} units")
            print(f"Current Price: {self._get_current_price():.4f} $/unit")

            # Show current features if available
            if self.num_features > 0:
                features = self._get_current_features()
                print(f"Current Features:")
                for i, col in enumerate(self.feature_columns):
                    print(f"  {col}: {features[i]:.4f}")

            print(f"Total Reward (Net): ${self.total_reward:.2f}")
            print(f"Total Money (Gross): ${self.total_money_earned:.2f}")
            print(f"Total Penalties: ${self.total_shortage_penalty:.2f}")
            print(f"Shortages: {self.shortage_count}")
            print(f"{'=' * 60}")

    def get_step_index(self) -> int:
        """
        Return the global half-hour step index since test_start_date.
        """
        days_from_start = (self.current_datetime - self.test_start_date).total_seconds() / 86400
        return int(days_from_start / self.dt)

    def set_step_index(self, idx: int):
        """
        Set the environment's current_datetime and intra-day current_step
        from a global half-hour index.
        """
        idx = int(idx)
        self.current_datetime = self.test_start_date + timedelta(minutes=30 * idx)
        self.current_step = int(idx % self.max_steps)


if __name__ == "__main__":
    env = PCSEnv(render_mode='human')

    num_days_to_run = 3
    accumulated_reward = 0.0
    render_every_n_steps = 1

    for day in range(num_days_to_run):
        obs, info = env.reset()
        env.render()

        print(f"\n{'=' * 60}")
        print(f"STARTING DAY {day + 1}  |  Date: {env.current_datetime.strftime('%Y-%m-%d')}")
        print(f"{'=' * 60}")

        day_reward = 0.0

        while True:
            action_value = float(np.random.uniform(-10.0, 10.0))
            action = np.array([action_value], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if env.render_mode == "human" and env.current_step % render_every_n_steps == 0:
                env.render()

            day_reward += reward
            accumulated_reward += reward

            print(f"Step Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"\n>>> Day {day + 1} Finished!")
                print(f">>> Total Day Reward: {day_reward:.2f}")
                print(f">>> Final Storage: {info['storage_after_units']:.2f}")
                print(f">>> Total Shortages Today: {env.shortage_count}")
                break

    print(f"\n{'!' * 60}")
    print(f"Simulation Complete. Total Reward over {num_days_to_run} days: {accumulated_reward:.2f}")
    print(f"{'!' * 60}")