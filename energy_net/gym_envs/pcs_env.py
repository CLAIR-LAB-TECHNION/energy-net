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
    """

    def __init__(self,
                 test_data_file='data_for_tests/synthetic_household_consumption_test.csv',
                 predictions_file='data_for_tests/consumption_predictions.csv',
                 dt=0.5 / 24,  # 30 minutes in days
                 episode_length_days=1,
                 prediction_horizon=48,
                 shortage_penalty=5.0,
                 base_price=0.10,
                 price_volatility=0.15,
                 log_path='../../tests/gym/logs'):  # <--- New optional parameter

        super().__init__()

        # -------------------------
        # Directory setup
        # -------------------------
        # Ensure the directory exists so the loggers don't crash
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
        self.log_path = log_path  # Storing it in case it's needed later

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
        # Load predictions
        # -------------------------
        pred_df = pd.read_csv(
            predictions_file,
            parse_dates=['timestamp']
        ).set_index('timestamp').sort_index()

        self.pred_df = pred_df
        self.prediction_cache = pred_df['predicted_consumption'].values
        self.prediction_timestamps = pred_df.index
        print(f"Loaded {len(self.prediction_cache)} aligned predictions")

        # ==============================================================
        # Battery
        # ==============================================================
        battery_dynamics = DeterministicBattery(model_parameters={})
        battery_config = {
            "min": 0.0,
            "max": 100.0,
            "init": 0.0
        }
        # Dynamic pathing using os.path.join for safety
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

        obs_dim = 3 + prediction_horizon
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

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

    def set_price_curve(self, prices: np.ndarray):
        """External setter to inject prices from the ISO agent."""
        if len(prices) != self.max_steps:
            raise ValueError(f"Price curve must have {self.max_steps} steps.")
        self.price_curve = prices

    def reset(self, start_date=None, options=None, seed=None):
        # 1. Standard Gymnasium seeding
        super().reset(seed=seed)

        # 2. Reset intra-day progress ONLY
        self.current_step = 0
        self.total_reward = 0.0
        self.total_money_earned = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        # 3. Handle the Calendar
        if start_date is not None:
            # Manual override (e.g., if you want to jump to a specific day)
            self.current_datetime = start_date
        else:
            # Check if we are at the end of the dataset
            if self.current_datetime >= self.test_end_date:
                print("Reached end of dataset. Looping back to start.")
                self.current_datetime = self.test_start_date

            # LOGIC: If we don't change self.current_datetime here,
            # it stays at the value it reached at the end of the last episode
            # (which is 00:00 of the next day).

        # 4. Wipe physical state (battery)
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

        # ----------------------------------------------------------
        # Split Reward Logic
        # ----------------------------------------------------------

        # 1. Calculate transaction money (Price * units moved)
        # Note: -energy_sold_or_bought because negative = selling = revenue
        step_money = self._get_current_price() * -energy_sold_or_bought

        # 2. Shortage check and penalty
        shortage = consumption_units > available_discharge_units
        step_penalty = 0.0
        if shortage:
            step_penalty = self.shortage_penalty
            self.shortage_count += 1
            self.total_shortage_penalty += step_penalty

        # 3. Update trackers
        self.total_money_earned += step_money

        # The step reward for the RL agent includes the shortage penalty
        reward = step_money - step_penalty
        self.total_reward += reward

        # ----------------------------------------------------------

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
            'step_money': step_money,
            'total_money_so_far': self.total_money_earned
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
            print(f"Total Reward (Net): ${self.total_reward:.2f}")
            print(f"Total Money (Gross): ${self.total_money_earned:.2f}")
            print(f"Total Penalties: ${self.total_shortage_penalty:.2f}")
            print(f"Shortages: {self.shortage_count}")
            print(f"{'=' * 60}")



if __name__ == "__main__":
    env = PCSEnv()

    # Run for 3 full days
    num_days_to_run = 3
    accumulated_reward = 0.0

    for day in range(num_days_to_run):
        # 1. Reset at the start of every day
        obs, info = env.reset()

        print(f"\n" + "=" * 60)
        print(f"STARTING DAY {day + 1}")
        print(f"Current Date in Simulation: {env.current_datetime.strftime('%Y-%m-%d')}")
        print("=" * 60)

        # 2. Print Initial State (T=0) before any steps are taken
        print(f"INITIAL STATE (Time: {env.current_datetime.strftime('%H:%M')})")
        print(f"  Storage: {env.pcs.get_total_storage():.2f} units")
        print(f"  Accumulated Reward so far: {accumulated_reward:.2f}")
        print(f"  Starting Price: {env._get_current_price():.4f} $/unit")
        print("-" * 50)

        day_reward = 0.0

        # 3. Run until the day is over
        while True:
            action_value = float(np.random.uniform(-10.0, 10.0))
            action = np.array([action_value], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            day_reward += reward
            accumulated_reward += reward

            print(f"Step {env.current_step} | Time: {env.current_datetime.strftime('%H:%M')} | "
                  f"Storage: {info['storage_after_units']:.2f} | "
                  f"Consumption (from timestep t-1): {info['consumption_units']:.2f} | "
                  f"Step Reward (from timestep t-1): {reward:.2f}")

            if terminated or truncated:
                print(f"\n>>> Day {day + 1} Finished!")
                print(f">>> Total Day Reward: {day_reward:.2f}")
                print(f">>> Final Storage: {info['storage_after_units']:.2f}")
                print(f">>> Total Shortages Today: {env.shortage_count}")
                break

    print(f"\n" + "!" * 60)
    print(f"Simulation Complete. Total Reward over {num_days_to_run} days: {accumulated_reward:.2f}")
    print("!" * 60)