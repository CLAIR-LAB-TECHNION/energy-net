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
from datetime import datetime, timedelta


class PCSGymEnv(gym.Env):
    """
    Enhanced Gym environment for PCSUnit with:
    - Consumption prediction
    - Dynamic electricity pricing
    - Economic rewards based on buying/selling electricity
    - Penalties for shortage
    """

    def __init__(self,
                 data_file='SystemDemand_30min_2023-2025.csv',
                 dt=0.5 / 24,  # 30 minutes in days
                 episode_length_days=1,
                 train_test_split=0.8,  # 80% train, 20% test
                 prediction_horizon=48,  # Number of future time steps to predict
                 shortage_penalty=1000.0,  # $/kWh penalty for shortage
                 base_price=0.10,  # Base electricity price $/kWh
                 price_volatility=0.05):  # Price variation

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

        # Split data chronologically
        split_idx = int(len(df) * train_test_split)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Save train data to temporary file for predictor
        train_file = data_file.replace('.csv', '_train.csv')
        train_df.to_csv(train_file)
        print(f"Training data: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"Testing data: {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")

        # Save test data to temporary file for consumption unit
        test_file = data_file.replace('.csv', '_test.csv')
        test_df.to_csv(test_file)

        # Store test date range
        self.test_start_date = test_df.index[0]
        self.test_end_date = test_df.index[-1]

        # --- Initialize predictor on TRAINING data only ---
        print("Creating consumption predictor on training data...")
        self.predictor = create_predictor(train_file)

        # ✅ NEW: Pre-generate ALL predictions for entire test period
        print("Pre-generating all consumption predictions for test period...")
        self._pregenerate_all_predictions()
        print(f"Generated {len(self.prediction_cache)} predictions")

        # --- New battery ---
        battery_dynamics = DeterministicBattery(
            model_parameters={
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.9,
                "lifetime_constant": 1e6
            }
        )
        battery_config = {
            "min": 0.0,
            "max": 2e6,  # 2 MWh
            "charge_rate_max": 2e5,  # 200 kW
            "discharge_rate_max": 2e5,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.9,
            "init": 1e6  # Start at 50% capacity
        }
        battery = Battery(dynamics=battery_dynamics, config=battery_config)

        # --- New consumption unit using TEST data ---
        print("Creating consumption unit on testing data...")
        consumption_dynamics = CSV_DataConsumptionDynamics(params={'data_file': test_file})
        consumption_unit_config = {
            'data_file': test_file,
            'consumption_capacity': 12000.0
        }
        consumption_unit = ConsumptionUnit(
            dynamics=consumption_dynamics,
            config=consumption_unit_config
        )

        # --- Build PCSUnit ---
        self.pcs = PCSUnit(
            storage_units=[battery],
            consumption_units=[consumption_unit]
        )

        # --- Action space: battery charge/discharge rate ---
        # Negative = discharge (sell), Positive = charge (buy)
        self.action_space = spaces.Box(
            low=-2e5,  # Max discharge rate
            high=2e5,  # Max charge rate
            shape=(1,),
            dtype=np.float32
        )

        # --- Observation space ---
        # [current_storage, current_consumption, current_price,
        #  next N predicted consumptions]
        obs_dim = 3 + prediction_horizon
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.start_date = self.test_start_date
        self.current_datetime = self.start_date
        self.total_profit = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        # Generate initial price curve (hidden from agent)
        self.price_curve = self._generate_price_curve(self.max_steps)

    def _pregenerate_all_predictions(self):
        """Generate all predictions for the entire test period once"""
        # Calculate total number of timesteps in test period
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
                # Use a default value if prediction fails
                pred = 0.0
                if step < 10:  # Only print first few warnings to avoid spam
                    print(f"Warning: Prediction failed for {date_str} {time_str}: {e}")

            self.prediction_cache.append(pred)
            current_dt += timedelta(days=self.dt)

        self.prediction_cache = np.array(self.prediction_cache)

    def _generate_price_curve(self, num_steps):
        """Generate a realistic price curve with daily and weekly patterns"""
        t = np.arange(num_steps)

        # Daily pattern (peak during day, low at night)
        daily_pattern = np.sin(2 * np.pi * t * self.dt) * 0.3

        # Weekly pattern (higher on weekdays)
        weekly_pattern = np.sin(2 * np.pi * t * self.dt / 7) * 0.15

        # Random variations
        noise = np.random.randn(num_steps) * self.price_volatility

        # Combine patterns
        prices = self.base_price + daily_pattern + weekly_pattern + noise

        # Ensure prices are positive
        prices = np.maximum(prices, 0.01)

        return prices

    def _get_current_price(self):
        """Get current electricity price"""
        if self.price_curve is None or self.current_step >= len(self.price_curve):
            return self.base_price
        return self.price_curve[self.current_step]

    def _get_predicted_consumption(self, num_steps):
        """Get predictions from pre-generated cache"""
        # Calculate index offset from test start date
        days_from_start = (self.current_datetime - self.test_start_date).total_seconds() / 86400
        current_idx = int(days_from_start / self.dt)

        end_idx = current_idx + num_steps

        # Handle edge cases
        if end_idx > len(self.prediction_cache):
            # Pad with last available prediction if we go beyond cache
            available = self.prediction_cache[current_idx:]
            padding = np.full(num_steps - len(available), self.prediction_cache[-1])
            return np.concatenate([available, padding])

        return self.prediction_cache[current_idx:end_idx]

    def reset(self, start_date=None, seed=None, options=None):
        """
        Reset environment to initial state.
        Compatible with SB3/Gymnasium.
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.total_profit = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        # Use given start_date or default to test start
        if start_date is not None:
            self.start_date = start_date
        else:
            self.start_date = self.test_start_date

        self.current_datetime = self.start_date

        # Generate new price curve for this episode
        self.price_curve = self._generate_price_curve(self.max_steps)

        # Reset PCS storage
        self.pcs.reset(initial_storage_unit_level=1e6)

        return self._get_obs(), {}  # Gymnasium expects obs, info

    def step(self, action):
        """
        Executes one environment timestep (t-1 -> t).

        Args:
            action: battery action (float), positive=charge, negative=discharge
        Returns:
            obs, reward, done, truncated, info
        """

        battery_action = float(action[0])  # convert from action array

        # --- Step 1: Record storage before applying action ---
        storage_before = self.pcs.get_total_storage()

        # --- Step 2: Create state object with current timestep ---
        current_time = self.current_step * self.dt
        state = State()
        state.set_attribute('time', current_time)

        # --- Step 3: Prepare actions dict ---
        actions = {
            "Battery_0": Action({'value': battery_action})
        }

        # --- Step 4: Apply PCS update and get actual energy moved ---
        energy_sold_or_bought = self.pcs.update(state, actions)  # returns storage_after - storage_before

        storage_after = self.pcs.get_total_storage()  # can use for info

        # --- Step 5: Calculate total consumption for this timestep ---
        total_consumption = self.pcs.get_consumption()

        # Check if we can meet demand (consider battery storage)
        battery_entity = self.pcs.storage_units[0]
        available_discharge = battery_entity.get_available_discharge_capacity()

        # Shortage occurs if total consumption exceeds production + what battery can provide
        shortage = total_consumption > available_discharge

        # --- Step 6: Compute reward ---
        reward = 0.0

        # Fixed penalty for shortage
        if shortage:
            reward -= self.shortage_penalty
            self.shortage_count += 1
            self.total_shortage_penalty += self.shortage_penalty

        # Reward for energy moved by action (charging/discharging)
        # Use energy_sold_or_bought returned from update
        reward += self._get_current_price() * -energy_sold_or_bought

        # Track total profit if desired (optional)
        self.total_profit += reward

        # --- Step 7: Increment timestep ---
        self.current_step += 1
        self.current_datetime += timedelta(days=self.dt)

        # --- Step 8: Terminated flag ---
        terminated = self.current_step >= self.max_steps

        truncated = False  # can be True if you implement time limits separately

        # --- Step 9: Observation ---
        obs = self._get_obs()

        # --- Step 10: Info dictionary ---
        info = {
            'storage_before': storage_before,
            'storage_after': storage_after,
            'energy_sold_or_bought': energy_sold_or_bought,
            'shortage': shortage,
            'total_consumption': total_consumption,
            'battery_action': battery_action
        }

        # ✅ Return 5 values for Gymnasium
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Get current observation"""
        # Current state
        current_storage = self.pcs.get_total_storage() / 1e6  # Normalize to [0, 2]
        current_consumption = self.pcs.get_consumption() / 10000  # Normalize
        current_price = self._get_current_price() / self.base_price  # Normalize

        # Future consumption predictions (now from cache!)
        predicted_consumption = self._get_predicted_consumption(self.prediction_horizon)
        predicted_consumption = predicted_consumption / 10000  # Normalize

        # Combine into observation (no predicted prices!)
        obs = np.concatenate([
            [current_storage, current_consumption, current_price],
            predicted_consumption
        ])

        return obs.astype(np.float32)

    def render(self, mode='human'):
        """Render current state"""
        if mode == 'human':
            print(f"\n{'=' * 60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Date: {self.current_datetime.strftime('%Y-%m-%d %H:%M')}")
            print(f"Storage: {self.pcs.get_total_storage() / 1000:.2f} kWh")
            print(f"Consumption: {self.pcs.get_consumption():.2f} kW")
            print(f"Current Price: ${self._get_current_price():.3f}/kWh")
            print(f"Total Profit: ${self.total_profit:.2f}")
            print(f"Total Penalties: ${self.total_shortage_penalty:.2f}")
            print(f"Shortages: {self.shortage_count}")
            print(f"{'=' * 60}")


def main():
    # Initialize environment
    env = PCSGymEnv()

    current_date = env.test_start_date
    num_episodes = 300
    episode_rewards = []
    counter = 0

    print(f"\n{'=' * 60}")
    print("Starting simulation with pre-generated predictions...")
    print(f"{'=' * 60}\n")

    while counter < num_episodes:
        # Reset environment at the start of the current day
        counter += 1
        obs, info = env.reset(start_date=current_date)
        done = False
        total_reward = 0.0

        # Run episode for **one day only**
        while not done:
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Day {counter} ({current_date.strftime('%Y-%m-%d')}): Total Reward = {total_reward:.2f}")

        episode_rewards.append(total_reward)
        current_date += timedelta(days=1)  # move to next day for next episode

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Simulated {len(episode_rewards)} days")
    print(f"Average daily reward: {np.mean(episode_rewards):.2f}")
    print(f"Total reward: {np.sum(episode_rewards):.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()