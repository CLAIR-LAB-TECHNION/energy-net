import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from energy_net.grid_entities.consumption.consumption_dynamics import CSV_DataConsumptionDynamics
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.management.price_curve import PriceCurveStrategy, SineWavePriceStrategy
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
                 shortage_penalty=1.0,
                 price_strategy: PriceCurveStrategy = None,
                 render_mode: str | None = None,
                 verbosity: int = 2):

        super().__init__()

        # -------------------------
        # Basic parameters
        # -------------------------
        self.dt = dt
        self.episode_length_days = episode_length_days
        self.max_steps = int(episode_length_days / dt)
        self.prediction_horizon = prediction_horizon
        self.shortage_penalty = shortage_penalty
        if price_strategy is None:
            self.price_strategy = SineWavePriceStrategy()
        else:
            self.price_strategy = price_strategy
        
        # Handle backward compatibility: render_mode='human' maps to verbosity=2
        if render_mode == 'human':
            self.verbosity = 2
        elif render_mode is None:
            self.verbosity = verbosity
        else:
            self.verbosity = verbosity
        
        self.render_mode = render_mode  # Keep for backward compatibility
        self.last_action = None
        self.cached_day_prices = None
        self.last_price_update_date = None

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
            config=battery_config
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
            config=consumption_unit_config
        )

        self.pcs = PCSUnit(
            storage_units=[battery],
            consumption_units=[consumption_unit]
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

    def get_money(self):
        """Returns the accumulated money made from transactions, ignoring shortage penalties."""
        return self.total_money_earned

    def _get_current_price(self) -> float:
        # Only calculate the full day curve if the date has changed
        if self.cached_day_prices is None or self.current_datetime.date() != self.last_price_update_date:
            predicted_consumption = self._get_predicted_consumption(self.prediction_horizon)
            day_features = self._get_feature_window(self.prediction_horizon)  # The 336 values
            iso_input = np.concatenate([predicted_consumption / 10.0, day_features])

            # Call the ISO brain ONLY ONCE per day
            self.cached_day_prices = self.price_strategy.calculate_price(iso_input)
            self.last_price_update_date = self.current_datetime.date()

        intra_day_step = self.current_step % self.max_steps
        return float(self.cached_day_prices[intra_day_step])
    def set_price_strategy(self, strategy: PriceCurveStrategy):
        """
        Explicitly updates the pricing logic by injecting a new
        PriceCurveStrategy object.
        """
        if not isinstance(strategy, PriceCurveStrategy):
            raise TypeError("The provided strategy must be an instance of PriceCurveStrategy.")

        self.price_strategy = strategy

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
            return self.feature_cache[-1]

        return self.feature_cache[current_idx]

    def reset(self, start_date=None, options=None, seed=None):
        """
        Resets the environment to an initial state for a new episode.

        This method handles time synchronization, internal PCS unit resets,
        and resets all financial/performance tracking metrics.
        """
        # 1. Standard Gym reset for seeding
        super().reset(seed=seed)

        # 2. Reset episode-specific counters
        self.current_step = 0
        self.total_reward = 0.0
        self.total_money_earned = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        # 3. Handle Time Synchronization
        # If a specific date is provided (e.g., during tandem training), use it.
        # Otherwise, check if we've hit the end of the test data and loop if necessary.
        if start_date is not None:
            self.current_datetime = start_date
        else:
            if self.current_datetime >= self.test_end_date:
                print("Reached end of dataset. Looping back to start.")
                self.current_datetime = self.test_start_date

        # 4. Reset Physical Units
        # Resets the battery to empty (or a configured initial level) and clears PCS logs.
        self.pcs.reset(initial_storage_unit_level=0)
        self.last_action = None

        # 5. Return Initial Observation
        # Note: _get_obs() will internally query the price_strategy for the starting price.
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes one 30-minute time step in the environment.

        Args:
            action: A NumPy array containing the normalized battery intent [-10, 10].

        Returns:
            obs: The next observation vector.
            reward: The net financial reward (Money earned - Shortage penalties).
            terminated: Whether the 24-hour day is complete.
            truncated: False (standard Gym API).
            info: A dictionary containing diagnostic information for the step.
        """
        # 1. Process the Agent's Action
        raw_action = float(action[0])
        self.last_action = raw_action

        # Record state before update for the info dict
        storage_before = self.pcs.get_total_storage()
        battery_entity = self.pcs.storage_units[0]

        # 2. Update the Physical Simulation
        # Convert step count to a relative time for the PCS logic
        current_time_relative = self.current_step * self.dt
        state = State()
        state.set_attribute('time', current_time_relative)

        # Apply action to the Battery via the PCS wrapper
        actions = {"Battery_0": Action({'value': raw_action})}
        energy_sold_or_bought = self.pcs.update(state, actions)

        # Record state after update
        storage_after = self.pcs.get_total_storage()
        consumption_units = float(self.pcs.get_consumption())

        # Determine if a shortage occurred (demand exceeded available discharge)
        try:
            available_discharge_units = float(battery_entity.get_available_discharge_capacity())
        except Exception:
            available_discharge_units = getattr(battery_entity, 'available_discharge', 0.0)

        # 3. Calculate Financials using the Price Strategy
        # Fetch the price for this specific slot from the strategy class
        current_price = self._get_current_price()

        # Money earned: (Price * Energy Sold). If energy is bought, this is negative.
        step_money = current_price * -energy_sold_or_bought

        # 4. Handle Penalties
        shortage = consumption_units > available_discharge_units
        step_penalty = 0.0
        if shortage:
            step_penalty = self.shortage_penalty
            self.shortage_count += 1
            self.total_shortage_penalty += step_penalty

        # 5. Update Cumulative Metrics
        self.total_money_earned += step_money
        reward = step_money - step_penalty
        self.total_reward += reward

        # 6. Advance Environment Time
        self.current_step += 1
        self.current_datetime += timedelta(days=self.dt)

        # 7. Check for Episode Termination (typically 48 steps / 1 day)
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 8. Prepare Return Values
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
            'current_price': current_price,  # Added for better traceability
            'total_money_so_far': self.total_money_earned
        }

        return obs, reward, terminated, truncated, info

    def _get_feature_window(self, num_steps):
        """Get feature values for a range of timesteps starting from current datetime."""
        if self.feature_cache is None or self.num_features == 0:
            return np.array([])

        # Calculate starting index in the cache
        days_from_start = (self.current_datetime - self.test_start_date).total_seconds() / 86400
        current_idx = int(days_from_start / self.dt)
        end_idx = current_idx + num_steps

        if end_idx > len(self.feature_cache):
            # Handle end-of-dataset by padding with the last known features
            available = self.feature_cache[current_idx:]
            padding_needed = num_steps - len(available)
            padding = np.tile(self.feature_cache[-1], (padding_needed, 1))
            features = np.vstack([available, padding])
        else:
            features = self.feature_cache[current_idx:end_idx]

        # Flatten from (48, 7) to (336,) to match ISO requirements
        return features.flatten()

    def _get_obs(self):
        # Base PCS observations
        current_storage = self.pcs.get_total_storage() / 100.0
        current_consumption = self.pcs.get_consumption() / 10.0
        current_price = self._get_current_price()  # This now correctly uses the 384-length window

        # Standard PCS data (Predictions + current single-step features)
        predicted_consumption = self._get_predicted_consumption(self.prediction_horizon)  # 48
        current_features = self._get_current_features()  # 7

        # Concatenate into the PCS Agent's observation (Length: 3 + 48 + 7 = 58)
        obs = np.concatenate([
            [current_storage, current_consumption, current_price],
            predicted_consumption / 10.0,
            current_features
        ])

        return obs.astype(np.float32)

    def render(self, verbosity=None):
        """
        Renders the current state of the environment with configurable verbosity.
        
        Args:
            verbosity: Override instance verbosity level. If None, uses self.verbosity.
                Level 0: Silent - return data dict only, no output
                Level 1: Summary - episode totals only
                Level 2: Condensed - current step with key metrics (default)
                Level 3: Detailed - all state information for current step
                Level 4: Debug - includes battery dynamics and internal state
        
        Returns:
            dict: Structured data containing current state information
        """
        # Determine which verbosity level to use
        v = verbosity if verbosity is not None else self.verbosity
        
        # Gather current state data
        current_price = self._get_current_price()
        strategy_name = self.price_strategy.__class__.__name__
        current_storage = self.pcs.get_total_storage()
        current_consumption = self.pcs.get_consumption()
        current_features = self._get_current_features()
        
        # Get battery details for higher verbosity levels
        battery_entity = self.pcs.storage_units[0] if len(self.pcs.storage_units) > 0 else None
        
        # Build structured data dict (returned at all verbosity levels)
        data = {
            'step': self.current_step,
            'max_steps': self.max_steps,
            'datetime': self.current_datetime,
            'storage': current_storage,
            'consumption': current_consumption,
            'price': current_price,
            'strategy': strategy_name,
            'last_action': self.last_action,
            'total_money': self.total_money_earned,
            'total_penalty': self.total_shortage_penalty,
            'total_reward': self.total_reward,
            'shortage_count': self.shortage_count,
        }
        
        # Add features if available
        if self.num_features > 0:
            data['features'] = {col: float(current_features[i]) for i, col in enumerate(self.feature_columns)}
        
        # Level 0: Silent - return data only
        if v == 0:
            return data
        
        # Level 1: Summary only (episode totals)
        if v == 1:
            if self.current_step == 0:  # Only print at episode start
                print(f"\n[PCS Episode Start] Date: {self.current_datetime.strftime('%Y-%m-%d')}")
            elif self.current_step >= self.max_steps - 1:  # Print at episode end
                print(f"\n{'=' * 60}")
                print(f"[PCS Episode Summary] {self.current_datetime.strftime('%Y-%m-%d')}")
                print(f"{'=' * 60}")
                print(f"Total Steps:           {self.current_step + 1}")
                print(f"Money Earned:          ${self.total_money_earned:.2f}")
                print(f"Shortage Penalties:    ${self.total_shortage_penalty:.2f}")
                print(f"Net Reward:            ${self.total_reward:.2f}")
                print(f"Shortage Count:        {self.shortage_count}")
                print(f"Final Storage:         {current_storage:.2f} units")
                print(f"Strategy Used:         {strategy_name}")
                print(f"{'=' * 60}\n")
            return data
        
        # Level 2: Condensed (current step with key metrics) - DEFAULT
        if v == 2:
            print(f"\n{'=' * 60}")
            print(f"[PCS Step {self.current_step}/{self.max_steps}] {self.current_datetime.strftime('%Y-%m-%d %H:%M')}")
            print(f"{'=' * 60}")
            print(f"Storage:               {current_storage:.2f} / 100.0 units")
            print(f"Consumption:           {current_consumption:.2f} units/step")
            print(f"Price ({strategy_name}): ${current_price:.4f}/unit")
            if self.last_action is not None:
                print(f"Battery Action:        {self.last_action:.2f} units")
            print(f"--- Episode Running Totals ---")
            print(f"Money:     ${self.total_money_earned:.2f}")
            print(f"Penalties: ${self.total_shortage_penalty:.2f}")
            print(f"Net:       ${self.total_reward:.2f}")
            print(f"Shortages: {self.shortage_count}")
            print(f"{'=' * 60}")
            return data
        
        # Level 3: Detailed (all state information)
        if v == 3:
            print(f"\n{'=' * 60}")
            print(f"[PCS DETAILED] Step {self.current_step}/{self.max_steps}")
            print(f"{'=' * 60}")
            print(f"Date/Time:             {self.current_datetime.strftime('%Y-%m-%d %H:%M')}")
            print(f"Storage Level:         {current_storage:.2f} / 100.0 units ({current_storage:.1f}%)")
            print(f"Consumption (Actual):  {current_consumption:.2f} units/step")
            print(f"Price ({strategy_name}): ${current_price:.4f}/unit")
            
            if self.last_action is not None:
                action_type = "Charge" if self.last_action < 0 else "Discharge" if self.last_action > 0 else "Idle"
                print(f"Battery Action:        {self.last_action:.2f} units ({action_type})")
            
            # Show features if available
            if self.num_features > 0:
                print(f"\n--- Environmental Features ---")
                for i, col in enumerate(self.feature_columns):
                    print(f"  {col:<20}: {current_features[i]:.4f}")
            
            print(f"\n--- Episode Cumulative Metrics ---")
            print(f"Gross Money Earned:    ${self.total_money_earned:.2f}")
            print(f"Shortage Penalties:    ${self.total_shortage_penalty:.2f}")
            print(f"Net Reward:            ${self.total_reward:.2f}")
            print(f"Shortage Occurrences:  {self.shortage_count}")
            print(f"{'=' * 60}")
            return data
        
        # Level 4: Debug (includes battery dynamics and internal state)
        if v >= 4:
            print(f"\n{'=' * 80}")
            print(f"[PCS DEBUG MODE] Step {self.current_step}/{self.max_steps}")
            print(f"{'=' * 80}")
            print(f"Date/Time:             {self.current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Relative Time:         {self.current_step * self.dt:.4f} days")
            print(f"\n--- Storage System ---")
            print(f"Storage Level:         {current_storage:.4f} / 100.0 units ({current_storage:.2f}%)")
            
            if battery_entity:
                try:
                    avail_charge = float(battery_entity.get_available_charge_capacity())
                    avail_discharge = float(battery_entity.get_available_discharge_capacity())
                    data['available_charge'] = avail_charge
                    data['available_discharge'] = avail_discharge
                    print(f"Available Charge Cap:  {avail_charge:.4f} units")
                    print(f"Available Discharge:   {avail_discharge:.4f} units")
                except Exception as e:
                    print(f"Battery capacity info unavailable: {e}")
            
            print(f"\n--- Consumption & Pricing ---")
            print(f"Consumption (Actual):  {current_consumption:.4f} units/step")
            print(f"Pricing Strategy:      {strategy_name}")
            print(f"Current Price:         ${current_price:.6f}/unit")
            
            if self.last_action is not None:
                action_type = "CHARGE" if self.last_action < 0 else "DISCHARGE" if self.last_action > 0 else "IDLE"
                print(f"\n--- Agent Action ---")
                print(f"Raw Action Value:      {self.last_action:.4f} units")
                print(f"Action Type:           {action_type}")
                print(f"Action Bounds:         [-10.0, 10.0]")
            
            # Show all features in debug mode
            if self.num_features > 0:
                print(f"\n--- Environmental Features (All {self.num_features}) ---")
                for i, col in enumerate(self.feature_columns):
                    print(f"  {col:<25}: {current_features[i]:.6f}")
            
            # Show prediction horizon
            try:
                predictions = self._get_predicted_consumption(min(5, self.prediction_horizon))
                print(f"\n--- Consumption Predictions (Next 5 steps) ---")
                for i, pred in enumerate(predictions):
                    print(f"  Step +{i+1}: {pred:.4f} units")
            except Exception as e:
                print(f"\nPrediction data unavailable: {e}")
            
            print(f"\n--- Financial Metrics (Cumulative) ---")
            print(f"Gross Money Earned:    ${self.total_money_earned:.6f}")
            print(f"Shortage Penalties:    ${self.total_shortage_penalty:.6f}")
            print(f"Net Reward (Total):    ${self.total_reward:.6f}")
            print(f"Shortage Count:        {self.shortage_count}")
            if self.current_step > 0:
                avg_reward = self.total_reward / (self.current_step + 1)
                print(f"Average Reward/Step:   ${avg_reward:.6f}")
            
            print(f"{'=' * 80}")
            return data
        
        return data
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

    def _get_feature_window(self, num_steps):
        """Get feature values for a range of timesteps starting from current."""
        if self.feature_cache is None or self.num_features == 0:
            return np.array([])

        days_from_start = (self.current_datetime - self.test_start_date).total_seconds() / 86400
        current_idx = int(days_from_start / self.dt)
        end_idx = current_idx + num_steps

        if end_idx > len(self.feature_cache):
            available = self.feature_cache[current_idx:]
            padding_needed = num_steps - len(available)
            padding = np.tile(self.feature_cache[-1], (padding_needed, 1))
            features = np.vstack([available, padding])
        else:
            features = self.feature_cache[current_idx:end_idx]

        return features.flatten()  # Flattens to (T * num_features)

if __name__ == "__main__":
    # 1. Initialize the Environment
    # By default, PCSEnv will now use the SineWavePriceStrategy if no strategy is passed.
    # Using verbosity=2 (condensed) to show key metrics per step.
    env = PCSEnv(verbosity=2)

    # 2. Simulation Configuration
    num_days_to_run = 3
    accumulated_reward = 0.0
    render_every_n_steps = 1  # How often to call env.render() during the day

    # 3. Main Simulation Loop
    for day in range(num_days_to_run):
        # Reset at the start of each day.
        # This clears financial metrics and ensures the strategy starts fresh for the date.
        obs, info = env.reset()
        env.render()

        print(f"\n{'=' * 60}")
        print(f"STARTING DAY {day + 1}  |  Date: {env.current_datetime.strftime('%Y-%m-%d')}")
        print(f"{'=' * 60}")

        day_reward = 0.0

        # 4. Intra-day Time Steps (Typically 48 steps per day)
        while True:
            # Generate a random battery action (Intent) between -10 and 10.
            # -10: Maximum Charge, 10: Maximum Discharge.
            action_value = float(np.random.uniform(-10.0, 10.0))
            action = np.array([action_value], dtype=np.float32)

            # Step the simulation.
            # The env will internally query the PriceCurveStrategy for the current price,
            # calculate the financial reward, and update the battery/consumption state.
            obs, reward, terminated, truncated, info = env.step(action)

            # Render the console UI for this specific time slot.
            if env.current_step % render_every_n_steps == 0:
                env.render()

            # Accumulate rewards (Net financial performance).
            day_reward += reward
            accumulated_reward += reward

            print(f"Step Reward: {reward:.2f}")

            # 5. Check for End of Day
            # Terminated becomes True once the current_step reaches max_steps (48).
            if terminated or truncated:
                print(f"\n>>> Day {day + 1} Finished!")
                print(f">>> Total Day Reward: {day_reward:.2f}")
                print(f">>> Final Storage: {info['storage_after_units']:.2f} units")
                print(f">>> Total Shortages Today: {env.shortage_count}")
                break

    # 6. Final Summary
    print(f"\n{'!' * 60}")
    print(f"Simulation Complete. Total Reward over {num_days_to_run} days: {accumulated_reward:.2f}")
    print(f"{'!' * 60}")
