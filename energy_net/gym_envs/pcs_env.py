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
from energy_net.gym_envs.pcs_rewards import (
    SHORTAGE_TOLERANCE,
    net_meter_reward,
    net_meter_reward_with_shortage_penalty,
)
from datetime import timedelta
import numpy as np
import os


class PCSEnv(gym.Env):
    """
    Simplified unit-less Gym environment for PCSUnit.
    - Uses internal "units" everywhere (battery capacity 0..100, consumption ~2 units/step).
    - Battery actions request energy changes in [-10, 10] (action_scale=1).
    - No unit conversions to kWh/kW — everything stays in the same internal units.
    - Loads pre-computed predictions from CSV file.
    - Queries actual CSV demand using the same calendar date as the predictions.
    - Settles net grid exchange; grid-reliance penalty is optional reward shaping.
    - Automatically detects and includes feature columns in observation space.
    """

    def __init__(self,
                 test_data_file=None,
                 predictions_file=None,
                 dt=0.5 / 24,  # 30 minutes in days
                 episode_length_days=1,
                 prediction_horizon=None,
                 shortage_penalty=1.0,
                 action_scale=1.0,
                 price_strategy: PriceCurveStrategy = None,
                 render_mode: str | None = None,
                 verbosity: int = 2,
                 include_shortage_penalty: bool = False):
        """
        Initialize PCS environment.
        
        Note: Predictions are encoded with zero-padding after episode end
        plus a validity mask (masked_adaptive encoding) for optimal performance.
        
        Args:
            prediction_horizon: Number of future timesteps to include in observation.
                If None, defaults to max_steps (full day visibility).
            include_shortage_penalty: If True, apply an additional fixed penalty
                whenever demand must be supplied by the grid. Defaults to the
                unpenalized POMDP reward; shortage_penalty controls its weight.
        """
        super().__init__()

        # Calculate project root for absolute paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Set default file paths if not provided
        if test_data_file is None:
            test_data_file = os.path.join(project_root, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
        if predictions_file is None:
            predictions_file = os.path.join(project_root, 'tests/gym/data_for_tests/consumption_predictions.csv')

        # -------------------------
        # Basic parameters
        # -------------------------
        self.dt = dt
        self.episode_length_days = episode_length_days
        self.max_steps = int(episode_length_days / dt)
        
        # Default prediction_horizon to full day if not specified
        if prediction_horizon is None:
            prediction_horizon = self.max_steps
        self.prediction_horizon = prediction_horizon
        self.shortage_penalty = float(shortage_penalty)
        if self.shortage_penalty < 0:
            raise ValueError("shortage_penalty must be nonnegative")
        self.include_shortage_penalty = include_shortage_penalty
        self.action_scale = action_scale
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

        if "Consumption" not in self.test_df.columns:
            raise ValueError(
                "Consumption CSV must have a Consumption column"
            )

        if (
            self.test_df.index.has_duplicates
            or not self.test_df.index.is_monotonic_increasing
        ):
            raise ValueError(
                "Consumption timestamps must be unique and sorted"
            )

        # Invalid numeric values are rejected by episode validation,
        # rather than silently corrected.
        self.test_df["Consumption"] = pd.to_numeric(
            self.test_df["Consumption"],
            errors="coerce",
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

        if "predicted_consumption" not in pred_df.columns:
            raise ValueError(
                "Forecast CSV must have a predicted_consumption column"
            )

        if pred_df.index.has_duplicates:
            raise ValueError("Forecast timestamps must be unique")

        pred_df = pred_df.apply(pd.to_numeric, errors="coerce")

        self.pred_df = pred_df
        self.prediction_cache = pred_df['predicted_consumption'].values
        self.prediction_timestamps = pred_df.index

        # Automatically detect feature columns (everything except timestamp and predicted_consumption)
        excluded_cols = {'predicted_consumption'}
        self.feature_columns = [col for col in pred_df.columns if col not in excluded_cols]
        self.num_features = len(self.feature_columns)

        if self.verbosity > 0:
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
        if self.verbosity > 0:
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

        # Observation space uses masked_adaptive encoding:
        # zero-padded predictions + validity mask + battery capacity constraints
        encoded_prediction_dim = prediction_horizon * 2
        
        # Observation space:
        # [storage, consumption, PCS buy tariff, PCS sell tariff]
        # + [zero-padded predictions + mask] + [battery capacities] + [features]
        obs_dim = 4 + encoded_prediction_dim + 2 + self.num_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        if self.verbosity > 0:
            print(f"Observation space dimension: {obs_dim}")
            print("  - Base features (storage, consumption, buy/sell tariffs): 4")
            print(f"  - Zero-padded predictions + validity mask: {encoded_prediction_dim}")
            print("  - Battery capacity constraints: 2")
            print(f"  - Additional time features: {self.num_features}")

        # ==============================================================
        # Episode state
        # ==============================================================
        self.current_step = 0
        self.start_date = self.test_start_date
        self.current_datetime = self.start_date
        # Demand realized in the interval preceding the current decision.
        # This is part of the POMDP observation state.
        self.previous_load = None
        self.total_reward = 0.0
        self.total_money_earned = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

    def get_money(self):
        """Returns the accumulated money made from transactions, ignoring shortage penalties."""
        return self.total_money_earned

    def _get_current_prices(self) -> tuple[float, float]:
        """Return PCS (buy/import, sell/export) tariffs announced before action.

        Pricing strategies name buy/sell from the ISO's viewpoint. Daily
        curves are cached for their calendar date and indexed by clock slot.
        """
        current_date = self.current_datetime.date()
        if self.cached_day_prices is None or current_date != self.last_price_update_date:
            iso_horizon = 48
            # ISO daily input is based on the calendar date, not the episode start.
            iso_input = None

            if self.price_strategy.requires_daily_observation:
                # Data-dependent ISO strategies require an entire calendar day.
                day_start = pd.Timestamp(current_date)

                predicted_consumption = self._get_predicted_consumption(
                    iso_horizon, start_datetime=day_start
                )
                day_features = self._get_feature_window(
                    iso_horizon, start_datetime=day_start
                )

                iso_input = np.concatenate([
                    predicted_consumption / 10.0,
                    day_features,
                ])
            pcs_buy, pcs_sell = self.price_strategy.calculate_pcs_tariffs(iso_input)
            self.cached_buy_prices = np.asarray(pcs_buy, dtype=float)
            self.cached_sell_prices = np.asarray(pcs_sell, dtype=float)
            if self.cached_buy_prices.shape != (48,) or self.cached_sell_prices.shape != (48,):
                raise ValueError("ISO pricing strategy must return two 48-slot daily curves")
            # Keep a legacy default price curve for callers inspecting this cache.
            self.cached_day_prices = self.cached_buy_prices
            self.last_price_update_date = current_date

        slot = self.current_datetime.hour * 2 + self.current_datetime.minute // 30
        return float(self.cached_buy_prices[slot]), float(self.cached_sell_prices[slot])

    def _get_current_price(self, energy_flow_direction=None) -> float:
        """Legacy single-price accessor; prefer _get_current_prices().

        The default returns the PCS buy/import tariff. The directional form
        preserves the old 'buy'/'sell' vocabulary for external callers.
        """
        buy_tariff, sell_tariff = self._get_current_prices()
        return sell_tariff if energy_flow_direction == "sell" else buy_tariff

    def set_price_strategy(self, strategy: PriceCurveStrategy):
        """
        Explicitly updates the pricing logic by injecting a new
        PriceCurveStrategy object.
        """
        if not isinstance(strategy, PriceCurveStrategy):
            raise TypeError("The provided strategy must be an instance of PriceCurveStrategy.")

        self.price_strategy = strategy
        self.cached_day_prices = None
        self.cached_buy_prices = None
        self.cached_sell_prices = None
        self.last_price_update_date = None


    def _prediction_rows(self, start_datetime, num_steps):
        """
        Return timestamp-aligned forecast rows.
    
        Never interpolate missing timestamps or numeric values.
        """
        timestamps = pd.date_range(
            start=pd.Timestamp(start_datetime),
            periods=num_steps,
            freq=pd.Timedelta(days=self.dt),
        )
    
        missing = timestamps.difference(self.pred_df.index)
    
        if len(missing):
            raise ValueError(
                f"Missing forecast timestamp: {missing[0]}"
            )
    
        rows = self.pred_df.loc[timestamps]
    
        if not np.isfinite(rows.to_numpy(dtype=float)).all():
            raise ValueError(
                "Non-finite forecast or calendar features "
                f"from {timestamps[0]}"
            )
    
        return rows
    
    
    def _get_predicted_consumption(
        self, num_steps, start_datetime=None
    ):
        start = (
            self.current_datetime
            if start_datetime is None
            else start_datetime
        )
    
        return self._prediction_rows(
            start, num_steps
        )["predicted_consumption"].to_numpy(dtype=float)
    
    
    def _get_current_features(self):
        if self.num_features == 0:
            return np.array([], dtype=float)
    
        return self._prediction_rows(
            self.current_datetime, 1
        )[self.feature_columns].iloc[0].to_numpy(dtype=float)



    def _episode_error(self, start):
        """
        Return None if the episode has valid required inputs.
        Otherwise, return a description of the invalid input.
        """
        start = pd.Timestamp(start)

        episode_times = pd.date_range(
            start=start,
            periods=self.max_steps,
            freq=pd.Timedelta(days=self.dt),
        )

        previous_time = start - pd.Timedelta(days=self.dt)

        # Previous demand + all realized demands in the episode.
        required_load_times = episode_times.insert(
            0, previous_time
        )

        missing_load = required_load_times.difference(
            self.test_df.index
        )

        if len(missing_load):
            return (
                "Missing preceding/episode consumption timestamp: "
                f"{missing_load[0]}"
            )

        loads = self.test_df.loc[
            required_load_times, "Consumption"
        ].to_numpy(dtype=float)

        if not np.isfinite(loads).all() or (loads < 0).any():
            return (
                "Non-finite or negative preceding/episode consumption"
            )

        # All 48 in-episode timestamps require forecasts.
        required_forecast_times = episode_times

        if self.price_strategy.requires_daily_observation:
            # Data-dependent ISO strategies require full daily
            # inputs for every calendar day touched by the episode.
            for day_start in episode_times.normalize().unique():
                day_times = pd.date_range(
                    start=day_start,
                    periods=48,
                    freq=pd.Timedelta(minutes=30),
                )

                required_forecast_times = (
                    required_forecast_times.union(day_times)
                )

        missing_predictions = required_forecast_times.difference(
            self.pred_df.index
        )

        if len(missing_predictions):
            return (
                "Missing forecast/calendar timestamp: "
                f"{missing_predictions[0]}"
            )

        pred_rows = self.pred_df.loc[required_forecast_times]

        if not np.isfinite(
            pred_rows.to_numpy(dtype=float)
        ).all():
            return "Non-finite forecast or calendar features"

        return None


    def reset(self, start_date=None, options=None, seed=None):
        """
        Reset the PCS to the pre-action state at c_0.

        The initial observation contains actual demand from c_0 - dt.
        Demand at c_0 is not realized until the first step().
        """

        explicit_date = start_date is not None
        candidate = pd.Timestamp(
            start_date if explicit_date else self.current_datetime
        )

        error = self._episode_error(candidate)

        if error is not None:
            if explicit_date:
                raise ValueError(
                    f"Cannot start episode at {candidate}: {error}"
                )

            # For automatic resets, select the first eligible
            # timestamp, skipping contaminated episodes.
            candidate = next(
                (
                    t for t in self.test_df.index
                    if self._episode_error(t) is None
                ),
                None,
            )

            if candidate is None:
                raise ValueError(
                    "No eligible episode with complete valid input data"
                )
        super().reset(seed=seed)

        # Reset episode counters.
        self.current_step = 0
        self.total_reward = 0.0
        self.total_money_earned = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0
        self.current_datetime = candidate

        # Reset physical units. No demand is processed at reset.
        self.pcs.reset(initial_storage_unit_level=0)
        self.last_action = None

        # Initialize previous demand using the actual preceding row.
        previous_time = candidate - pd.Timedelta(days=self.dt)
        self.previous_load = float(
            self.test_df.at[previous_time, "Consumption"]
        )

        # _get_obs() includes the announced current tariffs.
        # The first PCS update happens only inside step().
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes one 30-minute time step in the environment.

        Args:
            action: A NumPy array containing the normalized battery intent [-10, 10].

        Returns:
            obs: The next observation vector.
            reward: Negative net electricity bill, optionally minus grid-reliance penalty.
            terminated: Whether the 24-hour day is complete.
            truncated: False (standard Gym API).
            info: A dictionary containing diagnostic information for the step.
        """
        if self.current_step >= self.max_steps:
            raise RuntimeError(
                "Episode has terminated; call reset() before step()"
            )

        # 1. Process the Agent's Action
        raw_action = float(action[0]) * self.action_scale
        self.last_action = raw_action

        # Record state before update for the info dict
        storage_before = self.pcs.get_total_storage()
        buy_tariff, sell_tariff = self._get_current_prices()
        battery_entity = self.pcs.storage_units[0]

        # 2. Update the Physical Simulation
        # PCS time is measured in days from the start of the consumption dataset.
        current_time_relative = (self.current_datetime - self.test_start_date).total_seconds() / 86400
        state = State()
        state.set_attribute('time', current_time_relative)

        # Apply action to the Battery via the PCS wrapper
        actions = {"Battery_0": Action({'value': raw_action})}
        energy_sold_or_bought = self.pcs.update(state, actions)

        # Record state after update
        storage_after = self.pcs.get_total_storage()
        consumption_units = float(self.pcs.get_consumption())

        try:
            available_discharge_units = float(battery_entity.get_available_discharge_capacity())
        except Exception:
            available_discharge_units = getattr(battery_entity, 'available_discharge', 0.0)

        # 3. Physical energy balance: q is the realized explicit action.
        # Automatic battery service supplies some demand; grid covers the rest.
        battery_energy_used_for_consumption = max(
            storage_before + energy_sold_or_bought - storage_after, 0.0
        )
        uncovered_consumption_units = max(
            consumption_units - float(self.pcs.get_production())
            - battery_energy_used_for_consumption,
            0.0,
        )
        net_grid_exchange = energy_sold_or_bought + uncovered_consumption_units
        shortage = uncovered_consumption_units > SHORTAGE_TOLERANCE

        # 4. The money component is always the POMDP net-metered bill.
        # Reward shaping is an experimental choice, not different settlement.
        step_money = net_meter_reward(net_grid_exchange, buy_tariff, sell_tariff)
        if self.include_shortage_penalty:
            reward = net_meter_reward_with_shortage_penalty(
                net_grid_exchange, buy_tariff, sell_tariff,
                uncovered_consumption_units, self.shortage_penalty,
            )
        else:
            reward = step_money
        step_penalty = step_money - reward
        if shortage:
            self.shortage_count += 1
        self.total_shortage_penalty += step_penalty
        self.total_money_earned += step_money
        self.total_reward += reward
        current_price = buy_tariff if net_grid_exchange >= 0 else sell_tariff

        # 6. Advance information and environment time.
        # The demand just realized becomes the previous demand
        # available for the next decision.
        self.previous_load = consumption_units
        self.current_step += 1
        self.current_datetime += timedelta(days=self.dt)

        # 7. Check for Episode Termination (typically 48 steps / 1 day)
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 8. Prepare Return Values
        # No further decision or exogenous observation after termination.
        obs = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._get_obs()
        )

        info = {
            'storage_before_units': storage_before,
            'storage_after_units': storage_after,
            'energy_sold_or_bought_units': energy_sold_or_bought,
            'consumption_units': consumption_units,
            'battery_energy_used_for_consumption_units': battery_energy_used_for_consumption,
            'uncovered_consumption_units': uncovered_consumption_units,
            'available_discharge_units': available_discharge_units,
            'shortage': shortage,
            'battery_action': raw_action,
            'step_money': step_money,
            'grid_exchange_units': net_grid_exchange,
            'buy_tariff': buy_tariff,
            'sell_tariff': sell_tariff,
            'shortage_penalty': step_penalty,
            'include_shortage_penalty': self.include_shortage_penalty,
            'current_price': current_price,  # tariff actually used for net exchange
            'total_money_so_far': self.total_money_earned
        }

        return obs, reward, terminated, truncated, info

    def _get_feature_window(
        self, num_steps, start_datetime=None
    ):
        """Return a flattened timestamp-aligned feature window."""
        if self.num_features == 0:
            return np.array([], dtype=float)

        start = (
            self.current_datetime
            if start_datetime is None
            else start_datetime
        )

        return self._prediction_rows(
            start, num_steps
        )[self.feature_columns].to_numpy(dtype=float).ravel()
    
    def _get_obs(self):
        """
        Build observation vector using masked_adaptive encoding:
        - Zero-pad predictions after episode end
        - Include validity mask showing which predictions are valid
        - Include battery capacity constraints (available charge/discharge)
        """
        # Base PCS observations
        current_storage = self.pcs.get_total_storage() / 100.0
        previous_consumption = self.previous_load / 10.0
        buy_tariff, sell_tariff = self._get_current_prices()

        # Get raw predictions and normalize
        # Request only forecasts belonging to this episode.
        steps_remaining = self.max_steps - self.current_step
        num_valid = min(self.prediction_horizon, steps_remaining)

        zero_padded = np.zeros(
            self.prediction_horizon,
            dtype=float,
        )
        mask = np.zeros(
            self.prediction_horizon,
            dtype=float,
        )

        if num_valid > 0:
            zero_padded[:num_valid] = (
                self._get_predicted_consumption(num_valid) / 10.0
            )
            mask[:num_valid] = 1.0
        
        # Get battery capacity constraints
        battery_entity = self.pcs.storage_units[0] if len(self.pcs.storage_units) > 0 else None
        
        if battery_entity:
            try:
                # Get available capacities (normalized by same scale as actions)
                available_charge = float(battery_entity.get_available_charge_capacity()) / 10.0
                available_discharge = float(battery_entity.get_available_discharge_capacity()) / 10.0
            except Exception:
                # Fallback if methods don't exist
                available_charge = (100.0 - current_storage * 100.0) / 10.0  # space left
                available_discharge = (current_storage * 100.0) / 10.0  # current storage
        else:
            available_charge = 0.0
            available_discharge = 0.0
        
        # Get current time features
        current_features = self._get_current_features()

        # Concatenate in documented order: [storage, consumption, buy, sell, forecasts, mask, capacities, calendar].
        obs = np.concatenate([
            [current_storage, previous_consumption, buy_tariff, sell_tariff],
            zero_padded,
            mask,
            [available_charge, available_discharge],
            current_features
        ])

        return obs.astype(np.float32)

    def render(self, action=None, verbosity=None):
        """
        Renders the current state of the environment with configurable verbosity.
        
        Args:
            action: Optional action array that will be applied. If provided, displays the action
                   that is about to be applied. If None, shows self.last_action (for backward compatibility).
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
        
        # Determine which action to display
        if action is not None:
            display_action = float(action[0]) * self.action_scale
        else:
            display_action = self.last_action
        
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
            if display_action is not None:
                print(f"Battery Action:        {display_action:.2f} units")
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

            # Render BEFORE stepping to show current state + action that will be applied
            if env.current_step % render_every_n_steps == 0:
                env.render(action=action)

            # Step the simulation.
            # The env will internally query the PriceCurveStrategy for the current price,
            # calculate the financial reward, and update the battery/consumption state.
            obs, reward, terminated, truncated, info = env.step(action)

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
