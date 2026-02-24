# alternating_env.py (Enhanced with shortage & MAE tracking + avg ISO price tracking + dynamic features)
from datetime import timedelta

from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.grid_entities.management.price_curve import RLPriceCurveStrategy,SineWavePriceStrategy


from stable_baselines3 import PPO
import numpy as np
from energy_net.gym_envs.iso_env import ISOEnv


class AlternatingISOEnv(ISOEnv):
    """An ISO environment that internally coordinates a PCS environment.

    Behavior differences compared to a simple nested reset:
    - The ISO queries the PCS env for its global half-hour index and advances
      the ISO pointer forward to that index (prevents rewinding).
    - The ISO injects its price curve into the PCS and runs the PCS policy for
      T=48 steps to produce an economic reward which becomes the ISO reward.
    - TRACKS MAE and shortages separately from the money-based reward for comparison.
    - Automatically handles features from predictions CSV.
    """

    def __init__(self, actual_csv, predicted_csv, pcs_env, pcs_model,
                 steps_per_day=48, render_enabled: bool = False, render_every_n: int = 1,
                 iso_verbosity: int = 2, pcs_verbosity: int = 0):
        super().__init__(actual_csv, predicted_csv, steps_per_day)
        self.pcs_env = pcs_env
        self.pcs_model = pcs_model
        self.iso_model = None

        # Rendering control for ISO-driven PCS runs
        # Keep render_enabled for backward compatibility but map to verbosity
        if render_enabled:
            self.iso_verbosity = iso_verbosity if iso_verbosity != 2 else 2
            self.pcs_verbosity = pcs_verbosity if pcs_verbosity != 0 else 1
        else:
            self.iso_verbosity = iso_verbosity
            self.pcs_verbosity = pcs_verbosity
        
        self.render_enabled = bool(render_enabled)  # Keep for backward compatibility
        self.render_every_n = max(1, int(render_every_n))

    def sync_to_pcs(self, pcs_step_index: int):
        """Move the ISO's internal pointer to match the PCS half-hour index."""
        self._next_start_idx = int(pcs_step_index)

    def _get_iso_timestamp_from_pcs_index(self, pcs_index: int):
        return self.base_timestamp + timedelta(minutes=30 * int(pcs_index))

    def step(self, action):
        """
        ISO environment step.
        Uses RLPriceCurveStrategy to bridge the ISO brain to the PCS environment.
        """
        # 1) Anchor the index to the ISO's current position
        current_idx = self._next_start_idx
        current_iso_timestamp = self._get_iso_timestamp_from_pcs_index(current_idx)

        # 2) Get the observation window needed for the Strategy
        # This is the (384,) vector: 48 predictions + 336 features
        iso_obs = get_pred_window(self, current_idx, self.T)

        # 3) Setup the RL Price Strategy
        if self.iso_model is not None:
            strategy = RLPriceCurveStrategy(iso_model=self.iso_model)
            # Capture the ACTUAL scaled prices for the render/info dictionary
            actual_scaled_prices = strategy.calculate_price(iso_obs)
        else:
            strategy = SineWavePriceStrategy()
            actual_scaled_prices = strategy.calculate_price()

        # 4) Inject strategy into PCS and sync time
        self.pcs_env.set_price_strategy(strategy)
        self.pcs_env.set_step_index(current_idx)

        # Reset the PCS env for the day.
        obs, _ = self.pcs_env.reset(start_date=current_iso_timestamp)

        # Render PCS start based on verbosity
        if self.pcs_verbosity > 0:
            self.pcs_env.render(verbosity=self.pcs_verbosity)

        day_money = 0
        day_shortages = 0
        realized_consumption = []

        # 5) Run the PCS loop for 48 steps (one day)
        for step_i in range(self.T):
            pcs_action, _ = self.pcs_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.pcs_env.step(pcs_action)

            day_money += info.get('step_money', 0)
            realized_consumption.append(info.get('consumption_units', 0))
            if info.get('shortage', False):
                day_shortages += 1

            # Render PCS steps based on verbosity and render_every_n
            if self.pcs_verbosity >= 2 and (self.pcs_env.current_step % self.render_every_n == 0):
                self.pcs_env.render(verbosity=self.pcs_verbosity)

        # 6) Calculate metrics for ISO reward
        dispatch = action[self.T:]
        realized_array = np.array(realized_consumption, dtype=np.float32)
        mae = float(np.mean(np.abs(dispatch - realized_array)))

        # 7) Advance ISO state via parent class
        next_obs, _, _, _, iso_info = super().step(action)

        # 8) Enrich info dict with ACTUAL prices for the render function
        iso_info.update({
            "prices": actual_scaled_prices,  # Now render() shows real $ prices
            "money_earned": day_money,
            "mae": mae,
            "shortages": day_shortages,
            "realized_consumption": realized_array,
            "dispatch": dispatch
        })

        # Update render bookkeeping
        self._last_step_info.update(iso_info)
        self._last_reward = day_money

        return next_obs, day_money, True, False, iso_info
    def render(self, verbosity=None):
        """
        Render ISO environment with configurable verbosity.
        
        Args:
            verbosity: Override instance iso_verbosity. If None, uses self.iso_verbosity.
                Level 0: Silent - return data dict only
                Level 1: Summary - day metrics only (money, MAE, shortages)
                Level 2: Condensed - summary + sampled timesteps (every 4th) - DEFAULT
                Level 3: Detailed - summary + all timesteps
                Level 4: Debug - includes pricing strategy details and dispatch analysis
        
        Returns:
            dict: Structured data containing ISO day information
        """
        if self._last_reset_info is None or self._last_step_info is None:
            if verbosity is None:
                verbosity = self.iso_verbosity
            if verbosity > 0:
                print("Nothing to render yet. Call reset() and step() first for the day.")
            return None

        # Determine verbosity level
        v = verbosity if verbosity is not None else self.iso_verbosity
        
        info = self._last_reset_info
        step_info = self._last_step_info
        reward = self._last_reward
        
        # Extract metrics
        start_row = info['start_idx'] + 2
        end_row = info['start_idx'] + self.T + 1
        mae = step_info.get('mae', None)
        money = step_info.get('money_earned', None)
        shortages = step_info.get('shortages', None)
        
        # Build timestep rows for data return
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
            
            if self.num_features > 0:
                feature_start_idx = i * self.num_features
                feature_end_idx = feature_start_idx + self.num_features
                feature_values = self._current_features[feature_start_idx:feature_end_idx]
                for j, col in enumerate(self.feature_columns):
                    row_dict[col] = float(feature_values[j])
            
            rows.append(row_dict)
        
        # Build structured return data
        record = {
            "csv_range": (int(start_row), int(end_row)),
            "calendar_date": info['timestamp'],
            "rows": rows,
            "mae": float(mae) if mae is not None else None,
            "money_earned": float(money) if money is not None else None,
            "shortages": int(shortages) if shortages is not None else None,
            "reward": float(reward)
        }
        
        # Level 0: Silent - return data only
        if v == 0:
            return record
        
        # Level 1: Summary only
        if v == 1:
            print(f"\n[ISO Day] {info['timestamp'].strftime('%Y-%m-%d')} - Money: ${money:.2f}, MAE: {mae:.4f}, Shortages: {shortages}")
            return record
        
        # Level 2: Condensed (summary + sampled timesteps) - DEFAULT
        if v == 2:
            print(f"\n{'=' * 70}")
            print(f"[ISO DAY SUMMARY] {info['timestamp'].strftime('%Y-%m-%d')}")
            print(f"{'=' * 70}")
            print(f"Money Earned:      ${money:.2f}")
            print(f"MAE:               {mae:.4f}")
            print(f"Shortages:         {shortages}")
            print(f"{'=' * 70}")
            
            # Show sampled timesteps (every 4th)
            header = f"{'Time':<10} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}"
            print(header)
            print("-" * 55)
            
            for i in range(0, self.T, 4):
                row = rows[i]
                print(f"{row['slot_time'].strftime('%H:%M'):<10} | {row['pred']:<8.3f} | {row['actual']:<8.3f} | {row['dispatch']:<10.3f} | {row['price']:<6.2f}")
            
            print("-" * 55 + "\n")
            return record
        
        # Level 3: Detailed (all timesteps with features)
        if v == 3:
            print(f"\n{'=' * 70}")
            print(f"[ISO DAY DETAILED] {info['timestamp'].strftime('%Y-%m-%d')}")
            print(f"{'=' * 70}")
            print(f"CSV Row Range:     {start_row} to {end_row}")
            print(f"Money Earned:      ${money:.2f}")
            print(f"MAE:               {mae:.4f}")
            print(f"Shortages:         {shortages}")
            print(f"{'=' * 70}")
            
            # Build header with first 3 features
            header = f"{'Time':<10} | {'CSV Row':<8} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}"
            if self.num_features > 0:
                for col in self.feature_columns[:3]:
                    header += f" | {col:<10}"
                if self.num_features > 3:
                    header += " | ..."
            print(header)
            print("-" * (70 + min(self.num_features, 3) * 13 + (4 if self.num_features > 3 else 0)))
            
            # Show every 4th timestep with features
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
            print(f"Day Complete.\n")
            return record
        
        # Level 4: Debug (comprehensive analysis)
        if v >= 4:
            print(f"\n{'=' * 80}")
            print(f"[ISO DEBUG MODE] {info['timestamp'].strftime('%Y-%m-%d')}")
            print(f"{'=' * 80}")
            print(f"CSV Row Range:     {start_row} to {end_row}")
            print(f"Start Index:       {info['start_idx']}")
            
            print(f"\n--- Financial Metrics ---")
            print(f"Money Earned:      ${money:.6f}")
            print(f"Avg Price/Unit:    ${np.mean(step_info['prices']):.6f}")
            print(f"Price Range:       ${np.min(step_info['prices']):.4f} - ${np.max(step_info['prices']):.4f}")
            
            print(f"\n--- Dispatch Performance ---")
            print(f"MAE:               {mae:.6f}")
            print(f"RMSE:              {np.sqrt(np.mean((step_info['dispatch'] - step_info['realized'])**2)):.6f}")
            print(f"Max Error:         {np.max(np.abs(step_info['dispatch'] - step_info['realized'])):.6f}")
            
            print(f"\n--- Reliability ---")
            print(f"Shortages:         {shortages} / {self.T} ({100*shortages/self.T:.1f}%)")
            
            # Show pricing strategy
            print(f"\n--- Pricing Strategy ---")
            if self.iso_model is not None:
                print(f"Strategy:          RLPriceCurveStrategy (Active)")
            else:
                print(f"Strategy:          SineWavePriceStrategy (Fallback)")
            
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


class PenalizedAlternatingISOEnv(AlternatingISOEnv):
    """
    Child of AlternatingISOEnv that applies a small penalty for shortages to the ISO reward.
    Everything else behaves the same.
    """

    def __init__(self, *args, shortage_penalty: float = 100.0, **kwargs):
        """
        Pass all arguments to parent constructor.
        shortage_penalty: amount to subtract per shortage from the ISO reward
        """
        super().__init__(*args, **kwargs)
        self.shortage_penalty = shortage_penalty
        # This ensures that even the penalized version has the model placeholder
        # which is needed for RLPriceCurveStrategy to function.
        self.iso_model = None

    def step(self, action):
        # 1. Call the original step method (which now uses RLPriceCurveStrategy)
        expected, reward, done, truncated, info = super().step(action)

        # 2. Apply shortage penalty based on the data returned from the parent's PCS loop
        penalties = info.get("shortages", 0) * self.shortage_penalty
        net_reward = reward - penalties

        # 3. Update info and _last_reward for tracking and visualization
        info["shortage_penalty"] = penalties
        info["net_reward"] = net_reward
        self._last_reward = net_reward

        return expected, net_reward, done, truncated, info

class MultiObjectiveAlternatingISOEnv(AlternatingISOEnv):
    """
    ISO environment with multi-objective reward balancing money, shortages, and dispatch accuracy.
    
    This addresses the core coordination failure by giving the ISO incentives for:
    1. Economic efficiency (money earned by PCS)
    2. Reliability (minimizing shortages)
    3. Forecast accuracy (minimizing MAE between dispatch and actual consumption)
    
    The tunable weights allow experimentation with different objective priorities.
    """

    def __init__(self, *args, 
                 shortage_weight: float = 10.0,
                 mae_weight: float = 5.0,
                 money_weight: float = 1.0,
                 **kwargs):
        """
        Initialize multi-objective ISO environment.
        
        Args:
            shortage_weight: Penalty weight per shortage event (higher = prioritize reliability)
            mae_weight: Penalty weight for dispatch MAE (higher = prioritize forecast accuracy)
            money_weight: Reward weight for money earned (higher = prioritize economics)
            *args, **kwargs: Passed to parent AlternatingISOEnv
        """
        super().__init__(*args, **kwargs)
        self.shortage_weight = shortage_weight
        self.mae_weight = mae_weight
        self.money_weight = money_weight
        self.iso_model = None
        
        # Track historical metrics for analysis
        self.reward_history = {
            'money': [],
            'shortage_penalty': [],
            'mae_penalty': [],
            'net_reward': []
        }

    def step(self, action):
        # 1. Call parent step to get base metrics
        expected, reward, done, truncated, info = super().step(action)
        
        # 2. Extract components
        money = info.get("money_earned", 0)
        shortages = info.get("shortages", 0)
        mae = info.get("mae", 0)
        
        # 3. Compute weighted multi-objective reward
        # Maximize money, minimize shortages and MAE
        shortage_penalty = self.shortage_weight * shortages
        mae_penalty = self.mae_weight * mae
        net_reward = (self.money_weight * money) - shortage_penalty - mae_penalty
        
        # 4. Store detailed reward components for analysis
        info["reward_components"] = {
            "money": money,
            "money_contribution": self.money_weight * money,
            "shortage_penalty": shortage_penalty,
            "mae_penalty": mae_penalty,
            "net_reward": net_reward,
            "weights": {
                "money": self.money_weight,
                "shortage": self.shortage_weight,
                "mae": self.mae_weight
            }
        }
        
        # 5. Update tracking
        self._last_reward = net_reward
        self.reward_history['money'].append(money)
        self.reward_history['shortage_penalty'].append(shortage_penalty)
        self.reward_history['mae_penalty'].append(mae_penalty)
        self.reward_history['net_reward'].append(net_reward)
        
        return expected, net_reward, done, truncated, info

    def render(self, verbosity=None):
        """
        Extend parent render to show multi-objective reward breakdown.
        """
        record = super().render(verbosity=verbosity)
        
        v = verbosity if verbosity is not None else self.iso_verbosity
        
        # Add reward breakdown for verbosity > 0
        if v > 0 and record is not None and self._last_step_info is not None:
            components = self._last_step_info.get("reward_components", {})
            
            if components:
                if v == 1:
                    # Compact format
                    print(f"Reward: Money=${components['money']:.2f}, "
                          f"Shortage=-${components['shortage_penalty']:.2f}, "
                          f"MAE=-${components['mae_penalty']:.2f}, "
                          f"Net=${components['net_reward']:.2f}")
                else:
                    # Detailed format
                    print(f"\n--- Multi-Objective Reward Breakdown ---")
                    print(f"Money Earned (PCS):    ${components['money']:.2f} × {components['weights']['money']:.1f} = ${components['money_contribution']:.2f}")
                    print(f"Shortage Penalty:      -{components['shortage_penalty']:.2f}")
                    print(f"MAE Penalty:           -{components['mae_penalty']:.2f}")
                    print(f"Net ISO Reward:        ${components['net_reward']:.2f}")
                    print(f"{'=' * 45}\n")
            
            # Add components to return record
            if record:
                record["reward_components"] = components
        
        return record

    def get_reward_statistics(self):
        """
        Get summary statistics of reward components across all episodes.
        
        Returns:
            dict: Statistics for each reward component
        """
        if not self.reward_history['net_reward']:
            return None
            
        return {
            'mean_money': np.mean(self.reward_history['money']),
            'mean_shortage_penalty': np.mean(self.reward_history['shortage_penalty']),
            'mean_mae_penalty': np.mean(self.reward_history['mae_penalty']),
            'mean_net_reward': np.mean(self.reward_history['net_reward']),
            'total_episodes': len(self.reward_history['net_reward'])
        }

def get_pred_window(env: ISOEnv, start: int, length: int) -> np.ndarray:
    """
    Return a full observation window (predictions + features) starting at `start`.

    Args:
        env: ISO environment with feature support
        start: Starting index
        length: Number of timesteps (should be T=48)

    Returns:
        Full observation array (predictions + features flattened)
    """
    n = len(env.pred_data)
    if n == 0:
        raise ValueError("Predictions array is empty.")

    start = int(start) % n

    # Get predictions
    if start + length <= n:
        preds = env.pred_data[start:start + length]
    else:
        # wrap-around case
        first = env.pred_data[start:]
        remaining = length - len(first)
        second = env.pred_data[:remaining]
        preds = np.concatenate([first, second], axis=0)

    # Get features if available
    if env.num_features > 0:
        features = env._get_features_for_range(start, length)
        return np.concatenate([preds, features]).astype(np.float32)
    else:
        return preds.astype(np.float32)


# -----------------------
# Main training function
# -----------------------
def run_alternating_training(
        cycle_days: int = 7,
        total_iterations: int = 50,
        *,
        pcs_algo_cls=PPO,
        iso_algo_cls=PPO,
        pcs_policy: str = "MlpPolicy",
        iso_policy: str = "MlpPolicy",
        pcs_algo_kwargs: dict | None = None,
        iso_algo_kwargs: dict | None = None,
        pcs_steps_per_day: int | None = None,
        test_data_file: str = '../../tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
        predictions_file: str = '../../tests/gym/data_for_tests/consumption_predictions.csv',
        verbose: int = 0,
        render: bool = False,
        render_every_n_steps: int = 1,
):
    """
    Tandem ISO <-> PCS training loop with convergence tracking and feature support.
    Uses the RLPriceCurveStrategy to bridge the ISO agent's policy into the PCS environment.

    Args:
        cycle_days: Number of days per training cycle
        total_iterations: Total number of training iterations
        pcs_algo_cls: Algorithm class for PCS agent (default: PPO)
        iso_algo_cls: Algorithm class for ISO agent (default: PPO)
        pcs_policy: Policy type for PCS agent
        iso_policy: Policy type for ISO agent
        pcs_algo_kwargs: Additional kwargs for PCS algorithm
        iso_algo_kwargs: Additional kwargs for ISO algorithm
        pcs_steps_per_day: Steps per day for PCS training
        test_data_file: Path to test data CSV file
        predictions_file: Path to predictions CSV file
        verbose: Verbosity level
        render: Whether to render the environment
        render_every_n_steps: Render frequency
    """

    if cycle_days < 1:
        raise ValueError("cycle_days must be >= 1")
    if total_iterations < 1:
        raise ValueError("total_iterations must be >= 1")

    pcs_algo_kwargs = {} if pcs_algo_kwargs is None else dict(pcs_algo_kwargs)

    # Initialize the base environment for the household/battery agent (PCS)
    temp_env = PCSEnv(
        test_data_file=test_data_file,
        predictions_file=predictions_file,
        render_mode="human" if render else None
    )
    steps_per_day = int(pcs_steps_per_day or temp_env.max_steps)

    pcs_algo_kwargs.setdefault('n_steps', steps_per_day)
    pcs_algo_kwargs.setdefault('batch_size', steps_per_day)

    iso_algo_kwargs = {} if iso_algo_kwargs is None else dict(iso_algo_kwargs)
    iso_algo_kwargs.setdefault('n_steps', cycle_days)
    iso_algo_kwargs.setdefault('batch_size', cycle_days)

    print("\n--- INITIALIZING TANDEM TRAINING ---")

    base_pcs_env = temp_env

    # 1. Create the PCS (Household) Model
    pcs_model = pcs_algo_cls(pcs_policy, base_pcs_env, verbose=verbose, **pcs_algo_kwargs)

    # 2. Create the ISO (Grid) Environment
    # Note: AlternatingISOEnv needs the PCS model to simulate household responses
    iso_env = AlternatingISOEnv(
        actual_csv=test_data_file,
        predicted_csv=predictions_file,
        pcs_env=base_pcs_env,
        pcs_model=pcs_model,
        render_enabled=render,
        render_every_n=max(1, int(render_every_n_steps))
    )

    # 3. Create the ISO (Grid) Model
    iso_model = iso_algo_cls(iso_policy, iso_env, verbose=verbose, **iso_algo_kwargs)

    # 4. LINK MODELS: Crucial step to avoid shape mismatch errors
    # Inject the ISO model back into the env so it can use RLPriceCurveStrategy internally
    iso_env.iso_model = iso_model
    pricing_strategy = RLPriceCurveStrategy(iso_model=iso_model)

    # Metric Tracking Initialization
    history = {"iteration": [], "avg_money": [], "avg_mae": [], "total_shortages": [], "avg_iso_price": []}

    for iteration in range(1, total_iterations + 1):
        if render:
            print(f"\n[Iteration {iteration}] ISO Learning Phase ({cycle_days} Days)...")
        # ISO learns optimal pricing given current household behavior
        iso_model.learn(total_timesteps=cycle_days, reset_num_timesteps=False)

        if render:
            print(f"[Iteration {iteration}] PCS Learning Phase...")

        base_idx = iso_env._next_start_idx
        day_money_list = []
        iteration_maes = []
        iteration_shortages = 0
        day_avg_prices = []

        for day in range(cycle_days):
            start = base_idx + day * steps_per_day

            # Get the ISO's full observation (predictions + features)
            obs_window = get_pred_window(iso_env, start, steps_per_day)

            # Record average daily price using the strategy
            price_curve = pricing_strategy.calculate_price(obs_window)
            day_avg_prices.append(float(np.mean(price_curve)))

            # Inject the strategy into the PCS environment for the learning loop
            base_pcs_env.set_price_strategy(pricing_strategy)
            base_pcs_env.set_step_index(start)
            obs, _ = base_pcs_env.reset(start_date=iso_env._get_iso_timestamp_from_pcs_index(start))

            eval_day_money = 0.0
            day_actuals = []

            # Extract the ISO's target dispatch values (second half of action)
            iso_action, _ = iso_model.predict(obs_window, deterministic=True)
            day_dispatch = iso_action[steps_per_day:]

            for step_i in range(steps_per_day):
                # PCS Agent acts based on the current ISO-generated price
                pcs_action, _ = pcs_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = base_pcs_env.step(pcs_action)

                eval_day_money += info.get('step_money', 0.0)
                day_actuals.append(info.get('consumption_units', 0.0))
                if info.get('shortage', False):
                    iteration_shortages += 1

            day_money_list.append(eval_day_money)
            iteration_maes.append(np.mean(np.abs(np.array(day_actuals) - day_dispatch)))

            # PCS agent learns from the rewards earned today
            base_pcs_env.set_step_index(start)
            obs, _ = base_pcs_env.reset(start_date=iso_env._get_iso_timestamp_from_pcs_index(start))
            pcs_model.learn(total_timesteps=steps_per_day, reset_num_timesteps=False)

        # Advance global clock
        final_idx = base_idx + (cycle_days * steps_per_day)
        iso_env.sync_to_pcs(final_idx)

        # Update History
        avg_money = float(np.mean(day_money_list))
        avg_mae = float(np.mean(iteration_maes))
        avg_iso_price = float(np.mean(day_avg_prices)) if len(day_avg_prices) > 0 else 0.0

        history["iteration"].append(iteration)
        history["avg_money"].append(avg_money)
        history["avg_mae"].append(avg_mae)
        history["total_shortages"].append(iteration_shortages)
        history["avg_iso_price"].append(avg_iso_price)

        if render:
            print(f">>> Iteration {iteration} Avg Money: ${avg_money:.2f} | Avg MAE: {avg_mae:.4f} | Avg ISO Price: ${avg_iso_price:.4f}")

    print("\n--- TRAINING COMPLETE ---")
    return history
if __name__ == "__main__":
    try:
        run_alternating_training(render=True, render_every_n_steps=1)
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        import traceback

        traceback.print_exc()