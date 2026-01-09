# alternating_env.py (Enhanced with shortage & MAE tracking + avg ISO price tracking + dynamic features)
from datetime import timedelta

from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.grid_entities.management.price_curve import RLPriceCurveStrategy,SineWavePriceStrategy


from stable_baselines3 import PPO
import numpy as np
from iso_env import ISOEnv


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
                 steps_per_day=48, render_enabled: bool = False, render_every_n: int = 1):
        super().__init__(actual_csv, predicted_csv, steps_per_day)
        self.pcs_env = pcs_env
        self.pcs_model = pcs_model
        self.iso_model = None

        # Rendering control for ISO-driven PCS runs
        self.render_enabled = bool(render_enabled)
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

        if self.render_enabled:
            self.pcs_env.render()

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

            if self.render_enabled and (self.pcs_env.current_step % self.render_every_n == 0):
                self.pcs_env.render()

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
    def render(self):
        """Enhanced render that shows MAE, money earned, and shortages."""
        if self._last_reset_info is None or self._last_step_info is None:
            print("Nothing to render yet. Call reset() and step() first for the day.")
            return None

        info = self._last_reset_info
        step_info = self._last_step_info
        reward = self._last_reward

        # Print header for the day
        start_row = info['start_idx'] + 2
        end_row = info['start_idx'] + self.T + 1
        print(f"\n{'=' * 70}")
        print(f"--- [ISO DAY SUMMARY] ---")
        print(f"CSV Row Range: {start_row} to {end_row}")
        print(f"Calendar Date: {info['timestamp'].strftime('%Y-%m-%d')}")
        print(f"{'=' * 70}")

        # Show all metrics prominently
        mae = step_info.get('mae', 'N/A')
        money = step_info.get('money_earned', 'N/A')
        shortages = step_info.get('shortages', 'N/A')

        print(f"Money Earned (Primary Reward): ${money:.2f}" if isinstance(money,
                                                                           (int, float)) else f"Money Earned: {money}")
        print(f"MAE (Dispatch vs Realized):     {mae:.4f}" if isinstance(mae, (int, float)) else f"MAE: {mae}")
        print(
            f"Shortages This Day:             {shortages}" if isinstance(shortages, int) else f"Shortages: {shortages}")
        print(f"{'=' * 70}")

        # Build header dynamically with feature columns
        header = f"{'Time':<10} | {'CSV Row':<8} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}"
        if self.num_features > 0:
            for col in self.feature_columns[:3]:  # Show first 3 features to avoid clutter
                header += f" | {col:<10}"
            if self.num_features > 3:
                header += " | ..."
        print(header)
        print("-" * (70 + min(self.num_features, 3) * 13 + (4 if self.num_features > 3 else 0)))

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

            # Add feature values (show first 3)
            if self.num_features > 0:
                feature_start_idx = i * self.num_features
                feature_end_idx = feature_start_idx + self.num_features
                feature_values = self._current_features[feature_start_idx:feature_end_idx]

                for j, col in enumerate(self.feature_columns[:3]):
                    val = feature_values[j]
                    row_str += f" | {val:<10.4f}"
                    row_dict[col] = float(val)

                if self.num_features > 3:
                    row_str += " | ..."

            print(row_str)
            rows.append(row_dict)

        print("-" * (70 + min(self.num_features, 3) * 13 + (4 if self.num_features > 3 else 0)))
        print(f"Day Complete.\n")

        # return a structured record with all metrics
        record = {
            "csv_range": (int(start_row), int(end_row)),
            "calendar_date": info['timestamp'],
            "rows": rows,
            "mae": float(mae) if isinstance(mae, (int, float)) else None,
            "money_earned": float(money) if isinstance(money, (int, float)) else None,
            "shortages": int(shortages) if isinstance(shortages, int) else None,
            "reward": float(reward)
        }
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

    def render(self):
        """
        Extend render to include penalty, net reward, and the strategy name.
        """
        record = super().render()
        if record is not None and self._last_step_info is not None:
            # Safely retrieve values from the updated step info
            penalties = self._last_step_info.get("shortage_penalty", 0.0)
            net = self._last_step_info.get("net_reward", self._last_reward)

            # Add a clear indicator of which pricing strategy was active
            if self.iso_model is not None:
                print(f"Pricing Logic:                  RLPriceCurveStrategy (Active)")
            else:
                print(f"Pricing Logic:                  Fallback (SineWave)")

            print(f"Shortage Penalty Applied:       -${penalties:.2f}")
            print(f"Net Reward (ISO Profit):        ${net:.2f}\n")

        return record

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