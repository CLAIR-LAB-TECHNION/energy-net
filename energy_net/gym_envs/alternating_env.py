# alternating_env.py (Enhanced with shortage & MAE tracking + avg ISO price tracking)
from datetime import timedelta
from typing import Sequence

from energy_net.gym_envs.pcs_env import PCSEnv
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
from iso_env import ISOEnv


class ISOPricingWrapper:
    """Direct bridge between an ISO RL model and the PCS environment.
    Converts a 48-step predicted consumption window into a realistic
    price curve (in $/unit) using the ISO policy's first 48 action values.
    """

    def __init__(self, iso_model, base_price=0.10, price_scale=0.20):
        self.iso_model = iso_model
        self.base_price = base_price
        self.price_scale = price_scale

    def generate_price_curve(self, predicted_consumption: np.ndarray) -> np.ndarray:
        # ISO expects a float32 (48,) forecast observation
        obs = predicted_consumption.astype(np.float32)

        # Query the ISO policy deterministically
        action, _ = self.iso_model.predict(obs, deterministic=True)

        # First 48 entries represent an unscaled price signal
        raw_prices = action[:48]

        # Robust min-max normalization
        min_p, max_p = raw_prices.min(), raw_prices.max()
        denom = (max_p - min_p) + 1e-8
        normalized = (raw_prices - min_p) / denom

        # Center and scale into a realistic price range
        scaled_prices = (self.base_price - self.price_scale / 2) + (normalized * self.price_scale)
        return scaled_prices.astype(np.float32)


class AlternatingISOEnv(ISOEnv):
    """An ISO environment that internally coordinates a PCS environment.

    Behavior differences compared to a simple nested reset:
    - The ISO queries the PCS env for its global half-hour index and advances
      the ISO pointer forward to that index (prevents rewinding).
    - The ISO injects its price curve into the PCS and runs the PCS policy for
      T=48 steps to produce an economic reward which becomes the ISO reward.
    - TRACKS MAE and shortages separately from the money-based reward for comparison.
    """

    def __init__(self, actual_csv, predicted_csv, pcs_env, pcs_model,
                 steps_per_day=48, render_enabled: bool = False, render_every_n: int = 1):
        super().__init__(actual_csv, predicted_csv, steps_per_day)
        self.pcs_env = pcs_env
        self.pcs_model = pcs_model

        # Rendering control for ISO-driven PCS runs
        self.render_enabled = bool(render_enabled)
        self.render_every_n = max(1, int(render_every_n))

    def sync_to_pcs(self, pcs_step_index: int):
        """Move the ISO's internal pointer to match the PCS half-hour index."""
        self._next_start_idx = int(pcs_step_index)

    def _get_iso_timestamp_from_pcs_index(self, pcs_index: int):
        return self.base_timestamp + timedelta(minutes=30 * int(pcs_index))

    def step(self, action):
        # 1) Price mapping logic
        raw_prices = action[:self.T]
        p_min, p_max = raw_prices.min(), raw_prices.max()
        denom = (p_max - p_min) + 1e-8
        normalized_prices = (raw_prices - p_min) / denom
        price_curve = 0.05 + (normalized_prices * 0.20)

        # 2) Anchor the index to the ISO's current position
        current_idx = self._next_start_idx
        current_iso_timestamp = self._get_iso_timestamp_from_pcs_index(current_idx)

        # 3) Setup PCS
        self.pcs_env.set_price_curve(price_curve.astype(np.float32))
        self.pcs_env.set_step_index(current_idx)
        obs, _ = self.pcs_env.reset(start_date=current_iso_timestamp)

        # render initial PCS state if requested
        if self.render_enabled:
            self.pcs_env.render()

        day_money = 0
        day_shortages = 0  # Track shortages for this day
        realized_consumption = []

        for step_i in range(self.T):
            pcs_action, _ = self.pcs_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.pcs_env.step(pcs_action)
            day_money += info.get('step_money', 0)

            # Collect actual consumption and shortages
            realized_consumption.append(info.get('consumption_units', 0))
            if info.get('shortage', False):
                day_shortages += 1

            # render during the PCS internal loop (throttled)
            if self.render_enabled and (self.pcs_env.current_step % self.render_every_n == 0):
                self.pcs_env.render()

        # 4) Calculate MAE between dispatch (prices imply expected consumption) and realized
        dispatch = action[self.T:]
        realized_array = np.array(realized_consumption, dtype=np.float32)
        mae = float(np.mean(np.abs(dispatch - realized_array)))

        # 5) Advance ISO pointer via super().step()
        _, _, _, _, iso_info = super().step(action)

        # 6) Update info with all metrics
        iso_info["money_earned"] = day_money
        iso_info["mae"] = mae
        iso_info["shortages"] = day_shortages
        iso_info["realized_consumption"] = realized_array
        iso_info["dispatch"] = dispatch

        # Store for render
        self._last_step_info["mae"] = mae
        self._last_step_info["money_earned"] = day_money
        self._last_step_info["shortages"] = day_shortages
        self._last_reward = day_money

        return self._expected, day_money, True, False, iso_info

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

        print(f"{'Time':<10} | {'CSV Row':<8} | {'Pred':<8} | {'Actual':<8} | {'Dispatch':<10} | {'Price':<6}")
        print("-" * 70)

        rows = []
        for i in range(0, self.T, 4):
            slot_time = info['timestamp'] + timedelta(minutes=30 * i)
            row_num = info['start_idx'] + i + 2

            p = step_info['predicted'][i]
            a = step_info['realized'][i]
            d = step_info['dispatch'][i]
            pr = step_info['prices'][i]

            print(f"{slot_time.strftime('%H:%M'):<10} | {row_num:<8} | {p:<8.3f} | {a:<8.3f} | {d:<10.3f} | {pr:<6.2f}")

            rows.append({
                "slot_time": slot_time,
                "row_num": int(row_num),
                "pred": float(p),
                "actual": float(a),
                "dispatch": float(d),
                "price": float(pr)
            })

        print("-" * 70)
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

    def step(self, action):
        # Call the original step method to run everything as usual
        expected, reward, done, truncated, info = super().step(action)

        # Apply shortage penalty (minimal change)
        penalties = info.get("shortages", 0) * self.shortage_penalty
        net_reward = reward - penalties

        # Update info and _last_reward for render() and tracking
        info["shortage_penalty"] = penalties
        info["net_reward"] = net_reward
        self._last_reward = net_reward

        return expected, net_reward, done, truncated, info

    def render(self):
        """
        Extend render to include penalty and net reward.
        Calls parent render and then prints extra info.
        """
        record = super().render()
        if record is not None:
            penalties = self._last_step_info.get("shortage_penalty", 0)
            net = self._last_step_info.get("net_reward", self._last_reward)
            print(f"Shortage Penalty Applied:       ${penalties:.2f}")
            print(f"Net Reward (after penalty):     ${net:.2f}\n")
        return record

def get_pred_window(preds: Sequence[float], start: int, length: int) -> np.ndarray:
    """Return a length-sized window starting at `start` from preds, wrapping if necessary."""
    n = len(preds)
    if n == 0:
        raise ValueError("Predictions array is empty.")
    start = int(start) % n
    if start + length <= n:
        return preds[start:start + length]
    # wrap-around case
    first = preds[start:]
    remaining = length - len(first)
    second = preds[:remaining]
    return np.concatenate([first, second], axis=0)


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
        verbose: int = 0,
        per_day_debug: bool = True,
        render: bool = False,
        render_every_n_steps: int = 1,
):
    """Tandem ISO <-> PCS training loop with convergence tracking."""

    if cycle_days < 1:
        raise ValueError("cycle_days must be >= 1")
    if total_iterations < 1:
        raise ValueError("total_iterations must be >= 1")

    pcs_algo_kwargs = {} if pcs_algo_kwargs is None else dict(pcs_algo_kwargs)

    temp_env = PCSEnv(
        test_data_file='../../tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
        predictions_file='../../tests/gym/data_for_tests/consumption_predictions.csv',
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

    pcs_model = pcs_algo_cls(pcs_policy, base_pcs_env, verbose=verbose, **pcs_algo_kwargs)

    iso_env = AlternatingISOEnv(
        actual_csv='../../tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
        predicted_csv='../../tests/gym/data_for_tests/consumption_predictions.csv',
        pcs_env=base_pcs_env,
        pcs_model=pcs_model,
        render_enabled=render,
        render_every_n=max(1, int(render_every_n_steps))
    )

    iso_model = iso_algo_cls(iso_policy, iso_env, verbose=verbose, **iso_algo_kwargs)

    pred_df = pd.read_csv('../../tests/gym/data_for_tests/consumption_predictions.csv', parse_dates=['timestamp']).set_index(
        'timestamp').sort_index()
    predicted_vals = pred_df['predicted_consumption'].astype(float).to_numpy().flatten()
    pricing = ISOPricingWrapper(iso_model)

    # --- NEW: Metric Tracking Initialization ---
    history = {"iteration": [], "avg_money": [], "avg_mae": [], "total_shortages": [], "avg_iso_price": []}

    for iteration in range(1, total_iterations + 1):
        print(f"\n[Iteration {iteration}] ISO Learning Phase ({cycle_days} Days)...")
        iso_model.learn(total_timesteps=cycle_days, reset_num_timesteps=False)

        print(f"[Iteration {iteration}] PCS Learning Phase...")

        base_idx = iso_env._next_start_idx
        day_money_list = []
        iteration_maes = []  # Track MAE for this iteration
        iteration_shortages = 0  # Track Shortages for this iteration
        day_avg_prices = []  # Track average ISO price per day (true price produced by ISO at this iteration)

        for day in range(cycle_days):
            start = base_idx + day * steps_per_day
            pred_window = get_pred_window(predicted_vals, start, steps_per_day).astype(np.float32)
            price_curve = pricing.generate_price_curve(pred_window)

            # record average price for this day (true price using current iso_model)
            try:
                day_avg_prices.append(float(np.mean(price_curve)))
            except Exception:
                day_avg_prices.append(0.0)

            base_pcs_env.set_price_curve(price_curve.astype(np.float32))
            base_pcs_env.set_step_index(start)
            obs, _ = base_pcs_env.reset(start_date=iso_env._get_iso_timestamp_from_pcs_index(start))

            eval_day_money = 0.0
            day_actuals = []

            # Use the dispatch from the ISO's prediction (second half of action)
            iso_action, _ = iso_model.predict(pred_window, deterministic=True)
            day_dispatch = iso_action[steps_per_day:]

            for step_i in range(steps_per_day):
                pcs_action, _ = pcs_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = base_pcs_env.step(pcs_action)

                eval_day_money += info.get('step_money', 0.0)
                day_actuals.append(info.get('consumption_units', 0.0))
                if info.get('shortage', False):
                    iteration_shortages += 1

            day_money_list.append(eval_day_money)
            iteration_maes.append(np.mean(np.abs(np.array(day_actuals) - day_dispatch)))

            base_pcs_env.set_step_index(start)
            obs, _ = base_pcs_env.reset(start_date=iso_env._get_iso_timestamp_from_pcs_index(start))
            pcs_model.learn(total_timesteps=steps_per_day, reset_num_timesteps=False)

        final_idx = base_idx + (cycle_days * steps_per_day)
        iso_env.sync_to_pcs(final_idx)

        # --- NEW: Store Metrics for this Iteration ---
        avg_money = float(np.mean(day_money_list))
        avg_mae = float(np.mean(iteration_maes))
        # average the per-day mean prices to make iteration-level avg ISO price
        avg_iso_price = float(np.mean(day_avg_prices)) if len(day_avg_prices) > 0 else 0.0

        history["iteration"].append(iteration)
        history["avg_money"].append(avg_money)
        history["avg_mae"].append(avg_mae)
        history["total_shortages"].append(iteration_shortages)
        history["avg_iso_price"].append(avg_iso_price)

        print(f">>> Iteration {iteration} Avg Money: ${avg_money:.2f} | Avg MAE: {avg_mae:.4f} | Avg ISO Price: ${avg_iso_price:.4f}")

    print("\n--- TRAINING COMPLETE ---")
    return history  # Return the history for plotting


if __name__ == "__main__":
    try:
        run_alternating_training(render=False, render_every_n_steps=1)
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        import traceback

        traceback.print_exc()
