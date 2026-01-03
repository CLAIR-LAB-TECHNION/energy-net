import os
import uuid
from datetime import datetime, timedelta

import matplotlib

from energy_net.gym_envs.iso_env import ISOEnv

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_net.grid_entities.management.price_curve import RLPriceCurveStrategy  # New strategy class
from alternating_env import AlternatingISOEnv, get_pred_window, run_alternating_training
from stable_baselines3 import PPO, SAC
from energy_net.gym_envs.pcs_env import PCSEnv


def plot_training_convergence(history, output_prefix="convergence_results"):
    """
    Visualizes training convergence.

    Assumes `history` (DataFrame) contains:
        - 'iteration'
        - 'avg_money'
        - 'total_shortages'
        - 'avg_iso_price' (middle subplot)
    """
    # ensure history is a DataFrame
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 1. PCS Average Money (green)
    axes[0].plot(history["iteration"], history["avg_money"], color='green', marker='o')
    axes[0].set_title("PCS Average Daily Money (Convergence)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Dollars ($)")
    axes[0].grid(True, alpha=0.3)

    # 2. Average ISO Price per iteration (purple)
    axes[1].plot(history["iteration"], history["avg_iso_price"], color='purple', marker='o')
    axes[1].set_title("Average ISO Price per Training Iteration", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Price ($/Unit)")
    axes[1].set_xlabel("Training iteration")
    axes[1].grid(True, alpha=0.3)

    # 3. Total Shortages (red)
    axes[2].plot(history["iteration"], history["total_shortages"], color='red', marker='x')
    axes[2].set_title("Total Shortages per Iteration", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{output_prefix}_convergence.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"\nConvergence plot saved as '{out_path}'")


class AlternatingEvaluator:
    def __init__(self, iso_model, pcs_model, actual_csv, predicted_csv, config_name="Default"):
        self.iso_model = iso_model
        self.pcs_model = pcs_model
        self.iso_env = ISOEnv(actual_csv, predicted_csv)
        self.config_name = config_name

        # 2. Use the new Strategy Class instead of the Wrapper
        self.pricing_strategy = RLPriceCurveStrategy(iso_model)

        self.actual_csv = actual_csv
        self.predicted_csv = predicted_csv

        pred_df = pd.read_csv(predicted_csv, parse_dates=['timestamp']).set_index('timestamp').sort_index()
        self.predicted_vals = pred_df['predicted_consumption'].astype(float).to_numpy().flatten()

    def run_evaluation(self, num_days=7, start_idx=0):
        pcs_env = PCSEnv(test_data_file=self.actual_csv, predictions_file=self.predicted_csv, prediction_horizon=48)
        steps_per_day = 48
        history = []

        for day in range(num_days):
            current_start = start_idx + (day * steps_per_day)

            # 3. Get the full observation window (Predictions + Features)
            pred_window = get_pred_window(self.iso_env, current_start, steps_per_day).astype(np.float32)

            # 4. Generate prices using the Strategy class
            price_curve = self.pricing_strategy.calculate_price(pred_window)

            # 5. Inject the strategy object into PCSEnv
            pcs_env.set_price_strategy(self.pricing_strategy)
            pcs_env.set_step_index(current_start)
            obs, _ = pcs_env.reset(start_date=self.iso_env.base_timestamp + timedelta(minutes=30 * current_start))

            day_actual_consumption = []
            day_dispatch = []

            for t in range(steps_per_day):
                action, _ = self.pcs_model.predict(obs, deterministic=True)
                next_obs, reward, _, _, info = pcs_env.step(action)

                # Query the ISO model for the specific dispatch target
                iso_action, _ = self.iso_model.predict(pred_window, deterministic=True)
                dispatch_value = iso_action[steps_per_day + t]

                day_actual_consumption.append(info.get('consumption_units', 0))
                day_dispatch.append(dispatch_value)

                history.append({
                    "timestamp": pcs_env.current_datetime - timedelta(days=pcs_env.dt),
                    "price": price_curve[t],
                    "action": float(action[0]),
                    "money": info.get('step_money', 0),
                    "shortage": 1 if info.get('shortage', False) else 0,
                    "actual_consumption": info.get('consumption_units', 0),
                    "dispatch": dispatch_value,
                })
                obs = next_obs

            # Calculate Daily MAE
            day_mae = np.mean(np.abs(np.array(day_actual_consumption) - np.array(day_dispatch)))
            day_start_idx = len(history) - steps_per_day
            for i in range(day_start_idx, len(history)):
                history[i]['day_mae'] = day_mae

        return pd.DataFrame(history)
    def plot_results(self, df, output_prefix="evaluation"):
        df['cumulative_pcs_reward'] = df['money'].cumsum()
        df['cumulative_iso_reward'] = -df['money'].cumsum()
        df['cumulative_shortages'] = df['shortage'].cumsum()
        df['mae_instantaneous'] = np.abs(df['actual_consumption'] - df['dispatch'])
        df['mae_rolling'] = df['mae_instantaneous'].rolling(window=2, min_periods=1).mean()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Cumulative Performance - {self.config_name}", fontsize=18, fontweight='bold')

        axes[0, 0].plot(df['timestamp'], df['cumulative_pcs_reward'], color='green')
        axes[0, 0].set_title("PCS Cumulative Reward ($)")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(df['timestamp'], df['cumulative_iso_reward'], color='blue')
        axes[0, 1].set_title("ISO Cumulative Reward ($)")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(df['timestamp'], df['cumulative_shortages'], color='red')
        axes[1, 0].fill_between(df['timestamp'], 0, df['cumulative_shortages'], color='red', alpha=0.2)
        axes[1, 0].set_title("Total Cumulative Shortages")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(df['timestamp'], df['mae_rolling'], color='orange')
        axes[1, 1].set_title("MAE (Rolling Average)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out1 = f"{output_prefix}_cumulative_sheet.png"
        plt.savefig(out1, dpi=150)
        plt.close(fig)

        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
        fig2.suptitle(f"System Signals - {self.config_name}", fontsize=18, fontweight='bold')

        axes2[0].plot(df['timestamp'], df['price'], color='purple')
        axes2[0].set_title("ISO Price Signal ($/Unit)")
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(df['timestamp'], df['action'], color='teal')
        axes2[1].set_title("PCS Agent Action Magnitude")
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out2 = f"{output_prefix}_signals_sheet.png"
        plt.savefig(out2, dpi=150)
        plt.close(fig2)

        print(f"Evaluation sheets saved as '{out1}' and '{out2}'")

    def run_analysis(self, df):
        if 'mae_instantaneous' not in df.columns:
            df['mae_instantaneous'] = np.abs(df['actual_consumption'] - df['dispatch'])

        total_pcs_reward = df['money'].sum()
        total_iso_reward = -total_pcs_reward
        total_shortages = df['shortage'].sum()
        avg_mae = df['mae_instantaneous'].mean()

        print(f"\n{'=' * 60}")
        print(f"COMPREHENSIVE EVALUATION SUMMARY - {self.config_name}")
        print(f"{'=' * 60}")
        print(f"PCS Total Reward:       ${total_pcs_reward:,.2f}")
        print(f"ISO Total Reward:       ${total_iso_reward:,.2f}")
        print(f"Total Shortages:        {int(total_shortages)}")
        print(f"Average MAE:            {avg_mae:.4f}")
        print(f"{'=' * 60}\n")

        return {'pcs_reward': total_pcs_reward, 'total_shortages': int(total_shortages), 'avg_mae': avg_mae}


def make_run_id(provided_id=None):
    if provided_id:
        return provided_id
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}-{short}"


def run_experiment(actual_csv,
                   pred_csv,
                   iterations=30,
                   num_days=7,
                   out_dir="../../tests/gym/results",
                   run_id=None):
    """
    Run the full pipeline (training, convergence plot, evaluation).
    Returns (metrics_dict, results_df).
    """
    run_id = make_run_id(run_id)
    os.makedirs(out_dir, exist_ok=True)
    output_prefix = os.path.join(out_dir, f"run_{run_id}")

    print(f"Starting experiment run_id={run_id}  actual='{actual_csv}'  pred='{pred_csv}'")

    # TRAINING: expect run_alternating_training to return a dict-like history with keys per-iteration
    training_history = run_alternating_training(
        total_iterations=iterations,
        cycle_days=7,
        render=False
    )

    # 6. Post-Training Evaluation Setup
    # Note: In a real scenario, you would use the models returned/saved by training.
    # For this script, we assume they are initialized and then evaluated.
    temp_env = PCSEnv(test_data_file=actual_csv, predictions_file=pred_csv, prediction_horizon=48)

    # Dummy initialization for structural evaluation (replace with loaded models if needed)
    pcs_mod = SAC("MlpPolicy", temp_env)
    iso_env_eval = AlternatingISOEnv(actual_csv, pred_csv, temp_env, pcs_mod)
    iso_mod = PPO("MlpPolicy", iso_env_eval)

    evaluator = AlternatingEvaluator(
        iso_mod, pcs_mod, actual_csv, pred_csv,
        config_name=f"{iterations} Iterations / RL Strategy Eval"
    )

    results_df = evaluator.run_evaluation(num_days=num_days)
    evaluator.plot_results(results_df, output_prefix=output_prefix)
    metrics = evaluator.run_analysis(results_df)

    plot_training_convergence(training_history, output_prefix=output_prefix)

    return metrics, results_df

if __name__ == "__main__":
    # --- Example: list multiple runs here, edit file paths as needed ---
    runs = [
        {
            "actual_csv": "../../tests/gym/data_for_tests/synthetic_household_consumption_test.csv",
            "pred_csv": "../../tests/gym/data_for_tests/consumption_predictions.csv",
            "iterations": 30,
            "num_days": 7,
            "out_dir": "../../tests/gym/results",
            "run_id": "normal_run",
        },
        {
            "actual_csv": "../../tests/gym/data_for_tests/zero_consumption.csv",
            "pred_csv": "../../tests/gym/data_for_tests/zero_consumption_predictions.csv",
            "iterations": 30,
            "num_days": 7,
            "out_dir": "../../tests/gym/results",
            "run_id": "zero_consumption_run",
        },
    ]

    # Run them sequentially; each will produce files like:
    # ../../tests/gym/results/run_<run_id>_convergence.png
    # ../../tests/gym/results/run_<run_id>_cumulative_sheet.png
    # ../../tests/gym/results/run_<run_id>_signals_sheet.png
    for r in runs:
        run_experiment(**r)
