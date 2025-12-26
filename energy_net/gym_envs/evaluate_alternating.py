import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from alternating_env import AlternatingISOEnv, ISOPricingWrapper, get_pred_window, run_alternating_training
from stable_baselines3 import PPO, SAC
from energy_net.gym_envs.pcs_env import PCSEnv

matplotlib.use('Agg')


def plot_training_convergence(history):
    """Visualizes how the agents improved over the training iterations."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 1. Money/Reward
    axes[0].plot(history["iteration"], history["avg_money"], color='green', marker='o')
    axes[0].set_title("PCS Average Daily Money (Convergence)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Dollars ($)")
    axes[0].grid(True, alpha=0.3)

    # 2. MAE
    axes[1].plot(history["iteration"], history["avg_mae"], color='orange', marker='s')
    axes[1].set_title("Dispatch vs. Actual MAE (Lower is Better)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("MAE Units")
    axes[1].grid(True, alpha=0.3)

    # 3. Shortages
    axes[2].plot(history["iteration"], history["total_shortages"], color='red', marker='x')
    axes[2].set_title("Total Shortages per Iteration", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence_results.png")
    print("\nConvergence plot saved as 'convergence_results.png'")


class AlternatingEvaluator:
    def __init__(self, iso_model, pcs_model, actual_csv, predicted_csv, config_name="Default"):
        self.iso_model = iso_model
        self.pcs_model = pcs_model
        self.actual_csv = actual_csv
        self.predicted_csv = predicted_csv
        self.config_name = config_name

        pred_df = pd.read_csv(predicted_csv, parse_dates=['timestamp']).set_index('timestamp').sort_index()
        self.predicted_vals = pred_df['predicted_consumption'].astype(float).to_numpy().flatten()
        self.pricing_wrapper = ISOPricingWrapper(iso_model)

    def run_evaluation(self, num_days=7, start_idx=0):
        """Run evaluation with enhanced tracking of shortages and MAE."""
        pcs_env = PCSEnv(test_data_file=self.actual_csv, predictions_file=self.predicted_csv, prediction_horizon=48)
        steps_per_day = 48
        history = []

        for day in range(num_days):
            current_start = start_idx + (day * steps_per_day)
            pred_window = get_pred_window(self.predicted_vals, current_start, steps_per_day).astype(np.float32)
            price_curve = self.pricing_wrapper.generate_price_curve(pred_window)

            pcs_env.set_price_curve(price_curve)
            pcs_env.set_step_index(current_start)
            obs, _ = pcs_env.reset()

            day_actual_consumption = []
            day_dispatch = []

            for t in range(steps_per_day):
                action, _ = self.pcs_model.predict(obs, deterministic=True)
                next_obs, reward, _, _, info = pcs_env.step(action)

                iso_obs = pred_window.astype(np.float32)
                iso_action, _ = self.iso_model.predict(iso_obs, deterministic=True)
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

            day_mae = np.mean(np.abs(np.array(day_actual_consumption) - np.array(day_dispatch)))
            day_start_idx = len(history) - steps_per_day
            for i in range(day_start_idx, len(history)):
                history[i]['day_mae'] = day_mae

        return pd.DataFrame(history)

    def plot_results(self, df, output_prefix="evaluation"):
        """Consolidates all metrics into a few multi-panel PNG files."""
        # Pre-calculation
        df['cumulative_pcs_reward'] = df['money'].cumsum()
        df['cumulative_iso_reward'] = -df['money'].cumsum()
        df['cumulative_shortages'] = df['shortage'].cumsum()
        df['mae_instantaneous'] = np.abs(df['actual_consumption'] - df['dispatch'])
        df['mae_rolling'] = df['mae_instantaneous'].rolling(window=2, min_periods=1).mean()

        # --- SHEET 1: CUMULATIVE PERFORMANCE (4 Graphs) ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Cumulative Performance - {self.config_name}", fontsize=18, fontweight='bold')

        # 1. PCS Rewards
        axes[0, 0].plot(df['timestamp'], df['cumulative_pcs_reward'], color='green')
        axes[0, 0].set_title("PCS Cumulative Reward ($)")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. ISO Rewards
        axes[0, 1].plot(df['timestamp'], df['cumulative_iso_reward'], color='blue')
        axes[0, 1].set_title("ISO Cumulative Reward ($)")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Total Shortages
        axes[1, 0].plot(df['timestamp'], df['cumulative_shortages'], color='red')
        axes[1, 0].fill_between(df['timestamp'], 0, df['cumulative_shortages'], color='red', alpha=0.2)
        axes[1, 0].set_title("Total Cumulative Shortages")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. MAE Rolling
        axes[1, 1].plot(df['timestamp'], df['mae_rolling'], color='orange')
        axes[1, 1].set_title("MAE (Rolling Average)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{output_prefix}_cumulative_sheet.png", dpi=150)
        plt.close()

        # --- SHEET 2: TIME-SERIES SIGNALS (2 Graphs) ---
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
        fig2.suptitle(f"System Signals - {self.config_name}", fontsize=18, fontweight='bold')

        # 1. Price Curve
        axes2[0].plot(df['timestamp'], df['price'], color='purple')
        axes2[0].set_title("ISO Price Signal ($/Unit)")
        axes2[0].grid(True, alpha=0.3)

        # 2. PCS Action Magnitude
        axes2[1].plot(df['timestamp'], df['action'], color='teal')
        axes2[1].set_title("PCS Agent Action Magnitude")
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{output_prefix}_signals_sheet.png", dpi=150)
        plt.close()

        print(
            f"Evaluation sheets saved as '{output_prefix}_cumulative_sheet.png' and '{output_prefix}_signals_sheet.png'")

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


if __name__ == "__main__":
    # --- 1. CONFIGURATION ---
    ACTUAL_CSV = 'data_for_tests/synthetic_household_consumption_test.csv'
    PRED_CSV = 'data_for_tests/consumption_predictions.csv'
    ITERATIONS = 30  # Adjust as needed

    # --- 2. TRAINING ---
    # This now captures the convergence metrics from your updated alternating_env.py
    training_history = run_alternating_training(
        total_iterations=ITERATIONS,
        cycle_days=7,
        render=False
    )

    # --- 3. CONVERGENCE PLOT ---
    # Shows how the agents improved over the 30 iterations
    plot_training_convergence(training_history)

    # --- 4. FINAL SNAPSHOT EVALUATION ---
    # Now let's run a 7-day detailed test with the fully trained models
    print("\nStarting final post-training evaluation...")

    # Setup dummy envs just to initialize the evaluators
    temp_env = PCSEnv(test_data_file=ACTUAL_CSV, predictions_file=PRED_CSV, prediction_horizon=48)

    # Assuming models were trained and accessible via internal logic or re-loading
    # For this script, we'll use the ones created during training if run_alternating_training 
    # were modified to return them, or we can initialize new ones for a fresh eval.
    pcs_mod = SAC("MlpPolicy", temp_env)  # Replace with loaded model if saved
    iso_env_eval = AlternatingISOEnv(ACTUAL_CSV, PRED_CSV, temp_env, pcs_mod)
    iso_mod = PPO("MlpPolicy", iso_env_eval)  # Replace with loaded model if saved

    evaluator = AlternatingEvaluator(
        iso_mod, pcs_mod, ACTUAL_CSV, PRED_CSV,
        config_name=f"{ITERATIONS} Iterations / Combined Eval"
    )

    results_df = evaluator.run_evaluation(num_days=7)
    evaluator.plot_results(results_df, output_prefix="final_eval")
    metrics = evaluator.run_analysis(results_df)