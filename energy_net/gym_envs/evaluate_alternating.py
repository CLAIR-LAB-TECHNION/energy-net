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
from energy_net.gym_envs.alternating_env import (
    AlternatingISOEnv, 
    PenalizedAlternatingISOEnv,
    MultiObjectiveAlternatingISOEnv,
    get_pred_window, 
    run_alternating_training
)
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
        """
        Evaluator for alternating ISO/PCS experiments.

        Parameters
        ----------
        iso_model : RL model
            The ISO agent/model used to generate dispatch decisions.
        pcs_model : RL model
            The PCS agent/model used to make charging/discharging actions.
        actual_csv : str
            Path to CSV containing actual test data for the PCS environment.
        predicted_csv : str
            Path to CSV containing predicted consumption values for the ISO model.
        config_name : str, optional
            Human-readable name for this evaluation configuration (default "Default").

        Sets up:
            - ISOEnv for timestamp/indexing utilities
            - RLPriceCurveStrategy initialized with the iso_model
            - loads predicted values into self.predicted_vals for quick access
        """
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
        """
        Run multi-day evaluation of PCS vs ISO using the provided models.

        Parameters
        ----------
        num_days : int
            Number of days to simulate/evaluate. Each day uses `steps_per_day` steps.
        start_idx : int
            Starting step index within the dataset (offset in 30-minute steps).

        Returns
        -------
        pandas.DataFrame
            A DataFrame with a history of per-step observations and metrics including:
                - timestamp, price, action, money, shortage, actual_consumption, dispatch
                - day_mae for each step of that day
        """
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
        """
        Create and save a set of plots summarizing evaluation results.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame returned by run_evaluation (or with equivalent columns).
        output_prefix : str
            Prefix for saved image files (two PNGs will be created:
            {output_prefix}_cumulative_sheet.png and {output_prefix}_signals_sheet.png).
        """
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
        """
        Compute and print high-level summary metrics from evaluation DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Evaluation history (must include 'money', 'shortage', 'actual_consumption', 'dispatch').

        Returns
        -------
        dict
            Dictionary with keys:
                - 'pcs_reward' : total PCS money (sum of 'money')
                - 'total_shortages' : total number of shortage events (int)
                - 'avg_mae' : average instantaneous MAE across dataset
        """
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
    """
    Make a run identifier string.

    If provided_id is given, it is returned unchanged. Otherwise a timestamped id
    with a short UUID suffix is generated in the format YYYYmmdd-HHMMSS-<8hex>.

    Parameters
    ----------
    provided_id : str or None
        Optional user-provided run id.

    Returns
    -------
    str
        The run id string.
    """
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
                   run_id=None,
                   env_class=AlternatingISOEnv,
                   env_kwargs=None,
                   config_name=None):
    """
    Run the full pipeline (training, convergence plot, evaluation) with configurable environment.
    
    Parameters
    ----------
    actual_csv : str
        Path to actual consumption CSV
    pred_csv : str
        Path to predictions CSV  
    iterations : int
        Number of training iterations
    num_days : int
        Number of days to evaluate
    out_dir : str
        Output directory for results
    run_id : str, optional
        Run identifier
    env_class : class, optional
        Environment class to use (AlternatingISOEnv, PenalizedAlternatingISOEnv, 
        MultiObjectiveAlternatingISOEnv). Default: AlternatingISOEnv
    env_kwargs : dict, optional
        Additional kwargs to pass to env_class constructor (e.g., shortage_weight, mae_weight)
    config_name : str, optional
        Human-readable name for this configuration
    
    Returns
    -------
    tuple
        (metrics_dict, results_df)
    """
    run_id = make_run_id(run_id)
    os.makedirs(out_dir, exist_ok=True)
    output_prefix = os.path.join(out_dir, f"run_{run_id}")
    
    if env_kwargs is None:
        env_kwargs = {}
    
    if config_name is None:
        config_name = f"{env_class.__name__} - {iterations} Iterations"

    print(f"\nStarting experiment")
    print(f"  run_id: {run_id}")
    print(f"  env_class: {env_class.__name__}")
    print(f"  env_kwargs: {env_kwargs}")
    print(f"  actual: '{actual_csv}'")
    print(f"  pred: '{pred_csv}'")

    # TRAINING
    print("\n[1/4] Training models...")
    
    # Extract pcs_env_kwargs if provided
    pcs_env_kwargs = env_kwargs.pop('pcs_env_kwargs', {})
    temp_pcs_env = PCSEnv(
        test_data_file=actual_csv, 
        predictions_file=pred_csv, 
        prediction_horizon=48,
        **pcs_env_kwargs
    )
    
    # Create PCS model
    pcs_mod = PPO("MlpPolicy", temp_pcs_env, verbose=0, n_steps=48, batch_size=48)
    
    # Create ISO environment with specified class and kwargs
    iso_env_train = env_class(
        actual_csv=actual_csv,
        predicted_csv=pred_csv,
        pcs_env=temp_pcs_env,
        pcs_model=pcs_mod,
        **env_kwargs
    )
    
    # Create ISO model
    iso_mod = PPO("MlpPolicy", iso_env_train, verbose=0, n_steps=7, batch_size=7)
    
    # Link models
    iso_env_train.iso_model = iso_mod
    pricing_strategy = RLPriceCurveStrategy(iso_model=iso_mod)
    
    # Training loop (simplified version of run_alternating_training)
    history = {"iteration": [], "avg_money": [], "avg_mae": [], "total_shortages": [], "avg_iso_price": []}
    steps_per_day = 48
    cycle_days = 7
    
    for iteration in range(1, iterations + 1):
        # ISO learns
        iso_mod.learn(total_timesteps=cycle_days, reset_num_timesteps=False)
        
        # Evaluate and train PCS
        base_idx = iso_env_train._next_start_idx
        day_money_list = []
        iteration_maes = []
        iteration_shortages = 0
        day_avg_prices = []
        
        for day in range(cycle_days):
            start = base_idx + day * steps_per_day
            obs_window = get_pred_window(iso_env_train, start, steps_per_day)
            price_curve = pricing_strategy.calculate_price(obs_window)
            day_avg_prices.append(float(np.mean(price_curve)))
            
            temp_pcs_env.set_price_strategy(pricing_strategy)
            temp_pcs_env.set_step_index(start)
            obs, _ = temp_pcs_env.reset(start_date=iso_env_train._get_iso_timestamp_from_pcs_index(start))
            
            eval_day_money = 0.0
            day_actuals = []
            iso_action, _ = iso_mod.predict(obs_window, deterministic=True)
            day_dispatch = iso_action[steps_per_day:]
            
            for step_i in range(steps_per_day):
                pcs_action, _ = pcs_mod.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = temp_pcs_env.step(pcs_action)
                eval_day_money += info.get('step_money', 0.0)
                day_actuals.append(info.get('consumption_units', 0.0))
                if info.get('shortage', False):
                    iteration_shortages += 1
            
            day_money_list.append(eval_day_money)
            iteration_maes.append(np.mean(np.abs(np.array(day_actuals) - day_dispatch)))
            
            # PCS learns
            temp_pcs_env.set_step_index(start)
            obs, _ = temp_pcs_env.reset(start_date=iso_env_train._get_iso_timestamp_from_pcs_index(start))
            pcs_mod.learn(total_timesteps=steps_per_day, reset_num_timesteps=False)
        
        # Update history
        final_idx = base_idx + (cycle_days * steps_per_day)
        iso_env_train.sync_to_pcs(final_idx)
        
        avg_money = float(np.mean(day_money_list))
        avg_mae = float(np.mean(iteration_maes))
        avg_iso_price = float(np.mean(day_avg_prices))
        
        history["iteration"].append(iteration)
        history["avg_money"].append(avg_money)
        history["avg_mae"].append(avg_mae)
        history["total_shortages"].append(iteration_shortages)
        history["avg_iso_price"].append(avg_iso_price)
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}/{iterations}: Money=${avg_money:.2f}, Shortages={iteration_shortages}, MAE={avg_mae:.4f}")

    print("\n[2/4] Evaluating trained models...")
    evaluator = AlternatingEvaluator(
        iso_mod, pcs_mod, actual_csv, pred_csv,
        config_name=config_name
    )

    results_df = evaluator.run_evaluation(num_days=num_days)
    
    print("\n[3/4] Generating plots...")
    evaluator.plot_results(results_df, output_prefix=output_prefix)
    
    print("\n[4/4] Running analysis...")
    metrics = evaluator.run_analysis(results_df)

    plot_training_convergence(history, output_prefix=output_prefix)

    return metrics, results_df

if __name__ == "__main__":
    # --- Example configurations for different reward structures ---
    # Each will produce convergence plot + cumulative/signals sheets
    
    data_csv = "../../tests/gym/data_for_tests/synthetic_household_consumption_test.csv"
    pred_csv = "../../tests/gym/data_for_tests/consumption_predictions.csv"
    
    runs = [
        # 1. Baseline: Money-only reward
        {
            "actual_csv": data_csv,
            "pred_csv": pred_csv,
            "iterations": 30,
            "num_days": 7,
            "out_dir": "../../tests/gym/results",
            "run_id": "baseline_money_only",
            "env_class": AlternatingISOEnv,
            "env_kwargs": {},
            "config_name": "Baseline (Money Only)"
        },
        
        # 2. Penalized: Simple shortage penalty
        {
            "actual_csv": data_csv,
            "pred_csv": pred_csv,
            "iterations": 30,
            "num_days": 7,
            "out_dir": "../../tests/gym/results",
            "run_id": "penalized_100",
            "env_class": PenalizedAlternatingISOEnv,
            "env_kwargs": {"shortage_penalty": 100.0},
            "config_name": "Penalized ($100/shortage)"
        },
        
        # 3. Multi-Objective: Balanced approach (BEST FROM COMPARISON)
        {
            "actual_csv": data_csv,
            "pred_csv": pred_csv,
            "iterations": 30,
            "num_days": 7,
            "out_dir": "../../tests/gym/results",
            "run_id": "multi_obj_balanced",
            "env_class": MultiObjectiveAlternatingISOEnv,
            "env_kwargs": {
                "shortage_weight": 10.0,
                "mae_weight": 5.0,
                "money_weight": 1.0
            },
            "config_name": "Multi-Objective (Balanced: 10/5/1)"
        },
        
        # 4. Multi-Objective: Reliability-focused
        {
            "actual_csv": data_csv,
            "pred_csv": pred_csv,
            "iterations": 30,
            "num_days": 7,
            "out_dir": "../../tests/gym/results",
            "run_id": "multi_obj_reliability",
            "env_class": MultiObjectiveAlternatingISOEnv,
            "env_kwargs": {
                "shortage_weight": 20.0,
                "mae_weight": 5.0,
                "money_weight": 1.0
            },
            "config_name": "Multi-Objective (Reliability: 20/5/1)"
        },
    ]

    print("="*80)
    print("RUNNING VISUAL EVALUATION OF ALTERNATING ENVIRONMENTS")
    print("="*80)
    print(f"\nThis will generate visual outputs for {len(runs)} configurations:")
    for i, r in enumerate(runs, 1):
        print(f"  {i}. {r['config_name']}")
    print("\nOutputs will be saved to:", runs[0]['out_dir'])
    print("="*80)

    # Run them sequentially
    for idx, r in enumerate(runs, 1):
        print(f"\n{'='*80}")
        print(f"RUNNING CONFIGURATION {idx}/{len(runs)}")
        print(f"{'='*80}")
        run_experiment(**r)
    
    print(f"\n{'='*80}")
    print("ALL EVALUATIONS COMPLETE!")
    print(f"{'='*80}")
    print("\nCheck the results directory for:")
    print("  • *_convergence.png - Training progress over iterations")
    print("  • *_cumulative_sheet.png - Cumulative performance metrics")
    print("  • *_signals_sheet.png - ISO prices and PCS actions")
    print(f"{'='*80}\n")
