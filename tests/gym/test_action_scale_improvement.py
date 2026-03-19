#!/usr/bin/env python3
"""
Test script to demonstrate the action_scale improvement.
Runs evaluate_alternating.py with:
1. Baseline (no action_scale)
2. With action_scale=10.0
3. With MultiObjectiveAlternatingISOEnv

Results are labeled differently for comparison.
"""

import sys
import os
sys.path.insert(0, '../../')

from energy_net.gym_envs.evaluate_alternating import run_experiment
from energy_net.gym_envs.alternating_env import (
    AlternatingISOEnv,
    MultiObjectiveAlternatingISOEnv
)

# Common parameters - use absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ACTUAL_CSV = os.path.join(script_dir, 'data_for_tests/synthetic_household_consumption_test.csv')
PRED_CSV = os.path.join(script_dir, 'data_for_tests/consumption_predictions.csv')
OUT_DIR = os.path.join(script_dir, 'results')
ITERATIONS = 30
NUM_DAYS = 7

print("="*80)
print("TESTING ACTION SCALE IMPROVEMENT")
print("="*80)
print("\nRunning 3 configurations:")
print("  1. Baseline (no action_scale)")
print("  2. With action_scale=10.0")
print("  3. MultiObjectiveAlternatingISOEnv")
print(f"\nEach will train for {ITERATIONS} iterations\n")

# Test 1: Baseline (no action_scale)
print("\n" + "="*80)
print("TEST 1: BASELINE (NO ACTION SCALE)")
print("="*80)
metrics1, df1 = run_experiment(
    actual_csv=ACTUAL_CSV,
    pred_csv=PRED_CSV,
    iterations=ITERATIONS,
    num_days=NUM_DAYS,
    out_dir=OUT_DIR,
    run_id="baseline_no_action_scale",
    env_class=AlternatingISOEnv,
    env_kwargs={},
    config_name="Baseline (no action_scale)"
)

print(f"\nBaseline Results:")
print(f"   Avg Shortages: {metrics1.get('avg_shortages', 'N/A')}")
print(f"   Avg Money: ${metrics1.get('avg_money', 0):.2f}")
print(f"   Avg MAE: {metrics1.get('avg_mae', 'N/A'):.4f}")

# Test 2: With action_scale=10.0
print("\n" + "="*80)
print("TEST 2: WITH ACTION_SCALE=10.0")
print("="*80)
metrics2, df2 = run_experiment(
    actual_csv=ACTUAL_CSV,
    pred_csv=PRED_CSV,
    iterations=ITERATIONS,
    num_days=NUM_DAYS,
    out_dir=OUT_DIR,
    run_id="with_action_scale_10x",
    env_class=AlternatingISOEnv,
    env_kwargs={'pcs_env_kwargs': {'action_scale': 10.0}},
    config_name="With action_scale=10.0"
)

print(f"\nAction Scale Results:")
print(f"   Avg Shortages: {metrics2.get('avg_shortages', 'N/A')}")
print(f"   Avg Money: ${metrics2.get('avg_money', 0):.2f}")
print(f"   Avg MAE: {metrics2.get('avg_mae', 'N/A'):.4f}")

# Test 3: MultiObjectiveAlternatingISOEnv
print("\n" + "="*80)
print("TEST 3: MULTI-OBJECTIVE ENVIRONMENT")
print("="*80)
metrics3, df3 = run_experiment(
    actual_csv=ACTUAL_CSV,
    pred_csv=PRED_CSV,
    iterations=ITERATIONS,
    num_days=NUM_DAYS,
    out_dir=OUT_DIR,
    run_id="multi_objective_env",
    env_class=MultiObjectiveAlternatingISOEnv,
    env_kwargs={
        'shortage_weight': 10.0,
        'mae_weight': 5.0,
        'money_weight': 1.0,
        'pcs_env_kwargs': {'action_scale': 10.0}
    },
    config_name="MultiObjective (w/ action_scale)"
)

print(f"\nMulti-Objective Results:")
print(f"   Avg Shortages: {metrics3.get('avg_shortages', 'N/A')}")
print(f"   Avg Money: ${metrics3.get('avg_money', 0):.2f}")
print(f"   Avg MAE: {metrics3.get('avg_mae', 'N/A'):.4f}")

# Final Comparison
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"\n{'Configuration':<40} {'Shortages':<12} {'Money':<12} {'MAE':<10}")
print("-"*80)
print(f"{'Baseline (no action_scale)':<40} {metrics1.get('avg_shortages', 'N/A'):<12} ${metrics1.get('avg_money', 0):<10.2f} {metrics1.get('avg_mae', 0):<10.4f}")
print(f"{'With action_scale=10.0':<40} {metrics2.get('avg_shortages', 'N/A'):<12} ${metrics2.get('avg_money', 0):<10.2f} {metrics2.get('avg_mae', 0):<10.4f}")
print(f"{'MultiObjective (w/ action_scale)':<40} {metrics3.get('avg_shortages', 'N/A'):<12} ${metrics3.get('avg_money', 0):<10.2f} {metrics3.get('avg_mae', 0):<10.4f}")
print("-"*80)

# Calculate improvements
shortage_improvement = metrics1.get('avg_shortages', 0) - metrics2.get('avg_shortages', 0)
print(f"\nAction Scale Impact:")
print(f"   Shortage Reduction: {shortage_improvement:.1f} shortages/episode")
if metrics1.get('avg_shortages', 0) > 0:
    pct = 100 * shortage_improvement / metrics1.get('avg_shortages')
    print(f"   Percent Improvement: {pct:.1f}%")

print(f"\nResults saved to:")
print(f"   - results/run_baseline_no_action_scale_*")
print(f"   - results/run_with_action_scale_10x_*")
print(f"   - results/run_multi_objective_env_*")
print("="*80 + "\n")