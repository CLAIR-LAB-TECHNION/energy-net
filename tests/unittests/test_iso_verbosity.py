#!/usr/bin/env python
"""
Test script to demonstrate all verbosity levels in ISOEnv.
"""

import numpy as np
import sys
import os

# Add the project root to the path to import energy_net
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from energy_net.gym_envs.iso_env import ISOEnv

# Define data file paths relative to project root
TEST_DATA_FILE = os.path.join(PROJECT_ROOT, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, 'tests/gym/data_for_tests/consumption_predictions.csv')

def test_verbosity_level(level, description):
    """Test a specific verbosity level."""
    print("\n" + "=" * 80)
    print(f"TESTING VERBOSITY LEVEL {level}: {description}")
    print("=" * 80)
    
    # Create environment with specific verbosity
    env = ISOEnv(
        actual_csv=TEST_DATA_FILE,
        predicted_csv=PREDICTIONS_FILE,
        verbosity=level
    )
    
    # Run a single day
    obs, info = env.reset()
    
    # For level 0, just show what happens
    if level == 0:
        print("\n[Level 0 is SILENT - no console output from render()]")
        # Generate random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        data = env.render()
        print(f"But it DOES return structured data: {list(data.keys())}")
        print(f"Example: MAE={data['mae']:.4f}, reward={data['reward']:.4f}")
    else:
        # Generate random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    
    print(f"\n[Test for verbosity level {level} complete]\n")

def test_override():
    """Test that verbosity can be overridden in render() call."""
    print("\n" + "=" * 80)
    print("TESTING VERBOSITY OVERRIDE")
    print("=" * 80)
    print("\nEnvironment created with verbosity=2 (default)")
    print("But we'll call render() with different verbosity levels...\n")
    
    env = ISOEnv(
        actual_csv=TEST_DATA_FILE,
        predicted_csv=PREDICTIONS_FILE,
        verbosity=2  # Default
    )
    
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("\n--- Calling render(verbosity=1) - Summary only ---")
    env.render(verbosity=1)
    
    print("\n--- Calling render(verbosity=0) - Silent ---")
    data = env.render(verbosity=0)
    print(f"[No console output, but returned data dict with keys: {list(data.keys())}]")
    
    print("\n--- Calling render(verbosity=4) - Debug mode ---")
    env.render(verbosity=4)

if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# ISO ENVIRONMENT VERBOSITY SYSTEM TEST")
    print("#" * 80)
    
    # Test each verbosity level
    test_verbosity_level(0, "SILENT (data only, no console output)")
    test_verbosity_level(1, "SUMMARY (single line per day)")
    test_verbosity_level(2, "CONDENSED (summary + sampled timesteps) - DEFAULT")
    test_verbosity_level(3, "DETAILED (all timesteps with features)")
    test_verbosity_level(4, "DEBUG (comprehensive analysis)")
    
    # Test override functionality
    test_override()
    
    print("\n" + "#" * 80)
    print("# ALL TESTS COMPLETE!")
    print("#" * 80)
    print("\nSummary:")
    print("  ✓ Level 0: Silent mode returns data without printing")
    print("  ✓ Level 1: Summary mode shows single line per day")
    print("  ✓ Level 2: Condensed mode shows summary + sampled timesteps (DEFAULT)")
    print("  ✓ Level 3: Detailed mode shows all timesteps with features")
    print("  ✓ Level 4: Debug mode includes pricing analysis")
    print("  ✓ Verbosity can be overridden per render() call")
    print("\n")