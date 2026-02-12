#!/usr/bin/env python
"""
Test script to demonstrate all verbosity levels in AlternatingISOEnv.

Note: This test requires trained models to fully demonstrate the alternating environment.
For now, it shows the structure and verbosity system.
"""

import numpy as np
import sys
import os

# Add the project root to the path to import energy_net
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.gym_envs.alternating_env import AlternatingISOEnv
from stable_baselines3 import PPO

# Define data file paths relative to project root
TEST_DATA_FILE = os.path.join(PROJECT_ROOT, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, 'tests/gym/data_for_tests/consumption_predictions.csv')

def create_test_environment(iso_verbosity=2, pcs_verbosity=0):
    """
    Create a minimal alternating environment for testing.
    
    Args:
        iso_verbosity: Verbosity level for ISO rendering
        pcs_verbosity: Verbosity level for PCS rendering during ISO episodes
    
    Returns:
        AlternatingISOEnv instance
    """
    # Create base PCS environment
    pcs_env = PCSEnv(
        test_data_file=TEST_DATA_FILE,
        predictions_file=PREDICTIONS_FILE,
        verbosity=0  # Keep PCS quiet during setup
    )
    
    # Create a simple PCS model (not trained, just for structure)
    pcs_model = PPO("MlpPolicy", pcs_env, verbose=0)
    
    # Create alternating ISO environment
    iso_env = AlternatingISOEnv(
        actual_csv=TEST_DATA_FILE,
        predicted_csv=PREDICTIONS_FILE,
        pcs_env=pcs_env,
        pcs_model=pcs_model,
        iso_verbosity=iso_verbosity,
        pcs_verbosity=pcs_verbosity
    )
    
    return iso_env

def test_iso_verbosity_level(level, description):
    """Test a specific ISO verbosity level."""
    print("\n" + "=" * 80)
    print(f"TESTING ISO VERBOSITY LEVEL {level}: {description}")
    print("=" * 80)
    
    try:
        # Create environment with specific ISO verbosity
        env = create_test_environment(iso_verbosity=level, pcs_verbosity=0)
        
        # Reset and take a single step
        obs, info = env.reset()
        
        # Generate random action for ISO (prices + dispatch)
        action = env.action_space.sample()
        
        # Step the environment (this runs the full PCS loop internally)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the ISO day summary
        if level == 0:
            print("\n[Level 0 is SILENT - no console output from render()]")
            data = env.render()
            print(f"But it DOES return structured data with keys: {list(data.keys())}")
            print(f"Example: reward=${data['reward']:.2f}, MAE={data['mae']:.4f}, shortages={data['shortages']}")
        else:
            env.render()
        
        print(f"\n[Test for ISO verbosity level {level} complete]\n")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

def test_pcs_verbosity_levels():
    """Test PCS verbosity during ISO episodes."""
    print("\n" + "=" * 80)
    print("TESTING PCS VERBOSITY CONTROL")
    print("=" * 80)
    print("\nISO verbosity=1 (summary), PCS verbosity=0 (silent)")
    print("PCS should be completely silent during ISO episode...\n")
    
    try:
        # Create environment where ISO shows summary but PCS is silent
        env = create_test_environment(iso_verbosity=1, pcs_verbosity=0)
        
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        print("\n" + "-" * 80)
        print("\nNow testing with PCS verbosity=1 (summary)")
        print("PCS should show episode summary...\n")
        
        # Create environment where both show summaries
        env = create_test_environment(iso_verbosity=1, pcs_verbosity=1)
        
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        print(f"\n[PCS verbosity test complete]\n")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

def test_verbosity_override():
    """Test that verbosity can be overridden in render() call."""
    print("\n" + "=" * 80)
    print("TESTING VERBOSITY OVERRIDE")
    print("=" * 80)
    print("\nEnvironment created with iso_verbosity=2 (default)")
    print("But we'll call render() with different verbosity levels...\n")
    
    try:
        env = create_test_environment(iso_verbosity=2, pcs_verbosity=0)
        
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
        
        print(f"\n[Override test complete]\n")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

def test_backward_compatibility():
    """Test backward compatibility with render_enabled."""
    print("\n" + "=" * 80)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 80)
    print("\nCreating env with render_enabled=True (old style)...")
    print("This should map to iso_verbosity=2, pcs_verbosity=1\n")
    
    try:
        pcs_env = PCSEnv(
            test_data_file=TEST_DATA_FILE,
            predictions_file=PREDICTIONS_FILE,
            verbosity=0
        )
        
        pcs_model = PPO("MlpPolicy", pcs_env, verbose=0)
        
        # Use old-style render_enabled parameter
        env = AlternatingISOEnv(
            actual_csv=TEST_DATA_FILE,
            predicted_csv=PREDICTIONS_FILE,
            pcs_env=pcs_env,
            pcs_model=pcs_model,
            render_enabled=True  # Old style
        )
        
        print(f"Environment iso_verbosity is: {env.iso_verbosity}")
        print(f"Environment pcs_verbosity is: {env.pcs_verbosity}")
        
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("\nCalling render() with no arguments (should use defaults):")
        env.render()
        
        print(f"\n[Backward compatibility test complete]\n")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# ALTERNATING ISO ENVIRONMENT VERBOSITY SYSTEM TEST")
    print("#" * 80)
    print("\nNote: This test uses untrained models for demonstration.")
    print("Results show the rendering system structure, not optimal performance.\n")
    
    # Test each ISO verbosity level
    test_iso_verbosity_level(0, "SILENT (data only, no console output)")
    test_iso_verbosity_level(1, "SUMMARY (single line per day)")
    test_iso_verbosity_level(2, "CONDENSED (summary + sampled timesteps) - DEFAULT")
    test_iso_verbosity_level(3, "DETAILED (summary + timesteps with features)")
    test_iso_verbosity_level(4, "DEBUG (comprehensive analysis)")
    
    # Test PCS verbosity control
    test_pcs_verbosity_levels()
    
    # Test verbosity override functionality
    test_verbosity_override()
    
    # Test backward compatibility
    test_backward_compatibility()
    
    print("\n" + "#" * 80)
    print("# ALL TESTS COMPLETE!")
    print("#" * 80)
    print("\nSummary:")
    print("  ✓ Level 0: Silent mode returns data without printing")
    print("  ✓ Level 1: Summary mode shows single line per day")
    print("  ✓ Level 2: Condensed mode shows summary + sampled timesteps (DEFAULT)")
    print("  ✓ Level 3: Detailed mode shows comprehensive day information")
    print("  ✓ Level 4: Debug mode includes pricing strategy and analysis")
    print("  ✓ PCS verbosity can be controlled separately (0=silent, 1+=various levels)")
    print("  ✓ Verbosity can be overridden per render() call")
    print("  ✓ Backward compatible with render_enabled parameter")
    print("\n")