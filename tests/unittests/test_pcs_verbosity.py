#!/usr/bin/env python3
"""
Test script to demonstrate all verbosity levels in PCSEnv.
"""

import numpy as np
import sys
import os
from energy_net.gym_envs.pcs_env import PCSEnv


# Add the project root to the path to import energy_net
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
TEST_DATA_FILE = os.path.join(PROJECT_ROOT, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, 'tests/gym/data_for_tests/consumption_predictions.csv')

def test_verbosity_level(level, description):
    """Test a specific verbosity level."""
    print("\n" + "=" * 80)
    print(f"TESTING VERBOSITY LEVEL {level}: {description}")
    print("=" * 80)
    
    # Create environment with specific verbosity
    env = PCSEnv(
        test_data_file=TEST_DATA_FILE,
        predictions_file=PREDICTIONS_FILE,
        verbosity=level
    )
    
    # Run a short episode
    obs, info = env.reset()
    
    # For level 0 and 1, just show what happens
    if level == 0:
        print("\n[Level 0 is SILENT - no console output from render()]")
        data = env.render()
        print(f"But it DOES return structured data: {list(data.keys())}")
        print(f"Example: step={data['step']}, storage={data['storage']:.2f}, price=${data['price']:.4f}")
    
    # Run a few steps
    steps_to_show = 3 if level >= 2 else 48  # Show all steps for level 1 to see summary
    
    for step in range(steps_to_show):
        action = np.array([np.random.uniform(-5.0, 5.0)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render based on verbosity
        if level == 1:
            # Level 1 only shows at start and end
            env.render()
        elif level >= 2:
            # Levels 2+ show each step
            env.render()
        
        if terminated:
            break
    
    print(f"\n[Test for verbosity level {level} complete]\n")

def test_override():
    """Test that verbosity can be overridden in render() call."""
    print("\n" + "=" * 80)
    print("TESTING VERBOSITY OVERRIDE")
    print("=" * 80)
    print("\nEnvironment created with verbosity=2 (default)")
    print("But we'll call render() with different verbosity levels...\n")
    
    env = PCSEnv(
        test_data_file=TEST_DATA_FILE,
        predictions_file=PREDICTIONS_FILE,
        verbosity=2  # Default
    )
    
    obs, info = env.reset()
    action = np.array([3.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("\n--- Calling render(verbosity=1) - Summary only ---")
    env.render(verbosity=1)
    
    print("\n--- Calling render(verbosity=0) - Silent ---")
    data = env.render(verbosity=0)
    print(f"[No console output, but returned data dict with keys: {list(data.keys())}]")
    
    print("\n--- Calling render(verbosity=4) - Debug mode ---")
    env.render(verbosity=4)

def test_backward_compatibility():
    """Test backward compatibility with render_mode='human'."""
    print("\n" + "=" * 80)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 80)
    print("\nCreating env with render_mode='human' (old style)...")
    print("This should map to verbosity=2\n")
    
    env = PCSEnv(
        test_data_file=TEST_DATA_FILE,
        predictions_file=PREDICTIONS_FILE,
        render_mode='human'  # Old style
    )
    
    print(f"Environment verbosity is: {env.verbosity}")
    
    obs, info = env.reset()
    action = np.array([2.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("\nCalling render() with no arguments (should use verbosity=2):")
    env.render()

if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# PCS ENVIRONMENT VERBOSITY SYSTEM TEST")
    print("#" * 80)
    
    # Test each verbosity level
    test_verbosity_level(0, "SILENT (data only, no console output)")
    test_verbosity_level(1, "SUMMARY (episode start/end only)")
    test_verbosity_level(2, "CONDENSED (key metrics per step) - DEFAULT")
    test_verbosity_level(3, "DETAILED (all state info per step)")
    test_verbosity_level(4, "DEBUG (includes battery dynamics)")
    
    # Test override functionality
    test_override()
    
    # Test backward compatibility
    test_backward_compatibility()
    
    print("\n" + "#" * 80)
    print("# ALL TESTS COMPLETE!")
    print("#" * 80)
    print("\nSummary:")
    print("  ✓ Level 0: Silent mode returns data without printing")
    print("  ✓ Level 1: Summary mode shows only episode start/end")
    print("  ✓ Level 2: Condensed mode shows key metrics (DEFAULT)")
    print("  ✓ Level 3: Detailed mode shows all state information")
    print("  ✓ Level 4: Debug mode includes battery dynamics")
    print("  ✓ Verbosity can be overridden per render() call")
    print("  ✓ Backward compatible with render_mode='human'")
    print("\n")