# multi_agent_example.py
"""
Example demonstrating how to use the multi-agent training system.

This example shows how to set up and train multiple agents (ISOs and PCS units)
in custom sequences using the MultiAgentCoordinator.
"""

import os
from stable_baselines3 import PPO, A2C
from energy_net.gym_envs.multi_agent_env import AgentConfig, MultiAgentCoordinator, create_default_agents
from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.gym_envs.iso_env import ISOEnv


def example_simplest_default():
    """
    Example 0: Simplest possible setup - ZERO configuration!
    Just create the coordinator and train. That's it!
    """
    print("\n" + "="*70)
    print("EXAMPLE 0: Zero-Configuration Setup")
    print("="*70)
    
    # That's it! No arguments needed - creates 1 ISO + 1 PCS automatically
    coordinator = MultiAgentCoordinator(verbose=1)
    
    print(f"\nDefault setup created automatically!")
    print(f"Agents: {[a.agent_id for a in coordinator.agents]}")
    print(f"Training order: {coordinator.get_summary()['sequence_names']}")
    
    # Train for 2 iterations
    history = coordinator.train(total_iterations=2)
    
    print("\nTraining complete!")
    return coordinator, history

def example_customized_defaults():
    """
    Example 0b: Customize default agents without creating them manually.
    """
    print("\n" + "="*70)
    print("EXAMPLE 0b: Customized Defaults (Still Easy!)")
    print("="*70)
    
    # Create 1 ISO + 2 PCS agents with custom timesteps
    coordinator = MultiAgentCoordinator(
        num_pcs_agents=2,           # 2 PCS agents instead of 1
        iso_timesteps=5,            # ISO trains for 5 timesteps
        pcs_timesteps=24,           # Each PCS trains for 24 timesteps
        verbose=1
    )
    
    print(f"\nCustomized default setup:")
    print(f"Agents: {[a.agent_id for a in coordinator.agents]}")
    print(f"Training order: {coordinator.get_summary()['sequence_names']}")
    
    # Train for 2 iterations
    history = coordinator.train(total_iterations=2)
    
    print("\nTraining complete!")
    return coordinator, history

def example_basic_two_agent_alternating():
    """
    Example 1: Basic two-agent alternating training (similar to existing system).
    This replicates the classic ISO <-> PCS alternating pattern.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Two-Agent Alternating Training")
    print("="*70)
    
    # Get test data paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_data = os.path.join(project_root, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
    predictions = os.path.join(project_root, 'tests/gym/data_for_tests/consumption_predictions.csv')
    
    # Create environments
    pcs_env = PCSEnv(test_data_file=test_data, predictions_file=predictions, verbosity=0)
    iso_env = ISOEnv(actual_csv=test_data, predicted_csv=predictions)
    
    # Configure agents
    iso_agent = AgentConfig(
        agent_id="main_iso",
        agent_type="ISO",
        env=iso_env,
        algo_class=PPO,
        policy="MlpPolicy",
        timesteps_per_turn=7,  # Train for 7 days per cycle
        algo_kwargs={"n_steps": 7, "batch_size": 7}
    )
    
    pcs_agent = AgentConfig(
        agent_id="household_pcs",
        agent_type="PCS",
        env=pcs_env,
        algo_class=PPO,
        policy="MlpPolicy",
        timesteps_per_turn=48,  # Train for 48 half-hour steps per cycle
        algo_kwargs={"n_steps": 48, "batch_size": 48}
    )
    
    # Create coordinator with default sequence [0, 1] (ISO then PCS)
    coordinator = MultiAgentCoordinator(
        agents=[iso_agent, pcs_agent],
        verbose=1
    )
    
    print(f"\nCoordinator: {coordinator}")
    print(f"Training sequence: {coordinator.get_summary()['sequence_names']}")
    
    # Train for 3 iterations
    history = coordinator.train(total_iterations=3)
    
    print("\nTraining complete!")
    print(f"Summary: {coordinator.get_summary()}")
    
    return coordinator, history


def example_custom_sequence_three_agents():
    """
    Example 2: Three agents with custom training sequence.
    Demonstrates: ISO -> PCS1 -> PCS2 -> PCS1 -> ISO pattern
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Three Agents with Custom Sequence")
    print("="*70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_data = os.path.join(project_root, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
    predictions = os.path.join(project_root, 'tests/gym/data_for_tests/consumption_predictions.csv')
    
    # Create environments
    iso_env = ISOEnv(actual_csv=test_data, predicted_csv=predictions)
    pcs_env1 = PCSEnv(test_data_file=test_data, predictions_file=predictions, verbosity=0)
    pcs_env2 = PCSEnv(test_data_file=test_data, predictions_file=predictions, verbosity=0)
    
    # Configure three agents
    agents = [
        AgentConfig(
            agent_id="grid_iso",
            agent_type="ISO",
            env=iso_env,
            algo_class=PPO,
            timesteps_per_turn=7,
            algo_kwargs={"n_steps": 7, "batch_size": 7}
        ),
        AgentConfig(
            agent_id="household_1",
            agent_type="PCS",
            env=pcs_env1,
            algo_class=PPO,
            timesteps_per_turn=48,
            algo_kwargs={"n_steps": 48, "batch_size": 48}
        ),
        AgentConfig(
            agent_id="household_2",
            agent_type="PCS",
            env=pcs_env2,
            algo_class=A2C,  # Different algorithm
            timesteps_per_turn=48,
            algo_kwargs={"n_steps": 48}
        )
    ]
    
    # Custom sequence: ISO, PCS1, PCS2, PCS1, ISO
    # This gives PCS1 extra training time
    custom_sequence = [0, 1, 2, 1, 0]
    
    coordinator = MultiAgentCoordinator(
        agents=agents,
        training_sequence=custom_sequence,
        verbose=2  # More detailed output
    )
    
    print(f"\nCoordinator: {coordinator}")
    summary = coordinator.get_summary()
    print(f"\nTraining sequence: {summary['sequence_names']}")
    print(f"Sequence indices: {summary['training_sequence']}")
    
    # Train with callback to monitor progress
    def training_callback(metrics):
        print(f"  -> Iteration {metrics['iteration']} complete. Trained {len(metrics['agents_trained'])} agent steps.")
    
    history = coordinator.train(total_iterations=2, callback=training_callback)
    
    print("\nFinal summary:")
    for agent_info in summary['agents']:
        print(f"  {agent_info['id']}: {agent_info['algorithm']} ({agent_info['timesteps_per_turn']} steps/turn)")
    
    return coordinator, history


def example_different_timesteps():
    """
    Example 3: Agents with different training durations.
    Shows flexibility in timesteps_per_turn parameter.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Agents with Different Training Durations")
    print("="*70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_data = os.path.join(project_root, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
    predictions = os.path.join(project_root, 'tests/gym/data_for_tests/consumption_predictions.csv')
    
    iso_env = ISOEnv(actual_csv=test_data, predicted_csv=predictions)
    pcs_env = PCSEnv(test_data_file=test_data, predictions_file=predictions, verbosity=0)
    
    agents = [
        AgentConfig(
            agent_id="iso_short",
            agent_type="ISO",
            env=iso_env,
            algo_class=PPO,
            timesteps_per_turn=3,  # Short training cycles
            algo_kwargs={"n_steps": 3, "batch_size": 3}
        ),
        AgentConfig(
            agent_id="pcs_long",
            agent_type="PCS",
            env=pcs_env,
            algo_class=PPO,
            timesteps_per_turn=96,  # Longer training (2 days)
            algo_kwargs={"n_steps": 96, "batch_size": 96}
        )
    ]
    
    coordinator = MultiAgentCoordinator(agents=agents, verbose=1)
    
    print("\nAgent training durations:")
    for agent in agents:
        print(f"  {agent.agent_id}: {agent.timesteps_per_turn} timesteps per cycle")
    
    history = coordinator.train(total_iterations=2)
    
    return coordinator, history


def example_agent_lookup():
    """
    Example 4: Demonstrates agent lookup and access methods.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Agent Lookup and Access")
    print("="*70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_data = os.path.join(project_root, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
    predictions = os.path.join(project_root, 'tests/gym/data_for_tests/consumption_predictions.csv')
    
    iso_env = ISOEnv(actual_csv=test_data, predicted_csv=predictions)
    pcs_env1 = PCSEnv(test_data_file=test_data, predictions_file=predictions, verbosity=0)
    pcs_env2 = PCSEnv(test_data_file=test_data, predictions_file=predictions, verbosity=0)
    
    agents = [
        AgentConfig("iso_main", "ISO", iso_env, PPO, timesteps_per_turn=5, algo_kwargs={"n_steps": 5}),
        AgentConfig("pcs_alpha", "PCS", pcs_env1, PPO, timesteps_per_turn=24, algo_kwargs={"n_steps": 24}),
        AgentConfig("pcs_beta", "PCS", pcs_env2, A2C, timesteps_per_turn=24, algo_kwargs={"n_steps": 24})
    ]
    
    coordinator = MultiAgentCoordinator(agents=agents, verbose=0)
    
    # Lookup by ID
    print("\nLooking up agents by ID:")
    iso_agent = coordinator.get_agent_by_id("iso_main")
    print(f"  Found: {iso_agent}")
    
    pcs_alpha = coordinator.get_agent_by_id("pcs_alpha")
    print(f"  Found: {pcs_alpha}")
    
    # Lookup by index
    print("\nLooking up agents by index:")
    agent_0 = coordinator.get_agent_by_index(0)
    print(f"  Index 0: {agent_0.agent_id}")
    
    agent_2 = coordinator.get_agent_by_index(2)
    print(f"  Index 2: {agent_2.agent_id}")
    
    # Get summary
    print("\nCoordinator summary:")
    summary = coordinator.get_summary()
    print(f"  Total agents: {summary['num_agents']}")
    print(f"  Agents: {[a['id'] for a in summary['agents']]}")
    
    return coordinator


if __name__ == "__main__":
    """
    Run all examples to demonstrate the multi-agent system capabilities.
    """
    print("\n" + "#"*70)
    print("# MULTI-AGENT TRAINING SYSTEM EXAMPLES")
    print("#"*70)
    
    # Note: These examples are for demonstration.
    # In practice, you may want to run them separately to avoid resource issues.
    
    try:
        # Example 0: Zero-config
        print("\nRunning Example 0...")
        coord0, hist0 = example_simplest_default()
        
        # Example 0b: Customized defaults
        print("\nRunning Example 0b...")
        coord0b, hist0b = example_customized_defaults()
        
        # Example 1: Basic two-agent
        print("\nRunning Example 1...")
        coord1, hist1 = example_basic_two_agent_alternating()
        
        # Example 2: Three agents with custom sequence
        print("\nRunning Example 2...")
        coord2, hist2 = example_custom_sequence_three_agents()
        
        # Example 3: Different timesteps
        print("\nRunning Example 3...")
        coord3, hist3 = example_different_timesteps()
        
        # Example 4: Agent lookup
        print("\nRunning Example 4...")
        coord4 = example_agent_lookup()
        
        print("\n" + "#"*70)
        print("# ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("#"*70 + "\n")
        
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()