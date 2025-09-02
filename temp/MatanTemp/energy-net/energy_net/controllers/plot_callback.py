#!/usr/bin/env python
"""
Direct callback module for RL-Zoo3
"""
from energy_net.controllers.callbacks import ActionTrackingCallback
import os

class PlotCallback(ActionTrackingCallback):
    """
    Direct wrapper around ActionTrackingCallback to make it importable by RL-Zoo3
    """
    def __init__(self, verbose=1):
        # Determine agent type based on environment name
        # This will be set properly when called from RL-Zoo3 via setup_callback
        super().__init__(
            agent_name="agent",  # Temporary value, will be overridden
            verbose=verbose,
            is_training=True,
            save_path="logs/plots"  # Temporary value, will be overridden
        )
    
    def setup_callback(self, env_id, **kwargs):
        """Set up the callback for the given environment"""
        # Extract agent type from environment ID
        agent_type = "iso" if "ISO" in env_id else "pcs"
        
        # Read iteration from file (created by the training script)
        iteration = 1
        iteration_file = "temp/current_iteration.txt"
        if os.path.exists(iteration_file):
            try:
                with open(iteration_file, "r") as f:
                    iteration = int(f.read().strip())
            except (ValueError, IOError) as e:
                print(f"Warning: Could not read iteration from file: {e}")
                print("Using default iteration = 1")
        
        save_path = f"logs/plots/{agent_type}/iteration_{iteration}"
        
        # Update attributes
        self.agent_name = agent_type
        self.save_path = save_path
        
        print(f"Setting up ActionTrackingCallback for {agent_type} agent (iteration {iteration})")
        print(f"Plots will be saved to: {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        return self 