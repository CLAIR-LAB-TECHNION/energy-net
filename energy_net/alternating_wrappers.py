"""
Environment wrappers for alternating training of ISO and PCS agents.

These wrappers convert the multi-agent EnergyNetV0 environment into
single-agent environments suitable for training with RL Zoo. They handle
the sequential nature of the ISO-PCS interaction and maintain compatibility
with standard RL algorithms.
"""

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from energy_net.rewards.iso_reward import ISOReward
from energy_net.rewards.cost_reward import CostReward
import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alternating_wrappers")


class ISOEnvWrapper(gym.Wrapper):
    """
    Environment wrapper for ISO agent with fixed PCS policy.
    
    This wrapper converts the multi-agent EnergyNetV0 environment into a
    single-agent environment for training the ISO agent. It uses a fixed
    PCS policy to generate actions for the PCS agent.
    
    The wrapper ensures that the ISO agent receives properly formatted
    observations and rewards, and that the environment steps occur in the
    correct sequential order (ISO first, then PCS).
    """
    
    def __init__(self, env, pcs_policy=None):
        """
        Initialize the ISO environment wrapper.
        
        Args:
            env: The EnergyNetV0 environment to wrap
            pcs_policy: Optional fixed policy for the PCS agent
        """
        super().__init__(env)
        self.pcs_policy = pcs_policy
        
        # Use only ISO observation and action spaces
        self.observation_space = env.observation_space["iso"]
        self.action_space = env.action_space["iso"]
        
        # Store last observed state for PCS policy
        self.last_pcs_obs = None
        self.last_iso_action = None
        
        # Initialize the ISO reward calculator
        self.reward_calculator = ISOReward()
        
        # Set up logging
        self.logger = logger
        
    def reset(self, **kwargs):
        """
        Reset the environment and return the initial ISO observation.
        
        Returns:
            Initial observation for the ISO agent
            Info dictionary
        """
        obs_dict, info = self.env.reset(**kwargs)
        
        # Store PCS observation for future use
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Reset last ISO action
        self.last_iso_action = None
        
        return obs_dict["iso"], info
    
    def _unnormalize_pcs_action(self, normalized_action):
        """Convert PCS action from [-1, 1] to original space"""
        pcs_space = self.unwrapped.action_space["pcs"]
        low = pcs_space.low
        high = pcs_space.high
        
        # Standard linear rescaling from [-1, 1] to [low, high]
        unnormalized_action = low + (normalized_action + 1.0) * 0.5 * (high - low)
        
        # Add detailed INFO-level logging for PCS action rescaling
        if isinstance(normalized_action, np.ndarray) and len(normalized_action) > 0:
            self.logger.info(f"Rescaled PCS battery action from {normalized_action[0]:.4f} to {unnormalized_action[0]:.4f} [range: {low[0]:.1f}-{high[0]:.1f}]")
        else:
            self.logger.info(f"Rescaled PCS action from {normalized_action:.4f} to {unnormalized_action:.4f} [range: {low:.1f}-{high:.1f}]")
            
        self.logger.debug(f"Unnormalized PCS action from {normalized_action} to {unnormalized_action}")
        return unnormalized_action
    
    def step(self, action):
        """
        Execute ISO action and automatically handle PCS action.
        
        This method:
        1. Stores the ISO action
        2. Processes the ISO action to update prices
        3. Gets the updated PCS observation with new prices
        4. Gets PCS action from the fixed policy
        5. Steps the environment with both actions
        6. Returns ISO-specific results
        
        Args:
            action: Action from the ISO agent
            
        Returns:
            ISO observation, reward, terminated flag, truncated flag, info dict
        """
        # Debug log the incoming action
        self.logger.debug(f"ISOEnvWrapper received action from ISO agent: {action}")
        
        # Store ISO action
        self.last_iso_action = action
        
        # Access controller and process ISO action if possible
        # This updates prices before PCS observes them
        if hasattr(self.unwrapped, "controller"):
            controller = self.unwrapped.controller
            
            # Set the ISO prices based on the action
            self.logger.debug(f"ISOEnvWrapper passing action to controller: {action}")
            controller._process_iso_action(action)
            
            # Get updated PCS observation with new prices
            pcs_obs = controller._get_pcs_observation()
            self.last_pcs_obs = pcs_obs
        
        # Get PCS action from policy or use default action
        if self.pcs_policy is not None:
            # Convert to batch format for policy prediction
            pcs_obs_batch = np.array([self.last_pcs_obs])
            
            # Get action from policy - will be in [-1, 1] range from tanh activation
            pcs_action, _ = self.pcs_policy.predict(
                pcs_obs_batch, 
                deterministic=True
            )
            
            # Extract from batch
            pcs_action = pcs_action[0]
            
            # Convert from [-1, 1] to PCS action space
            pcs_action = self._unnormalize_pcs_action(pcs_action)
            
            self.logger.debug(f"ISOEnvWrapper got PCS action from policy: {pcs_action}")
        else:
            # Default action (neutral battery action)
            pcs_action = np.zeros(self.unwrapped.action_space["pcs"].shape)
            self.logger.debug(f"ISOEnvWrapper using default PCS action: {pcs_action}")
        
        # Create joint action dict - ISO must go first!
        action_dict = {
            "iso": action,
            "pcs": pcs_action
        }
        
        self.logger.debug(f"ISOEnvWrapper stepping environment with action_dict: {action_dict}")
        
        # Step the environment
        obs_dict, rewards, terminations, truncations, info = self.env.step(action_dict)
        
        # Store updated PCS observation
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Calculate ISO reward using the ISOReward class
        # First, extract ISO-specific metrics from the info dictionary
        iso_info = info.get('iso', {})
        
        # Make sure required keys exist in iso_info for reward calculation
        if 'shortfalls' in iso_info and iso_info['shortfalls']:
            iso_info['shortfall'] = iso_info['shortfalls'][-1]
        if 'reserve_costs' in iso_info and iso_info['reserve_costs']:
            iso_info['reserve_cost'] = iso_info['reserve_costs'][-1]
        if 'dispatch_costs' in iso_info and iso_info['dispatch_costs']:
            iso_info['dispatch_cost'] = iso_info['dispatch_costs'][-1]
        if 'pcs_demands' in iso_info and iso_info['pcs_demands']:
            iso_info['pcs_demand'] = iso_info['pcs_demands'][-1]
        if 'buy_prices' in iso_info and iso_info['buy_prices']:
            iso_info['iso_buy_price'] = iso_info['buy_prices'][-1]
        if 'sell_prices' in iso_info and iso_info['sell_prices']:
            iso_info['iso_sell_price'] = iso_info['sell_prices'][-1]
        
        # Calculate the reward
        custom_reward = self.reward_calculator.compute_reward(iso_info)
        
        self.logger.debug(f"ISOEnvWrapper returning custom reward: {custom_reward}")
        
        # Return only ISO related outputs with the custom reward
        return (
            obs_dict["iso"],
            custom_reward,
            terminations["iso"],
            truncations["iso"],
            info
        )


class PCSEnvWrapper(gym.Wrapper):
    """
    Environment wrapper for PCS agent with fixed ISO policy.
    
    This wrapper converts the multi-agent EnergyNetV0 environment into a
    single-agent environment for training the PCS agent. It uses a fixed
    ISO policy to generate actions for the ISO agent.
    
    The wrapper ensures that the PCS agent receives properly formatted
    observations and rewards, and that the environment steps occur in the
    correct sequential order (ISO first, then PCS).
    """
    
    def __init__(self, env, iso_policy=None):
        """
        Initialize the PCS environment wrapper.
        
        Args:
            env: The EnergyNetV0 environment to wrap
            iso_policy: Optional fixed policy for the ISO agent
        """
        super().__init__(env)
        self.iso_policy = iso_policy
        
        # Use only PCS observation and action spaces
        self.observation_space = env.observation_space["pcs"]
        self.action_space = env.action_space["pcs"]
        
        # Store last observed state for ISO policy
        self.last_iso_obs = None
        self.last_pcs_obs = None
        
        # Initialize the PCS reward calculator
        self.reward_calculator = CostReward()
        
        # Set up logging
        self.logger = logger
        
    def reset(self, **kwargs):
        """
        Reset the environment and return the initial PCS observation.
        
        Returns:
            Initial observation for the PCS agent
            Info dictionary
        """
        obs_dict, info = self.env.reset(**kwargs)
        
        # Store observations for future use
        self.last_iso_obs = obs_dict["iso"]
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Reset the reward calculator
        self.reward_calculator.reset()
        
        return obs_dict["pcs"], info
    
    def _unnormalize_iso_action(self, normalized_action):
        """Convert ISO action from [-1, 1] to original space"""
        iso_space = self.unwrapped.action_space["iso"]
        low = iso_space.low
        high = iso_space.high
        
        # Get pricing policy and dispatch flag directly
        pricing_policy = None
        use_dispatch = False
        
        if hasattr(self.unwrapped, "controller"):
            controller = self.unwrapped.controller
            if hasattr(controller, "pricing_policy"):
                pricing_policy = controller.pricing_policy
            if hasattr(controller, "use_dispatch_action"):
                use_dispatch = controller.use_dispatch_action
        
        # Standard linear rescaling from [-1, 1] to [low, high]
        unnormalized_action = low + (normalized_action + 1.0) * 0.5 * (high - low)
        
        # Log the rescaled actions with their respective ranges
        if isinstance(normalized_action, np.ndarray) and len(normalized_action) > 0:
            policy_name = pricing_policy.value if pricing_policy and hasattr(pricing_policy, "value") else "Unknown"
            
            if policy_name == "CONSTANT" or policy_name == "ONLINE":
                if len(unnormalized_action) >= 1:
                    self.logger.info(f"Rescaled {policy_name} buy_price from {normalized_action[0]:.4f} to {unnormalized_action[0]:.4f} [range: {low[0]:.1f}-{high[0]:.1f}]")
                if len(unnormalized_action) >= 2:
                    self.logger.info(f"Rescaled {policy_name} sell_price from {normalized_action[1]:.4f} to {unnormalized_action[1]:.4f} [range: {low[1]:.1f}-{high[1]:.1f}]")
                if use_dispatch and len(unnormalized_action) >= 3:
                    self.logger.info(f"Rescaled {policy_name} dispatch from {normalized_action[2]:.4f} to {unnormalized_action[2]:.4f} [range: {low[2]:.1f}-{high[2]:.1f}]")
            
            elif policy_name == "QUADRATIC":
                if len(unnormalized_action) >= 6:
                    self.logger.info(f"Rescaled QUADRATIC buy coef from {normalized_action[0:3]} to {unnormalized_action[0:3]} [range: {low[0:3]}-{high[0:3]}]")
                    self.logger.info(f"Rescaled QUADRATIC sell coef from {normalized_action[3:6]} to {unnormalized_action[3:6]} [range: {low[3:6]}-{high[3:6]}]")
                if use_dispatch and len(unnormalized_action) >= 7:
                    self.logger.info(f"Rescaled QUADRATIC dispatch from {normalized_action[6]:.4f} to {unnormalized_action[6]:.4f} [range: {low[6]:.1f}-{high[6]:.1f}]")
            else:
                for i in range(len(unnormalized_action)):
                    if i < len(low) and i < len(high):
                        self.logger.info(f"Rescaled action[{i}] from {normalized_action[i]:.4f} to {unnormalized_action[i]:.4f} [range: {low[i]:.1f}-{high[i]:.1f}]")
        else:
            # Single scalar action
            self.logger.info(f"Rescaled scalar action from {normalized_action:.4f} to {unnormalized_action:.4f} [range: {low:.1f}-{high:.1f}]")
        
        return unnormalized_action
    
    def step(self, action):
        """
        Execute PCS action with prior ISO action from policy.
        
        This method:
        1. Gets ISO action from the fixed policy
        2. Creates an action dictionary with both actions
        3. Steps the environment with the action dictionary
        4. Returns PCS-specific results
        
        Args:
            action: Action from the PCS agent
            
        Returns:
            PCS observation, reward, terminated flag, truncated flag, info dict
        """
        # Debug log the incoming action
        self.logger.debug(f"PCSEnvWrapper received action from PCS agent: {action}")
        
        # Get ISO action from policy or use default action
        if self.iso_policy is not None:
            # Convert to batch format for policy prediction
            iso_obs_batch = np.array([self.last_iso_obs])
            
            # Get action from policy - will be in [-1, 1] range from tanh activation
            iso_action, _ = self.iso_policy.predict(
                iso_obs_batch, 
                deterministic=True
            )
            
            # Extract from batch
            iso_action = iso_action[0]
            
            # Convert from [-1, 1] to ISO action space
            iso_action = self._unnormalize_iso_action(iso_action)
            
            self.logger.debug(f"PCSEnvWrapper got ISO action from policy: {iso_action}")
        else:
            # Default action (mid-range price)
            iso_action = np.zeros(self.unwrapped.action_space["iso"].shape)
            
            # Set a reasonable default dispatch value if needed
            if len(iso_action) > 2 and hasattr(self.unwrapped, "controller") and hasattr(self.unwrapped.controller, "use_dispatch_action") and self.unwrapped.controller.use_dispatch_action:
                if hasattr(self.unwrapped.controller, "predicted_demand"):
                    # Default to predicted demand as a reasonable value (this is NOT rescaling, just a default)
                    iso_action[2] = 0.0  # Neutral value that will be properly scaled
                
            self.logger.debug(f"PCSEnvWrapper using default ISO action: {iso_action}")
        
        # Process ISO action to set prices if possible
        if hasattr(self.unwrapped, "controller"):
            controller = self.unwrapped.controller
            self.logger.debug(f"PCSEnvWrapper passing ISO action to controller: {iso_action}")
            controller._process_iso_action(iso_action)
        
        # Create joint action dict - ISO must go first!
        action_dict = {
            "iso": iso_action,
            "pcs": action
        }
        
        self.logger.debug(f"PCSEnvWrapper stepping environment with action_dict: {action_dict}")
        
        # Step the environment
        obs_dict, rewards, terminations, truncations, info = self.env.step(action_dict)
        
        # Store updated observations
        self.last_iso_obs = obs_dict["iso"]
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Calculate PCS reward using the CostReward class
        # First, extract relevant information for reward calculation
        pcs_info = {}
        
        # Extract PCS energy exchange
        if 'pcs' in info and 'energy_exchanges' in info['pcs'] and info['pcs']['energy_exchanges']:
            pcs_info['net_exchange'] = info['pcs']['energy_exchanges'][-1]
        else:
            pcs_info['net_exchange'] = 0.0
            
        # Extract ISO prices
        if 'iso' in info:
            if 'buy_prices' in info['iso'] and info['iso']['buy_prices']:
                pcs_info['iso_buy_price'] = info['iso']['buy_prices'][-1]
            else:
                pcs_info['iso_buy_price'] = 0.0
                
            if 'sell_prices' in info['iso'] and info['iso']['sell_prices']:
                pcs_info['iso_sell_price'] = info['iso']['sell_prices'][-1]
            else:
                pcs_info['iso_sell_price'] = 0.0
        
        # Calculate the PCS reward
        custom_reward = self.reward_calculator.compute_reward(pcs_info)
        
        self.logger.debug(f"PCSEnvWrapper returning custom reward: {custom_reward}")
        
        # Return only PCS related outputs with the custom reward
        return (
            obs_dict["pcs"],
            custom_reward,
            terminations["pcs"],
            truncations["pcs"],
            info
        )


# Factory functions to create wrapped environments
def make_iso_env(
    steps_per_iteration=1000,
    cost_type="CONSTANT",
    pricing_policy="ONLINE",
    demand_pattern="SINUSOIDAL",
    seed=None,
    log_dir="logs",
    model_dir="saved_models",
    plot_dir="plots",
    pcs_policy=None,
    norm_path=None,
    use_dispatch_action=False,
    dispatch_strategy="PROPORTIONAL"
):
    """
    Create a wrapped environment for ISO training.
    
    Args:
        steps_per_iteration: Number of timesteps per training iteration
        cost_type: Type of cost model
        pricing_policy: Price setting policy
        demand_pattern: Pattern of demand
        seed: Random seed
        log_dir: Directory for logs
        model_dir: Directory for saved models
        plot_dir: Directory for plots
        pcs_policy: Optional fixed policy for PCS agent
        norm_path: Path to normalization file for consistent normalization
        use_dispatch_action: Whether ISO should output a dispatch action
        dispatch_strategy: Strategy for dispatch when not controlled by agent
        
    Returns:
        Wrapped environment ready for ISO training
    """
    from energy_net.env import EnergyNetV0
    
    # Create the base environment
    base_env = EnergyNetV0(
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        cost_type=cost_type,
        dispatch_config={
            "use_dispatch_action": use_dispatch_action,
            "default_strategy": dispatch_strategy
        }
    )
    
    # First wrap with ISO wrapper
    env = ISOEnvWrapper(base_env, pcs_policy)
    
    # Create monitor directory if it doesn't exist
    monitor_dir = os.path.join(log_dir, "iso_monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    
    # Add monitoring wrapper
    env = Monitor(env, monitor_dir, allow_early_resets=True)
    
    # Add action scaling wrapper - AFTER the wrapper and monitor
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    
    # Log the action spaces to help diagnose scaling issues
    logger.info(f"ISO wrapped action space: {env.action_space}")
    logger.info(f"Original ISO action space: {base_env.action_space['iso']}")
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: env])
    
    # Add normalization - if norm_path is provided, load from it
    if norm_path and os.path.exists(norm_path):
        print(f"Loading ISO normalization from: {norm_path}")
        env = VecNormalize.load(norm_path, env)
        # Just update stats during training
        env.training = True
        env.norm_reward = True
    else:
        print(f"Creating new ISO normalization")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=1.,
            clip_reward=1.,
            gamma=0.99,
            epsilon=1e-8
        )
    
    return env


def make_pcs_env(
    steps_per_iteration=1000,
    cost_type="CONSTANT",
    pricing_policy="ONLINE",
    demand_pattern="SINUSOIDAL",
    seed=None,
    log_dir="logs",
    model_dir="saved_models",
    plot_dir="plots",
    iso_policy=None,
    norm_path=None,
    use_dispatch_action=False,
    dispatch_strategy="PROPORTIONAL"
):
    """
    Create a wrapped environment for PCS training.
    
    Args:
        steps_per_iteration: Number of timesteps per training iteration
        cost_type: Type of cost model
        pricing_policy: Price setting policy
        demand_pattern: Pattern of demand
        seed: Random seed
        log_dir: Directory for logs
        model_dir: Directory for saved models
        plot_dir: Directory for plots
        iso_policy: Optional fixed policy for ISO agent
        norm_path: Path to normalization file for consistent normalization
        use_dispatch_action: Whether ISO should output a dispatch action
        dispatch_strategy: Strategy for dispatch when not controlled by agent
        
    Returns:
        Wrapped environment ready for PCS training
    """
    from energy_net.env import EnergyNetV0
    
    # Create the base environment
    base_env = EnergyNetV0(
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        cost_type=cost_type,
        dispatch_config={
            "use_dispatch_action": use_dispatch_action,
            "default_strategy": dispatch_strategy
        }
    )
    
    # First wrap with PCS wrapper
    env = PCSEnvWrapper(base_env, iso_policy)
    
    # Create monitor directory if it doesn't exist
    monitor_dir = os.path.join(log_dir, "pcs_monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    
    # Add monitoring wrapper
    env = Monitor(env, monitor_dir, allow_early_resets=True)
    
    # Add action scaling wrapper - AFTER the wrapper and monitor
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    
    # Log the action spaces to help diagnose scaling issues
    logger.info(f"PCS wrapped action space: {env.action_space}")
    logger.info(f"Original PCS action space: {base_env.action_space['pcs']}")
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: env])
    
    # Add normalization - if norm_path is provided, load from it
    if norm_path and os.path.exists(norm_path):
        print(f"Loading PCS normalization from: {norm_path}")
        env = VecNormalize.load(norm_path, env)
        # Just update stats during training
        env.training = True
        env.norm_reward = True
    else:
        print(f"Creating new PCS normalization")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=1.,
            clip_reward=1.,
            gamma=0.99,
            epsilon=1e-8
        )
    
    return env 