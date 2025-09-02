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
from typing import Dict, Any, Tuple, Optional, Union, List, Sequence
import os
import logging
from energy_net.envs import EnergyNetV0
# Convert enum parameters from string form if needed
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
from energy_net.market.pricing.cost_types import CostType
from energy_net.envs import EnergyNetV0


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
    
    def __init__(self, env, pcs_policy=None, eval_mode=False):
        """
        Initialize the ISO environment wrapper.
        
        Args:
            env: The EnergyNetV0 environment to wrap
            pcs_policy: Optional fixed policy for the PCS agent
            eval_mode: Whether the wrapper is in evaluation mode
        """
        super().__init__(env)
        self.pcs_policy = pcs_policy
        self.eval_mode = eval_mode
        
        # Use only ISO observation and action spaces
        self.observation_space = env.observation_space["iso"]
        self.action_space = env.action_space["iso"]
        
        # Store last observed state for PCS policy
        self.last_pcs_obs = None
        self.last_iso_action = None
        
        # Set up logging
        self.logger = logger
        
        # Epsilon for epsilon-greedy exploration (only used during training)
        self.epsilon = 0.0 if not eval_mode else 0.0
        
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
        
        # (Removed manual controller._process_iso_action to avoid double processing)

        # Get PCS action from policy or use default action
        if self.pcs_policy is not None and self.last_pcs_obs is not None:
            try:
                # Convert to batch format for policy prediction
                pcs_obs_batch = np.array([self.last_pcs_obs])
                
                # Get policy output (normalized in [-1,1])
                raw_action, _ = self.pcs_policy.predict(
                    pcs_obs_batch,
                    deterministic=self.eval_mode  # Deterministic in eval mode
                )
                norm_action = raw_action[0]
                
                # Epsilon-greedy: with prob epsilon, take a random normalized action
                if not self.eval_mode and np.random.rand() < self.epsilon:
                    self.logger.debug(f"Epsilon-greedy: randomizing PCS action (eps={self.epsilon})")
                    norm_action = np.random.uniform(-1.0, 1.0, size=norm_action.shape)
                # Unnormalize to original space
                pcs_action = self._unnormalize_pcs_action(norm_action)
                
                self.logger.debug(f"ISOEnvWrapper got PCS action from policy: {pcs_action}")
            except Exception as e:
                # Fallback to default action if policy prediction fails
                self.logger.warning(f"PCS policy prediction failed: {e}, using default action")
                pcs_action = np.zeros(self.unwrapped.action_space["pcs"].shape)
        else:
            # Default action (neutral battery action)
            pcs_action = np.zeros(self.unwrapped.action_space["pcs"].shape)
            if self.pcs_policy is None:
                self.logger.debug(f"ISOEnvWrapper using default PCS action (NO POLICY): {pcs_action}")
            else:
                self.logger.warning(f"ISOEnvWrapper using default PCS action (INVALID OBS): obs={self.last_pcs_obs}")
        
        # Create joint action dict - ISO must go first!
        action_dict = {
            "iso": action,
            "pcs": pcs_action
        }
        
        self.logger.debug(f"ISOEnvWrapper stepping environment with action_dict: {action_dict}")
        
        # Step the environment and get native rewards
        obs_dict, rewards, terminations, truncations, info = self.env.step(action_dict)
        
        # Store updated PCS observation
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Use the environment-provided ISO reward
        if isinstance(rewards, dict):
            iso_reward = rewards.get('iso', 0.0)
        else:
            # tuple or list
            iso_reward = rewards[0] if len(rewards) > 0 else 0.0
        # ALSO extract PCS reward for logging
        if isinstance(rewards, dict):
            pcs_reward = rewards.get('pcs', 0.0)
        else:
            pcs_reward = rewards[1] if len(rewards) > 1 else 0.0
        # Inject rewards and PCS action into info for Monitor and plotting
        if info is None:
            info = {}
        info['iso_reward'] = iso_reward
        info['pcs_reward'] = pcs_reward
        info['pcs_action'] = pcs_action  # Add PCS action for evaluation plotting
        info.setdefault('actions', {})['pcs'] = pcs_action

        self.logger.debug(f"ISOEnvWrapper returning env ISO reward: {iso_reward}")
        # Return ISO obs, native ISO reward, done/trunc flags, and info
        return (
            obs_dict["iso"],
            iso_reward,
            terminations.get("iso", False),
            truncations.get("iso", False),
            info
        )


class PreDefinedPCSWrapper(ISOEnvWrapper):
    """
    Environment wrapper for ISO agent with a predefined sequence of PCS actions.
    
    This wrapper converts the multi-agent EnergyNetV0 environment into a
    single-agent environment for training the ISO agent, where the PCS agent
    follows a predetermined sequence of actions instead of using a policy.
    
    The wrapper consumes actions from the sequence on each step, and cycles
    through the sequence if the episode length exceeds the sequence length.
    """
    
    def __init__(self, env, pcs_action_sequence=None):
        """
        Initialize the wrapper with a predefined sequence of PCS actions.
        
        Args:
            env: The EnergyNetV0 environment to wrap
            pcs_action_sequence: Sequence of PCS actions to use, should be in the PCS action space
                                 If None, will default to zero actions
        """
        # Initialize with pcs_policy=None from parent class
        super().__init__(env, pcs_policy=None)
        
        # Store the predefined PCS action sequence
        self.pcs_action_sequence = pcs_action_sequence
        
        # Validate and process the action sequence
        self._process_action_sequence()
        
        # Current position in the sequence
        self.current_sequence_idx = 0
        
        self.logger.info(f"Initialized PreDefinedPCSWrapper with {len(self.pcs_action_sequence)} actions")
    
    def _process_action_sequence(self):
        """Process and validate the PCS action sequence"""
        pcs_space = self.unwrapped.action_space["pcs"]
        
        # If no sequence is provided, create a sequence of zero actions
        if self.pcs_action_sequence is None:
            # Create a single zero action that will be repeated
            self.pcs_action_sequence = [np.zeros(pcs_space.shape)]
            self.logger.warning("No PCS action sequence provided, using zero actions")
            return
        
        # Convert to numpy array if it's not already
        if not isinstance(self.pcs_action_sequence, np.ndarray):
            try:
                self.pcs_action_sequence = np.array(self.pcs_action_sequence)
            except:
                self.logger.error("Failed to convert PCS action sequence to numpy array")
                # Fallback to a sequence of zero actions
                self.pcs_action_sequence = [np.zeros(pcs_space.shape)]
                return
        
        # Check if the sequence is 1D with a single action value
        if len(self.pcs_action_sequence.shape) == 1:
            # Convert to 2D array of shape (T, 1) where T is sequence length
            self.pcs_action_sequence = self.pcs_action_sequence.reshape(-1, 1)
        
        # Ensure each action in the sequence is in the correct space
        for i, action in enumerate(self.pcs_action_sequence):
            # Check if action is within bounds
            if not pcs_space.contains(action):
                self.logger.warning(f"Action at index {i} with value {action} is outside the PCS action space {pcs_space.low}-{pcs_space.high}")
                # Clip to valid range
                self.pcs_action_sequence[i] = np.clip(action, pcs_space.low, pcs_space.high)
                self.logger.warning(f"Clipped to {self.pcs_action_sequence[i]}")
    
    def reset(self, **kwargs):
        """
        Reset the environment and return the initial ISO observation.
        
        Also resets the action sequence index.
        
        Returns:
            Initial observation for the ISO agent
            Info dictionary
        """
        # Reset the sequence index
        self.current_sequence_idx = 0
        
        # Call parent reset
        return super().reset(**kwargs)
    
    def step(self, action):
        """
        Execute ISO action and use the next predefined PCS action.
        
        This method:
        1. Takes the ISO action
        2. Gets the next PCS action from the predefined sequence
        3. Steps the environment with both actions
        4. Advances the sequence index
        5. Returns ISO-specific results
        
        Args:
            action: Action from the ISO agent
            
        Returns:
            ISO observation, reward, terminated flag, truncated flag, info dict
        """
        # Store ISO action from parent class pattern
        self.last_iso_action = action
        
        # Get the next PCS action from the sequence
        if len(self.pcs_action_sequence) > 0:
            # Get current action based on index and use modulo to cycle if needed
            pcs_action = self.pcs_action_sequence[self.current_sequence_idx % len(self.pcs_action_sequence)]
            
            self.logger.debug(f"Using predefined PCS action at index {self.current_sequence_idx}: {pcs_action}")
            
            # Advance the sequence index
            self.current_sequence_idx += 1
        else:
            # Fallback to zero action if sequence is empty
            pcs_action = np.zeros(self.unwrapped.action_space["pcs"].shape)
            self.logger.warning("PCS action sequence is empty, using zero action")
        
        # Create joint action dict - ISO must go first!
        action_dict = {
            "iso": action,
            "pcs": pcs_action
        }
        
        self.logger.debug(f"PreDefinedPCSWrapper stepping environment with action_dict: {action_dict}")
        
        # Step the environment
        obs_dict, rewards, terminations, truncations, info = self.env.step(action_dict)
        
        # Store updated PCS observation for consistency with parent class
        self.last_pcs_obs = obs_dict["pcs"]
        
        # Use the environment-provided ISO reward
        if isinstance(rewards, dict):
            iso_reward = rewards.get('iso', 0.0)
        else:
            # tuple or list
            iso_reward = rewards[0] if len(rewards) > 0 else 0.0
        # ALSO extract PCS reward
        if isinstance(rewards, dict):
            pcs_reward = rewards.get('pcs', 0.0)
        else:
            pcs_reward = rewards[1] if len(rewards) > 1 else 0.0
        # Add sequence idx, rewards and PCS action to info
        if info is None:
            info = {}
        info['pcs_action_sequence_idx'] = self.current_sequence_idx - 1
        info['iso_reward'] = iso_reward
        info['pcs_reward'] = pcs_reward
        info['pcs_action'] = pcs_action  # Add PCS action for evaluation plotting
        
        self.logger.debug(f"PreDefinedPCSWrapper returning ISO reward: {iso_reward}")
        
        # Return ISO-specific results
        return (
            obs_dict["iso"],
            iso_reward,
            terminations.get("iso", False),
            truncations.get("iso", False),
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
    
    def __init__(self, env, iso_policy=None, eval_mode=False, iso_deterministic=False):
        """
        Initialize the PCS environment wrapper.
        
        Args:
            env: The EnergyNetV0 environment to wrap
            iso_policy: Optional fixed policy for the ISO agent
            eval_mode: Whether the wrapper is in evaluation mode
            iso_deterministic: Whether to use deterministic ISO actions (useful when ISO is opponent)
        """
        super().__init__(env)
        self.iso_policy = iso_policy
        self.eval_mode = eval_mode
        self.iso_deterministic = iso_deterministic
        
        # Use only PCS observation and action spaces
        self.observation_space = env.observation_space["pcs"]
        self.action_space = env.action_space["pcs"]
        
        # Store last observed state for ISO policy
        self.last_iso_obs = None
        self.last_pcs_obs = None
        
        # Set up logging
        self.logger = logger
        
        # Epsilon for epsilon-greedy exploration (only used during training)
        self.epsilon = 0.0 if not eval_mode else 0.0
        
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
            
            # Get policy output (normalized) and apply epsilon-greedy
            raw_action, _ = self.iso_policy.predict(
                iso_obs_batch,
                deterministic=self.eval_mode or self.iso_deterministic  # Deterministic in eval mode or when iso_deterministic is True
            )
            norm_action = raw_action[0]
            if not self.eval_mode and np.random.rand() < self.epsilon:
                self.logger.debug(f"Epsilon-greedy: randomizing ISO action (eps={self.epsilon})")
                norm_action = np.random.uniform(-1.0, 1.0, size=norm_action.shape)
            iso_action = self._unnormalize_iso_action(norm_action)
            
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
        
        # (Removed manual controller._process_iso_action to avoid double processing)
        
        # Create joint action dict - ISO must go first!
        action_dict = {
            "iso": iso_action,
            "pcs": action
        }
        
        self.logger.debug(f"PCSEnvWrapper stepping environment with action_dict: {action_dict}")
        
        # Step the environment and forward its native PCS reward
        obs_dict, rewards, terminations, truncations, info = self.env.step(action_dict)

        # Store updated observations
        self.last_iso_obs = obs_dict["iso"]
        self.last_pcs_obs = obs_dict["pcs"]

        # Extract the PCS reward provided by the environment
        if isinstance(rewards, dict):
            pcs_reward = rewards.get('pcs', 0.0)
        else:
            pcs_reward = rewards[1] if len(rewards) > 1 else 0.0

        self.logger.debug(f"PCSEnvWrapper returning env PCS reward: {pcs_reward}")
        # Return observation, native reward, done/trunc flags, and info
        return (
            obs_dict["pcs"],
            pcs_reward,
            terminations.get('pcs', False),
            truncations.get('pcs', False),
            info
        )


class PreDefinedISOWrapper(gym.Wrapper):
    """
    Environment wrapper for PCS agent with a predefined ISO action sequence.
    Steps the multi-agent env by pulling ISO actions from a fixed sequence and applying PCS agent actions.
    """
    def __init__(self, env, iso_action_sequence=None):
        super().__init__(env)
        # Ensure numpy array sequence (normalized in [-1,1])
        self.iso_action_sequence = np.array(iso_action_sequence)
        self.current_idx = 0
        # Use only PCS observation/action spaces
        self.observation_space = env.observation_space["pcs"]
        self.action_space = env.action_space["pcs"]
        self.logger = logger

    def reset(self, **kwargs):
        self.current_idx = 0
        obs_dict, info = self.env.reset(**kwargs)
        return obs_dict["pcs"], info

    def step(self, pcs_action):
        # Get normalized ISO action from sequence and unnormalize to real space
        raw = self.iso_action_sequence[self.current_idx % len(self.iso_action_sequence)]
        iso_action = self._unnormalize_iso_action(raw)
        self.current_idx += 1
        self.logger.debug(f"PreDefinedISOWrapper using ISO action: {iso_action}")

        action_dict = {"iso": iso_action, "pcs": pcs_action}
        obs_dict, rewards, terminations, truncations, info = self.env.step(action_dict)

        # Extract PCS reward
        if isinstance(rewards, dict):
            pcs_reward = rewards.get('pcs', 0.0)
            iso_reward = rewards.get('iso', 0.0)
        else:
            pcs_reward = rewards[1] if len(rewards) > 1 else 0.0
            iso_reward = rewards[0] if len(rewards) > 0 else 0.0
        # Inject into info for Monitor
        if info is None:
            info = {}
        info['pcs_reward'] = pcs_reward
        info['iso_reward'] = iso_reward
        info['iso_action_sequence_idx'] = self.current_idx - 1

        return (
            obs_dict["pcs"],
            pcs_reward,
            terminations.get('pcs', False),
            truncations.get('pcs', False),
            info
        )

    def _unnormalize_iso_action(self, normalized_action):
        """Convert ISO action from normalized [-1,1] to original action space"""
        iso_space = self.unwrapped.action_space["iso"]
        low = iso_space.low
        high = iso_space.high
        unnormalized = low + (normalized_action + 1.0) * 0.5 * (high - low)
        return unnormalized


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
    pcs_action_sequence=None,
    norm_path=None,
    use_dispatch_action=False,
    use_dayahead_dispatch=False,
    dispatch_strategy="PROPORTIONAL",
    eval_mode=False,
    demand_data_path=None
):
    """
    Create a wrapped environment for training the ISO agent.
    
    This function creates an instance of the EnergyNetV0 environment and wraps
    it for ISO agent training. The wrapper handles the sequential nature of the
    ISO-PCS interaction, where the ISO agent sets prices first, and the PCS
    agent responds.
    
    The function supports using a fixed PCS policy or a predefined sequence of PCS actions,
    creating the appropriate wrapper based on the provided parameters.
    
    Args:
        steps_per_iteration: Number of steps to run per iteration
        cost_type: How grid operation costs are calculated
        pricing_policy: Policy for determining energy prices
        demand_pattern: Pattern of demand variation over time
        seed: Random seed for reproducibility
        log_dir: Directory for logs
        model_dir: Directory for saved models
        plot_dir: Directory for plots
        pcs_policy: Optional fixed policy for the PCS agent
        pcs_action_sequence: Optional sequence of PCS actions to use (instead of policy)
        norm_path: Path to normalization statistics file
        use_dispatch_action: Whether to include dispatch in ISO action space
        dispatch_strategy: Strategy for grid dispatch
        eval_mode: Whether the environment is being created for evaluation
        demand_data_path: Path to data file for DATA_DRIVEN pattern
        
    Returns:
        A VecNormalize-wrapped environment suitable for ISO agent training
    """
    # Validate exclusive use of either policy or sequence
    if pcs_policy is not None and pcs_action_sequence is not None:
        logger.warning("Both pcs_policy and pcs_action_sequence provided. Using pcs_action_sequence.")
        pcs_policy = None
    
    os.makedirs(log_dir, exist_ok=True)
    

    
    if isinstance(pricing_policy, str):
        pricing_policy = PricingPolicy[pricing_policy.upper()]
    
    if isinstance(demand_pattern, str):
        demand_pattern = DemandPattern[demand_pattern.upper()]
        
    if isinstance(cost_type, str):
        cost_type = CostType[cost_type.upper()]
    
    dispatch_config = {
        "use_dispatch_action": use_dispatch_action,
        "dispatch_strategy": dispatch_strategy,
        "use_dayahead_dispatch": use_dayahead_dispatch
    }
    
    # Factory function to create the environment
    def _make_env():
        # Create the base environment
        
        env = EnergyNetV0(
            cost_type=cost_type,
            pricing_policy=pricing_policy,
            demand_pattern=demand_pattern,
            num_pcs_agents=1,
            dispatch_config=dispatch_config,
            demand_data_path=demand_data_path
        )
    
        # Create the proper wrapper based on available PCS inputs
        if pcs_action_sequence is not None:
            # Use predefined PCS action sequence
            env = PreDefinedPCSWrapper(env, pcs_action_sequence=pcs_action_sequence)
            logger.info(f"Using predefined PCS action sequence with {len(pcs_action_sequence)} actions")
        else:
            # Use standard ISOEnvWrapper with optional PCS policy
            if pcs_policy:
                env = ISOEnvWrapper(env, pcs_policy=pcs_policy, eval_mode=eval_mode)
                logger.info("Using provided PCS policy")
            else:
                # Default: zero PCS actions
                env = ISOEnvWrapper(env, pcs_policy=None, eval_mode=eval_mode)
                logger.info("Using default zero PCS actions")
        
        # Add monitor wrapper for logging
        env = Monitor(
            env,
            f"{log_dir}/iso/monitor/{pricing_policy.value}_{cost_type.value}_{int(steps_per_iteration)}.csv",
            info_keywords=("iso_reward", "pcs_reward", "net_exchange", "battery_level", "predicted_demand", "realized_demand")
        )
        
        # Add action scaling wrapper - CRITICAL: normalize ISO actions to [-1, 1]
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        
        return env
    
    # Create the vectorized environment
    env = DummyVecEnv([_make_env])
    
    # Load or create VecNormalize wrapper
    if norm_path and os.path.exists(norm_path):
        try:
            env = VecNormalize.load(norm_path, env)
            logger.info(f"Loaded normalization statistics from {norm_path}")
            # For evaluation, disable normalization updates and reward normalization
            if eval_mode:
                env.training = False
                env.norm_reward = False
                logger.info("Evaluation mode: disabled normalization updates and reward normalization")
        except Exception as e:
            logger.error(f"Failed to load normalization statistics from {norm_path}: {e}")
            logger.info("Creating new VecNormalize wrapper")
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
    else:
        # Create a new VecNormalize wrapper
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        if eval_mode:
            env.training = False
            env.norm_reward = False
            logger.info("Evaluation mode: disabled normalization updates and reward normalization")
    
    # Set the seed if provided
    if seed is not None:
        env.seed(seed)
    
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
    iso_action_sequence=None,
    norm_path=None,
    use_dispatch_action=False,
    dispatch_strategy="PROPORTIONAL",
    eval_mode=False,
    iso_deterministic=False,
    demand_data_path=None
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
        eval_mode: Whether this is evaluation mode
        iso_deterministic: Whether to use deterministic ISO actions (useful when ISO is opponent)
        demand_data_path: Path to demand data for DATA_DRIVEN pattern
        
    Returns:
        Wrapped environment ready for PCS training
    """
    
    # Create the base environment
    base_env = EnergyNetV0(
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        cost_type=cost_type,
        dispatch_config={
            "use_dispatch_action": use_dispatch_action,
            "default_strategy": dispatch_strategy
        },
        demand_data_path=demand_data_path  # Pass the demand data path
    )
    
    # Wrap with predefined ISO sequence or standard PCS wrapper
    if iso_action_sequence is not None:
        env = PreDefinedISOWrapper(base_env, iso_action_sequence=iso_action_sequence)
        logger.info(f"Using predefined ISO action sequence with {len(iso_action_sequence)} steps")
    else:
        env = PCSEnvWrapper(base_env, iso_policy, eval_mode=eval_mode, iso_deterministic=iso_deterministic)
    
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
            epsilon=0
        )
    
    return env