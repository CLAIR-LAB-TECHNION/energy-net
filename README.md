# EnergyNet: Multi-Agent Reinforcement Learning for Smart Grid Simulation

EnergyNet is a sophisticated framework for simulating smart grid environments and training reinforcement learning agents to optimize grid operations. The framework features a multi-agent environment with two key strategic entities: the Independent System Operator (ISO) and Power Control System (PCS) agents.

## System Overview

### Key Components

1. **Independent System Operator (ISO)**: Sets energy prices and manages dispatch commands for the grid
2. **Power Control System (PCS)**: Controls battery storage systems by deciding when to charge/discharge in response to price signals
3. **EnergyNetV0 Environment**: Multi-agent environment that handles the sequential interactions between ISO and PCS
4. **Alternating Training Framework**: Enables stable training of multiple agents through an iterative process

### Training Workflow

The system uses an alternating training approach where:

1. First, the ISO agent is trained with a fixed (default) PCS policy
2. Then, the PCS agent is trained with the fixed ISO policy from step 1
3. Next, the ISO agent is retrained with the fixed PCS policy from step 2
4. Steps 2-3 are repeated for a specified number of iterations

This approach helps find stable equilibrium policies between the two agents, similar to game-theoretic approaches for multi-agent systems.

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/CLAIR-LAB-TECHNION/EnergyNet.git
   cd EnergyNet
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Install RL-Zoo3 (if not automatically installed):
   ```bash
   pip install rl_zoo3
   ```

## Running Training

### Using the Direct Training Script

The easiest way to train both agents is to use the provided training script:

```bash
chmod +x train_rlzoo_direct.sh
./train_rlzoo_direct.sh
```

This script implements an alternating training approach where:

1. **Initial ISO Training**:
   - Trains the ISO agent with a default PCS policy
   - Saves the model in `logs/iso/ppo/run_1/`

2. **Alternating Training**:
   - Trains PCS agent with fixed ISO policy
   - Trains ISO agent with fixed PCS policy
   - Repeats for specified number of iterations

3. **Key Parameters** (in the script):
   ```bash
   # Number of training iterations
   ITERATIONS=5
   
   # Steps per training iteration
   TIMESTEPS=50
   
   # Random seed for reproducibility
   SEED=422
   
   # Environment configuration
   BASE_ENV_KWARGS=(
     "cost_type:'CONSTANT'"    # Fixed operating costs
     "pricing_policy:'Online'" # Dynamic pricing
     "demand_pattern:'CONSTANT'" # Steady demand
     "use_dispatch_action:True"  # ISO can set dispatch
   )
   ```

4. **Output**:
   - Models saved in `logs/iso/ppo/run_1/` and `logs/pcs/ppo/run_1/`
   - Training plots in respective model directories
   - TensorBoard logs in `logs/iso/tensorboard/` and `logs/pcs/tensorboard/`

### Customizing Training

You can modify the training process by editing `train_rlzoo_direct.sh`:

1. **Change Training Duration**:
   ```bash
   ITERATIONS=10  # More training iterations
   TIMESTEPS=100  # More steps per iteration
   ```

2. **Modify Environment Settings**:
   ```bash
   BASE_ENV_KWARGS=(
     "cost_type:'VARIABLE'"     # Variable costs
     "pricing_policy:'QUADRATIC'" # Quadratic pricing
     "demand_pattern:'SINUSOIDAL'" # Cyclic demand
     "use_dispatch_action:True"
   )
   ```

3. **Adjust Hyperparameters**:
   Edit the YAML files in `rl-baselines3-zoo/hyperparams/ppo/`:
   ```yaml
   ISO-RLZoo-v0:
     n_steps: 1024        # More steps per update
     batch_size: 128      # Larger batch size
     learning_rate: 1e-4  # Different learning rate
   ```

4. **Change Evaluation Frequency**:
   ```bash
   --eval-freq 100        # Evaluate every 100 steps
   --eval-episodes 20     # More evaluation episodes
   ```

### Training Process Flow

1. **Setup**:
   - Creates necessary directories
   - Sets up hyperparameter files
   - Configures environment parameters

2. **Initial ISO Training**:
   - Trains ISO agent with default PCS policy
   - Saves best model

3. **Alternating Training Loop**:
   - For each iteration:
     1. Train PCS with fixed ISO policy
     2. Save PCS model
     3. Train ISO with fixed PCS policy
     4. Save ISO model
     5. Update model paths for next iteration

4. **Monitoring**:
   - Plots saved in model directories
   - TensorBoard logs for metrics
   - Evaluation results every `eval_freq` steps

### Troubleshooting Training

1. **Model Not Found**:
   - Check if training completed successfully
   - Verify model paths in the script
   - Look for error messages in logs

2. **Training Instability**:
   - Adjust learning rate in hyperparameter files
   - Modify batch size or number of steps
   - Check reward scaling in environment

3. **Memory Issues**:
   - Reduce batch size
   - Decrease number of environments
   - Lower evaluation frequency

## Configuration

### Configuration Files

The system uses three primary configuration files:

1. **environment_config.yaml**: General environment settings
   - Time parameters (step duration, max steps)
   - Pricing parameters
   - Demand prediction parameters

2. **iso_config.yaml**: ISO-specific settings
   - Pricing ranges and defaults
   - Dispatch configuration
   - Observation and action space parameters

3. **pcs_unit_config.yaml**: PCS-specific settings
   - Battery parameters (capacity, charge/discharge rates)
   - Observation and action space parameters
   - Consumption and production unit settings

### Environment Parameters

When creating environments, you can specify various parameters:

```python
# Example of creating an environment with custom parameters
env = EnergyNetV0(
    cost_type=CostType.CONSTANT,         # How grid costs are calculated
    pricing_policy=PricingPolicy.ONLINE,  # How prices are determined
    demand_pattern=DemandPattern.SINUSOIDAL, # Demand pattern over time
    num_pcs_agents=1,                    # Number of PCS units
    dispatch_config={                     # Dispatch configuration
        "use_dispatch_action": True,
        "default_strategy": "PROPORTIONAL"
    }
)
```

Available options include:

1. **Cost Types**:
   - `CONSTANT`: Fixed operating costs
   - `VARIABLE`: Costs that vary with demand
   - `TIME_OF_USE`: Time-dependent costs

2. **Pricing Policies**:
   - `ONLINE`: Dynamic pricing based on current conditions
   - `QUADRATIC`: Prices following quadratic functions
   - `CONSTANT`: Fixed prices

3. **Demand Patterns**:
   - `SINUSOIDAL`: Smooth cyclic demand pattern
   - `RANDOM`: Randomized demand
   - `PERIODIC`: Repeating patterns
   - `SPIKES`: Demand with occasional spikes

## Environment Wrappers

The system uses several wrappers to adapt the multi-agent environment for single-agent training:

1. **ISOEnvWrapper**: Wraps the environment for ISO training, handling PCS actions automatically
   - Exposes only ISO observation and action spaces
   - Uses a fixed PCS policy to generate PCS actions
   - Calculates ISO-specific rewards

2. **PCSEnvWrapper**: Wraps the environment for PCS training, handling ISO actions automatically
   - Exposes only PCS observation and action spaces
   - Uses a fixed ISO policy to generate ISO actions
   - Calculates PCS-specific rewards

3. **RescaleAction**: Scales actions between neural network output range [-1, 1] and environment action space

## Monitoring and Visualization

The framework includes callbacks for monitoring training progress:

1. **PlotCallback**: Tracks and visualizes agent actions during training
   - Automatically detects agent type from environment ID
   - Creates plots based on the current iteration
   - Saves plots to appropriate directories based on agent type

Logs and plots are saved in:
- `logs/iso/`: ISO agent logs and plots
- `logs/pcs/`: PCS agent logs and plots
- `logs/tensorboard/`: TensorBoard logs for both agents

## Reward Functions

Each agent has a specialized reward function:

1. **ISO Reward**: Balances multiple objectives
   - Minimizes reserve costs
   - Minimizes dispatch costs 
   - Avoids demand shortfalls
   - Maintains stable prices

2. **PCS Reward**: Cost-based rewards
   - Buys energy when prices are low
   - Sells energy when prices are high
   - Maximizes profit from energy arbitrage

## Advanced Usage

### Custom Environment Creation

For fine-grained control, you can create and wrap environments manually:

```python
from energy_net.env import EnergyNetV0
from stable_baselines3 import PPO
from alternating_wrappers import ISOEnvWrapper, PCSEnvWrapper

# Create base environment
env = EnergyNetV0(
    pricing_policy="ONLINE",
    demand_pattern="SINUSOIDAL",
    cost_type="CONSTANT"
)

# Load fixed policy
pcs_policy = PPO.load("logs/pcs/ppo/run_1/PCS-RLZoo-v0.zip")

# Create wrapped environment for ISO training
wrapped_env = ISOEnvWrapper(env, pcs_policy=pcs_policy)
```

### Customizing Hyperparameters

You can modify the RL algorithm hyperparameters by editing the YAML files created in the `rl-baselines3-zoo/hyperparams/ppo/` directory:

```yaml
# Example for ISO agent
ISO-RLZoo-v0:
  normalize: "{'norm_obs': True, 'norm_reward': True}"
  n_envs: 1
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  n_steps: 2048
  batch_size: 64
  gae_lambda: 0.95
  gamma: 0.99
  n_epochs: 10
  ent_coef: 0.0
  learning_rate: !!float 3e-4
  clip_range: 0.2
```

### Advanced RL-Zoo3 Features

For more advanced RL-Zoo3 usage with Energy-Net, including:
- Experimenting with different algorithms (SAC, TD3)
- Automated hyperparameter optimization with Optuna
- Comparative analysis and benchmarking
- Scaling up experiments with parallel training
- Advanced tracking and visualization
- Sharing and publishing models

Please refer to [READMEzoo.md](READMEzoo.md) for comprehensive examples and instructions.

## Troubleshooting

### Common Issues

1. **Error: Module not found**
   - Make sure you've installed the package with `pip install -e .`
   - Verify PYTHONPATH includes the project directory

2. **Action scaling issues**
   - If you see unusually small values in the logs, check that the action rescaling is working correctly
   - Debug by adding logging statements to the unnormalization methods

3. **Unable to load policies**
   - Verify that policy paths are correct in the training script
   - Ensure the saved policies have compatible architecture with the current environment
