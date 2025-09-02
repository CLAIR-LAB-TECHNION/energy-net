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

## Project Structure & Modules
Below is the high-level layout of the **energy_net** package and its key submodules:

```
energy_net/                 # Top-level Python package
├── env/                    # Environment registration and base EnergyNetV0 implementation
│   ├── register_envs.py    # Gym environment registration hooks
│   └── iso_env.py          # Core multi-agent Gym env with ISO↔PCS interaction
│   └── pcs_env.py          # Single-agent PCS view of the environment
├── controllers/            # Custom wrappers and training callbacks
│   ├── alternating_wrappers.py  # ISOEnvWrapper, PCSEnvWrapper, and predefined sequence wrappers
│   └── plot_callback.py          # Matplotlib callbacks for action/metric plotting
├── dynamics/               # Demand and consumption dynamics code
│   └── consumption_dynamics/      # DemandPattern implementations (constant, sinusoidal, data-driven)
├── market/                 # Pricing and cost modules
│   ├── pricing/            # PricingPolicy classes (ONLINE, CONSTANT, QUADRATIC)
│   └── cost_types/         # CostType classes (CONSTANT, VARIABLE, TIME_OF_USE)
├── model/                  # Custom neural network architectures (if any)
├── utils/                  # Utility functions (logging, seeding, config parsing)
├── wrappers/               # Generic Gym wrappers and helper code
│   └── rescale_action.py   # Action rescaling between [–1,1] and real action spaces
├── components/             # Front-end or visualization components (if used)
└── scripts/                # Top-level training scripts
    ├── train_iso_recurrent.py      # Direct PPO/RecurrentPPO trainer for ISO agent
    └── train_alternating_td3.py    # Alternating TD3 trainer for ISO↔PCS agents
```

Each folder contains:

- **env/**: Defines and registers the base `EnergyNetV0` environment and adapters for single-agent training.
- **controllers/**: Environment wrappers (ISOEnvWrapper, PCSEnvWrapper, PreDefined wrappers) and the `PlotCallback` for real-time action/metric plots.
- **dynamics/**: Encapsulates various demand patterns and consumption dynamics used by the environment.
- **market/**: Implements pricing policies and cost structures to simulate grid economics.
- **model/**: (Optional) any custom PyTorch `nn.Module` definitions for policies or networks.
- **utils/**: Helper functions for data loading, seeding, logging, and configuration management.
- **wrappers/**: Reusable Gym wrappers, e.g. action resampling or monitoring.
- **components/**: Front-end components or visualization modules (if present).
- **scripts/**: Entry scripts to launch different training workflows: direct ISO-only training, alternating ISO↔PCS training, etc.

With this structure in mind, you can dive into any submodule to inspect specific implementations or extend them for your experiments.

## Agent Interfaces

### Observation Spaces

- **ISO Observation (`env.observation_space['iso']`)**:
  1. `time` (float in [min, max])
     • Fraction of day, typically normalized between 0.0 and 1.0.
  2. `predicted_demand` (float in [min, max])
     • Demand forecast from the chosen pattern (e.g. sinusoidal, data-driven).
  3. `pcs_demand` (float in [min, max])
     • Current PCS power injection/withdrawal (MWh) divided by time step.
  
- **PCS Observation (`env.observation_space['pcs']`)**:
  1. `battery_level` (float from `battery.min` to `battery.max`)
     • Current state of charge (MWh).
  2. `time` (float in [0.0, 1.0])
     • Fraction of day, same as ISO.
  3. `iso_buy_price` (float in [min_price, max_price])
  4. `iso_sell_price` (float in [min_price, max_price])
     • Latest ISO prices after its action, exposed so PCS can arbitrage.

### Action Spaces

- **ISO Action (`env.action_space['iso']`)**:
  Depends on `pricing_policy` and `use_dispatch_action`:
  - **ONLINE** (linear pricing):
    - 2 dims: `[buy_price, sell_price]` mapped via `RescaleAction` from `[-1,1]` → `[min_price, max_price]`.
  - **CONSTANT** (single constant price):
    - 1 dim: `[price]` in `[-1,1]` → `[min_price, max_price]`.
  - **QUADRATIC** (polynomial pricing):
    - 6 dims: `[a_buy, b_buy, c_buy, a_sell, b_sell, c_sell]` → each `[-1,1]` → its own coefficient range.
  - **Dispatch** (if `--use-dispatch` or `dispatch_config.use_dispatch_action`):
    - Adds a 3rd or 7th dimension for the dispatch level, also rescaled in `[-1,1]` → `[0, max_dispatch]`.

- **PCS Action (`env.action_space['pcs']`)**:
  - Single continuous dim:
    - `[charge_discharge_rate]` in `[-discharge_rate_max, charge_rate_max]` (MWh per step).
    - Positive → charging, negative → discharging, clipped by battery capacity and rates.

### Customizing via Environment Inputs

You can tailor the observation and action spaces by passing different flags to the env factories (`make_iso_env`, `make_pcs_env`) or via YAML config:

- `--pricing-policy`: switches between `ONLINE`, `CONSTANT`, `QUADRATIC`, altering ISO action dim.
- `--cost-type`: affects cost and hence reward but does not change action dimensions.
- `--demand-pattern`: chooses how `predicted_demand` is generated, e.g. `SINUSOIDAL`, `DATA_DRIVEN`.
- `--use-dispatch`: toggles whether ISO actions include a dispatch command.

All spaces are ultimately normalized or clipped by `VecNormalize` and `RescaleAction` to ensure stable learning. Feel free to inspect or override them in your custom configs or wrappers.
