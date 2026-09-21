# EnergyNet

> A multi-agent reinforcement learning framework for smart grid simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**EnergyNet** is a Python framework for simulating smart power grids with interacting reinforcement learning agents. It provides OpenAI Gym-compatible environments for modeling energy management scenarios, including battery storage optimization, grid operator pricing strategies, and multi-agent coordination.

The framework enables researchers and practitioners to experiment with energy arbitrage, demand response, and grid optimization using state-of-the-art RL algorithms. Each environment simulates realistic grid dynamics with configurable consumption patterns, pricing strategies, and storage constraints.

EnergyNet is designed for flexibility and extensibility, making it suitable for both academic research and practical experimentation in smart grid management.

---

## Key Features

- **Three OpenAI Gym Environments**
  - `PCSEnv`: Battery storage management with price arbitrage
  - `ISOEnv`: Grid operator pricing and dispatch optimization
  - `AlternatingISOEnv`: Multi-agent training with ISO-PCS interaction

- **Realistic Grid Simulation**
  - Configurable consumption patterns (constant, sinusoidal, data-driven)
  - Dynamic pricing strategies
  - Battery storage dynamics with charge/discharge constraints
  - Shortage penalties and demand-supply balancing

- **Multi-Agent Learning**
  - Alternating training framework for ISO and PCS agents
  - Coordinated or adversarial agent objectives
  - Flexible reward structures

- **Integration with Popular RL Libraries**
  - Compatible with Stable-Baselines3, SB3-Contrib
  - Supports PPO, SAC, TD3, and other algorithms
  - VecNormalize support for training stability

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/CLAIR-LAB-TECHNION/energy-net.git
cd energy-net

# Install the package
pip install -e .

# For RL training capabilities
pip install -e .[rl]

# For development and testing
pip install -e .[test]
```

### Basic Usage

```python
from energy_net.gym_envs.pcs_env import PCSEnv
from stable_baselines3 import SAC

# Create environment
env = PCSEnv(
    test_data_file='tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
    predictions_file='tests/gym/data_for_tests/consumption_predictions_without_features.csv',
    shortage_penalty=1.0
)

# Train an agent
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
obs, info = env.reset()
for _ in range(48):  # One day (48 half-hour steps)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## Environments

### PCSEnv (Production-Consumption-Storage)

Simulates a household or facility managing:
- **Energy consumption**: Time-varying demand that must be met
- **Battery storage**: Charge/discharge decisions for cost optimization
- **Grid interaction**: Buy/sell energy at dynamic prices

**Agent Goal**: Minimize energy costs while meeting consumption requirements through strategic battery management.

### ISOEnv (Independent System Operator)

Simulates a grid operator managing:
- **Pricing**: Set electricity prices for each time period
- **Dispatch**: Determine energy supply to meet predicted demand

**Agent Goal**: Maximize revenue while maintaining grid stability and minimizing forecast errors.

### AlternatingISOEnv (Multi-Agent)

Coordinates ISO and PCS agents in an alternating training framework:
- ISO sets prices → PCS responds → ISO adapts → PCS adapts
- Enables study of strategic interactions between grid operators and consumers
- Configurable reward structures (competitive or cooperative)

---

## Documentation

### Energy prediction

Select the target explicitly when creating a predictor. Demand, generation, net load, and other energy series use the same API, and tabular outputs are named for the selected target.

```python
from prediction.energy_predictor import create_energy_predictor

predictor = create_energy_predictor(
    "generation.csv",
    target_col="Production",
)
forecast = predictor.predict("2025-06-15", "12:00")
daily_forecasts = predictor.predict_days("2025-06-15")
```

Consumption-named callables and the former module import path remain available only as deprecated compatibility interfaces.

For detailed tutorials, examples, and advanced usage, see:

📓 **[EnergyNet Tutorial](Energy_Net_Tutorial.ipynb)** - Comprehensive guide covering:
- Environment setup and configuration
- Training agents with different algorithms
- Visualization and analysis
- Multi-agent coordination scenarios

📈 **[Prediction Tutorial](Prediction_Tutorial.ipynb)** - Target-neutral forecasting for explicitly selected demand, generation, or other energy series.

Run the standalone prediction example from the repository root:

```bash
python tests/unittests/energy_prediction_demonstration.py
```

---

## Project Structure

```
energy-net/
├── prediction/                    # Target-neutral energy forecasting
│   └── energy_predictor.py        # Primary prediction API
├── energy_net/                    # Core package
│   ├── gym_envs/                  # Gym environments (PCS, ISO, Alternating)
│   ├── grid_entities/             # Grid components (batteries, consumption units, etc.)
│   └── foundation/                # Base classes and dynamics
├── tests/                         # Test suite and test data
│   ├── gym/                       # Environment test data and generators
│   └── unittests/                 # Unit tests and prediction demonstration
├── Energy_Net_Tutorial.ipynb      # Comprehensive environment tutorial
├── Prediction_Tutorial.ipynb      # Energy-target prediction tutorial
└── README.md                      # This file
```

---

## Dependencies

**Core Requirements:**
- Python 3.10+
- numpy, scipy, pandas
- gymnasium
- pyyaml
- scikit-learn

**For RL Training:**
- stable-baselines3
- torch
- sb3-contrib
- matplotlib
- tensorboard

Install RL dependencies with: `pip install -e .[rl]`

Install development and testing dependencies with: `pip install -e .[test]`

Install Excel import support for `process_consumption_file` with:
`pip install -e .[excel]`

---

## Contributing

This project is developed by [CLAIR-LAB-TECHNION](https://github.com/CLAIR-LAB-TECHNION). Contributions, issues, and feature requests are welcome!

---

## License

This project is licensed under the MIT License.
