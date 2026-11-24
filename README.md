# EnergyNet

> A multi-agent reinforcement learning framework for smart grid simulation

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**EnergyNet** is a Python framework for simulating smart power grids with interacting reinforcement learning agents. It provides OpenAI Gym-compatible environments for modeling energy management scenarios, including battery storage optimization, grid operator pricing strategies, and multi-agent coordination.

The framework enables researchers and practitioners to experiment with energy arbitrage, demand response, and grid optimization using state-of-the-art RL algorithms. Each environment simulates realistic grid dynamics with configurable consumption patterns, pricing strategies, and storage constraints.

EnergyNet is designed for flexibility and extensibility, making it suitable for both academic research and practical experimentation in smart grid management.

---

## Key Features

- **Three OpenAI Gym Environments**
  - `ProsumerEnv`: A prosumer (battery + load + generator) managing storage under dynamic tariffs
  - `ISOEnv`: Grid operator pricing and dispatch optimization
  - `AlternatingISOEnv`: Multi-agent training with ISO-prosumer interaction

- **Realistic Grid Simulation**
  - Configurable consumption patterns (constant, sinusoidal, data-driven)
  - Dynamic pricing strategies
  - Battery storage dynamics with charge/discharge constraints
  - Shortage penalties and demand-supply balancing

- **Multi-Agent Learning**
  - Alternating training framework for ISO and prosumer agents
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
from energy_net.gym_envs.prosumer_env import ProsumerEnv
from stable_baselines3 import SAC

# Create environment
env = ProsumerEnv(
    load_data_file='tests/gym/data_for_tests/synthetic_household_consumption_test.csv',
    load_forecast_file='tests/gym/data_for_tests/consumption_predictions_without_features.csv',
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

### ProsumerEnv

Simulates one prosumer with exactly **one battery, one load and one
generator** (a household without PV uses a zero-output generator):
- **Load and generation**: not controlled by any agent
- **Battery storage**: the agent's charge/discharge decision
- **Grid interaction**: net-metered import/export at dynamic tariffs; surplus
  generation first charges the battery and the rest is exported

**Agent Goal**: Minimize energy costs through strategic battery management.

#### Architecture

The gym environment is a thin adapter; the simulation runs without gym:

| Layer | Package | Responsibility |
|---|---|---|
| Gym adapter | `energy_net.gym_envs` | spaces, action decoding, observation encoding |
| Simulation | `energy_net.simulation` | episodes, tariffs, settlement, rewards |
| Entities | `energy_net.grid_entities` | `Prosumer`, `Battery`, `ConsumptionUnit`, `ProductionUnit` |
| Ground truth | `energy_net.dynamics` | `CSVDynamics`, `ProfileDynamics`, `ConstantDynamics` |
| Predictions | `energy_net.forecasting` | `TableForecaster`, `PredictorForecaster`, `NoisyGroundTruthForecaster` |

Each load and generator entity owns **ground-truth dynamics** (what actually
happens) and a **forecaster** (what some algorithm predicts). Any object with
`forecast(issue_time, target_times)` can be a forecaster, including one backed
by your own model-based or model-free learner; its optional
`observe(times, realized)` hook only ever receives values already realized.

```python
from energy_net.simulation import make_prosumer_simulator

simulator = make_prosumer_simulator(
    generation_data_file="pv.csv",          # column "Production"
    generation_forecast_file="pv_forecast.csv",  # column "predicted_production"
)
env = ProsumerEnv(simulator)

# Privileged query of the ground truth (never part of the observation).
env.ground_truth("generator", ["2025-07-10 12:00", "2025-07-10 12:30"])
```

#### Observation, tariffs and reward variants

The observation is a flat vector with a named layout (`env.observation_layout`):
storage, previous load and generation, buy and sell tariffs (prosumer
perspective), load and generation forecasts with a validity mask, battery
capacities, and seven calendar features (158 values with 48 forecast steps).
Use `env.encoder.field(obs, "buy_tariff")` instead of indices.

The default reward is the money settled under net metering:

```python
# Documented POMDP baseline: no additional reward shaping.
env = ProsumerEnv(include_shortage_penalty=False)

# Experimental variant: same electricity bill, plus a fixed penalty per
# interval where any demand must be supplied by the grid.
env = ProsumerEnv(include_shortage_penalty=True, shortage_penalty=1.0)
```

Any `RewardFunction` can be injected (`ProsumerEnv(reward_fn=...)`). The
penalty is *not* the price of unmet demand: household demand is always
satisfied because the grid supplies any residual. Settle and log the real
monetary cost (`info["step_money"]`) separately from the shaped reward
(`info["reward_components"]`).

### ISOEnv (Independent System Operator)

Simulates a grid operator managing:
- **Pricing**: Set electricity prices for each time period
- **Dispatch**: Determine energy supply to meet predicted demand

**Agent Goal**: Maximize revenue while maintaining grid stability and minimizing forecast errors.

### AlternatingISOEnv (Multi-Agent)

Coordinates ISO and prosumer agents in an alternating training framework:
- ISO sets prices → prosumer responds → ISO adapts → prosumer adapts
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
│   ├── gym_envs/                  # Thin gym adapters (Prosumer, ISO, Alternating)
│   ├── simulation/                # Gym-free prosumer simulator, tariffs, rewards
│   ├── grid_entities/             # Prosumer, battery, load and generator units
│   ├── dynamics/                  # Ground-truth dynamics (CSV, profiles, constants)
│   ├── forecasting/               # Built-in forecasters
│   └── foundation/                # Base classes and interfaces
├── tests/                         # Test suite and test data
│   ├── golden/                    # Recorded legacy trajectories (regression oracle)
│   ├── gym/                       # Environment test data and generators
│   └── unittests/                 # Unit tests and prediction demonstration
├── Energy_Net_Tutorial.ipynb      # Comprehensive environment tutorial
├── Prediction_Tutorial.ipynb      # Energy-target prediction tutorial
└── README.md                      # This file
```

---

## Dependencies

**Core Requirements:**
- Python 3.14 (`>=3.14,<3.15`)
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
