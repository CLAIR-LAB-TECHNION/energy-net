# EnergyNet: Multi-Agent Reinforcement Learning for Smart Grid Simulation

EnergyNet is a framework for simulating a smart power grid with two interacting agents:
- **ISO:** Sets energy prices and manages dispatch.
- **PCS:** Manages battery charge/discharge to arbitrage prices.

This README provides everything you need to get started: installation, project layout, configuration, and example commands.

---

## 1. Installation

Prerequisites:
- Python 3.8+
- `pip` package manager

Steps:

```bash
# Clone the repository
git clone https://github.com/CLAIR-LAB-TECHNION/energy-net.git
cd energy-net

# Install foundation and tests dependencies
pip install -e .[tests]
```

---

## 2. Dependencies

Core libraries:
- torch
- gymnasium
- stable-baselines3
- sb3-contrib
- numpy
- tqdm
- matplotlib
- yaml

Optional (for logging and visualization):
- tensorboard
- wandb
- opencv-python

---

## 3. Project Structure

```
energy-net/
├── configs/               # YAML config files for env and agents
├── energy_net/            # Core Python package
│   ├── envs/              # Gym environment and wrappers
│   ├── controllers/       # Training wrappers & callbacks
│   ├── dynamics/          # Demand & energy dynamics
│   ├── market/            # Pricing & cost modules
│   ├── components/        # Grid components (batteries, units)
│   └── utils/             # Logging & factories
├── scripts/               # Entry-point training & run scripts
│   ├── train_iso.py       # Train ISO agent
│   ├── train_pcs.py       # Train PCS agent
│   └── run_predefined_schedule_dual.py  # Evaluate ISO+PCS on fixed sequences
├── create_pcs_action_sequence.py  # Generate PCS action sequences
├── create_iso_action_sequence.py  # Generate ISO action sequences
├── pcs_schedule.yml       # Example PCS schedule (yaml_config)
├── iso_dispatch_schedule.yml  # Example ISO schedule
└── tests/                 # Unit tests and examples
```

---

## 4. Configuration Files

- **environment_config.yaml:** global time, pricing, and demand settings
- **iso_config.yaml:** ISO pricing ranges, dispatch bounds, action/observation specs
- **pcs_unit_config.yaml:** PCS battery & action parameters
- **demand_data_sample.yaml:** sample data-driven demand profile

Pass these via `--demand-pattern DATA_DRIVEN --demand-data <path>` when needed.

---

## 5. Generating Predefined Action Sequences

### PCS action sequences

```bash
python3 create_pcs_action_sequence.py \
  --output-file pcs_actions/random.npy \
  --sequence-length 1000 \
  --method pattern \
  --pattern-type charge_discharge_cycle \
  --seed 42
```

### ISO action sequences

```bash
python3 create_iso_action_sequence.py \
  --use-dispatch \
  --output-file iso_actions/schedule.npy \
  --sequence-length 1000 \
  --method yaml_config \
  --pattern-config iso_dispatch_schedule.yml
```

---

## 6. Training Scripts

All training scripts support `--help` to list full options. Below are key parameters and examples.

### 6.1 Train ISO Agent (`train_iso.py`)

Key flags:
- `--algorithm {ppo,recurrent_ppo,td3}`
- `--timesteps` (steps per iteration)
- `--iterations` (number of updates)
- `--seed` (random seed)
- `--demand-pattern {CONSTANT,SINUSOIDAL,DOUBLE_PEAK,TWO_PEAK,DATA_DRIVEN}`
- `--demand-data <yaml>` (required if DATA_DRIVEN)
- `--pricing-policy {ONLINE,CONSTANT,QUADRATIC,INTERVALS}`
- `--cost-type {CONSTANT,VARIABLE,TIME_OF_USE}`
- `--use-dispatch` (enable dispatch action)
- `--pcs-action-file <.npy>` (predefined PCS sequence)
- `--pcs-model <.zip>` (pre-trained PCS policy)
- `--pcs-norm-path <.pkl>` (VecNormalize stats for PCS)
- `--noise-type {normal,ou}` (TD3 policy noise)
- `--policy-noise` / `--final-noise` (TD3 noise decay)

Example:

```bash
python3 train_iso.py \
  --algorithm td3 \
  --timesteps 480 \
  --iterations 2 \
  --demand-pattern DATA_DRIVEN \
  --demand-data configs/demand_data_sample.yaml \
  --pricing-policy ONLINE \
  --cost-type CONSTANT \
  --use-dispatch \
  --pcs-model pcs/pcs_td3_400.zip \
  --pcs-norm-path pcs/vec_normalize_pcs.pkl \
  --policy-noise 5.0 \
  --final-noise 0.5 \
  --noise-type ou
```

### 6.2 Train PCS Agent (`train_pcs.py`)

Key flags:
- `--algorithm {ppo,recurrent_ppo,td3}`
- `--timesteps`, `--iterations`, `--seed`
- `--demand-pattern`, `--demand-data`, `--pricing-policy`, `--cost-type`, `--use-dispatch`
- `--iso-action-file <.npy>` (predefined ISO sequence)
- `--best-model` / `--norm-path` for evaluation-only

Example:

```bash
python3 train_pcs.py \
  --algorithm ppo \
  --timesteps 480 \
  --iterations 10 \
  --demand-pattern SINUSOIDAL \
  --pricing-policy CONSTANT \
  --cost-type VARIABLE \
  --iso-action-file iso_actions/schedule.npy
```

### 6.3 Run Predefined Schedules (`run_predefined_schedule_dual.py`)

Evaluate both agents with fixed action files:

```bash
python3 run_predefined_schedule_dual.py \
  --iso-action-file iso_actions/schedule.npy \
  --pcs-action-file pcs_actions/random.npy \
  --timesteps 500 \
  --demand-pattern SINUSOIDAL \
  --pricing-policy ONLINE \
  --cost-type CONSTANT \
  --use-dispatch \
  --demand-data configs/demand_data_sample.yaml \
  --seed 0 \
  --plot-dir plots/dual_run
```

---

## 7. Logs, Models & Plots

- **logs/**: Monitor CSVs and TensorBoard logs
- **models/**: Saved `.zip` policies and `.pkl` normalization files
- **plots/**: Training and evaluation plots via `PlotCallback`

---

## 8. Evaluation
Use the evaluation-only modes of the training scripts or the dual-run script to assess agent performance after training.

### 8.1 Evaluate ISO Agent
Run `train_iso.py` in evaluation-only mode:

```bash
python3 train_iso.py \
  --eval-only \
  --best-model <path_to_iso_model.zip> \
  --norm-path <path_to_iso_vecnormalize.pkl> \
  --demand-pattern <pattern> \
  --demand-data <path> \
  --pricing-policy <policy> \
  --cost-type <type> \
  --use-dispatch \
  --pcs-model <path_to_pcs_model.zip> \
  --pcs-norm-path <path_to_pcs_norm.pkl> \
  --pcs-action-file <path_to_pcs_sequence.npy> \
  --eval-episodes <n>
```

Key flags:
- `--eval-only`: run evaluation and exit
- `--best-model`: path to the saved ISO model (default: `models/iso_best.zip`)
- `--norm-path`: path to ISO VecNormalize stats (default: `logs/iso/vec_normalize.pkl`)
- `--pcs-model`, `--pcs-norm-path`, `--pcs-action-file`: override PCS behavior during evaluation
- `--eval-episodes`: number of episodes (default: 5)

### 8.2 Evaluate PCS Agent
Run `train_pcs.py` in evaluation-only mode:

```bash
python3 train_pcs.py \
  --eval-only \
  --best-model <path_to_pcs_model.zip> \
  --norm-path <path_to_pcs_norm.pkl> \
  --demand-pattern <pattern> \
  --demand-data <path> \
  --pricing-policy <policy> \
  --cost-type <type> \
  --use-dispatch \
  --iso-action-file <path_to_iso_sequence.npy> \
  --eval-episodes <n>
```

Key flags:
- `--eval-only`: run evaluation and exit
- `--best-model`: path to the saved PCS model (default: `models/pcs_best.zip`)
- `--norm-path`: path to PCS VecNormalize stats (default: `models/pcs_vecnormalize.pkl`)
- `--iso-action-file`: use a fixed ISO action sequence during evaluation
- `--eval-episodes`: number of episodes (default: 5)

### 8.3 Evaluate Both Agents on Predefined Sequences
Run `run_predefined_schedule_dual.py` with fixed ISO and PCS sequences:

```bash
python3 run_predefined_schedule_dual.py \
  --iso-action-file <path_to_iso_sequence.npy> \
  --pcs-action-file <path_to_pcs_sequence.npy> \
  --timesteps <steps> \
  --demand-pattern <pattern> \
  --pricing-policy <policy> \
  --cost-type <type> \
  --use-dispatch \
  --demand-data <path> \
  --seed <seed> \
  --plot-dir <output_dir>
```

