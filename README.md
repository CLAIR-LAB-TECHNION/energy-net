# EnergyNet

EnergyNet is a package for smart grid simulation, designed to model and simulate energy dynamics in a network of interconnected entities. It provides tools for creating, managing, and analyzing energy networks, making it suitable for research and development in the field of smart grids.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)


## Features

- **Network Simulation**: Simulate energy dynamics in a network of interconnected entities.
- **Multi-Agent Support**: Utilize the PettingZoo API for multi-agent reinforcement learning environments.
- **Customizable Entities**: Define custom network entities with specific energy dynamics.
- **Experiment Tracking**: Track and visualize experiments using Weights & Biases (wandb) and TensorBoard.

## Installation

To install the dependencies for this project, run:

```sh
pip install -e .
```


## Dependencies

The project has the following dependencies:

- **numpy**
  - **Description:** A fundamental package for scientific computing with Python. It provides support for arrays, matrices, and many mathematical functions.
  - **Usage:** Used for numerical operations and handling large datasets.

- **pettingzoo**
  - **Description:** A library for multi-agent reinforcement learning environments.
  - **Usage:** Provides environments for training and evaluating multi-agent systems.

- **scipy**
  - **Description:** A Python library used for scientific and technical computing. It builds on NumPy and provides a large number of higher-level functions.
  - **Usage:** Used for advanced mathematical functions and algorithms.

- **stable-baselines3**
  - **Description:** A set of improved implementations of reinforcement learning algorithms in Python.
  - **Usage:** Used for training reinforcement learning models.

- **wandb**
  - **Description:** Weights & Biases (wandb) is a tool for experiment tracking, model optimization, and dataset versioning.
  - **Usage:** Used for tracking and visualizing machine learning experiments.

- **tensorboard**
  - **Description:** A suite of visualization tools for TensorFlow. It provides visualizations and tooling for machine learning experiments.
  - **Usage:** Used for visualizing training metrics and model graphs.

- **rl_zoo3**
  - **Description:** A collection of pre-trained reinforcement learning agents using Stable Baselines3.
  - **Usage:** Provides pre-trained models and training scripts for reinforcement learning tasks.
