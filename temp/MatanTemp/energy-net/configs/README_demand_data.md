# Data-Driven Demand Pattern

This document explains how to use the data-driven demand pattern feature in EnergyNet.

## Overview

The data-driven demand pattern allows you to specify custom hourly demand values from a YAML configuration file instead of using the built-in demand patterns (CONSTANT, SINUSOIDAL, etc.). This is useful for simulating real-world demand scenarios or testing specific demand profiles.

## Configuration File Format

The demand data configuration file should be in YAML format with the following structure:

```yaml
demand_data:
  format: "hourly"  # Format of the data (hourly or minutes)
  interpolation: "linear"  # Interpolation method (linear or step)
  description: "Optional description of the demand profile"
  values:
    # Time in 24-hour format (HH:MM)
    "00:00": 75.0  # Midnight - demand value in MW
    "01:00": 70.0
    "02:00": 65.0
    # More time points...
    "23:00": 85.0  # 11 PM
```

### Required Fields

- `demand_data`: The root element of the configuration
- `values`: Dictionary mapping time strings to demand values

### Optional Fields

- `format`: Data format, either "hourly" or "minutes" (default: "hourly")
- `interpolation`: Method used to interpolate between time points, either "linear" or "step" (default: "linear")
- `description`: Optional description of the demand profile

### Time Format

Time strings must be in the 24-hour format "HH:MM" where:
- HH: Hours (00-23)
- MM: Minutes (00-59)

Examples: "00:00" (midnight), "08:30" (8:30 AM), "13:45" (1:45 PM), "23:59" (11:59 PM)

### Demand Values

Demand values should be positive numbers representing power demand in megawatts (MW).

## Usage

### Command Line

To use a data-driven demand pattern, specify the `DATA_DRIVEN` pattern type and provide the path to your demand data file:

```bash
python train_iso_recurrent.py \
  --demand-pattern DATA_DRIVEN \
  --demand-data configs/demand_data_sample.yaml \
  --other-options ...
```

### Programmatically

When creating an environment programmatically:

```python
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
from energy_net.controllers.alternating_wrappers import make_iso_env

env = make_iso_env(
    demand_pattern=DemandPattern.DATA_DRIVEN,
    demand_data_path="configs/demand_data_sample.yaml",
    # Other parameters...
)
```

## How It Works

1. The system reads the time-demand pairs from the configuration file
2. Each time string is converted to a fraction of a day (0.0 to 1.0)
3. For simulation times between specified points, the demand is linearly interpolated
4. If the simulation time is outside the range of specified points, the system wraps around (end of day to start of day)

## Examples

The `configs/demand_data_sample.yaml` file provides a sample demand profile representing a typical weekday demand pattern with lower demand at night and higher demand during the day.

## Advanced Features

### Scaling Demand

You can scale the demand values using the `scale_factor` parameter in the environment configuration:

```python
env_config['predicted_demand']['scale_factor'] = 1.2  # Scale demand by 20%
```

This lets you easily adjust the magnitude of the demand without modifying the original data file.

### Custom Interpolation

While the system currently uses linear interpolation, you can specify "step" interpolation in the configuration file to use constant values between time points (useful for discrete demand changes). 