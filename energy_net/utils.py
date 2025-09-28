import logging
import os
import re
import yaml
import logging
from typing import Dict, Any
_data_cache = {}

def load_data_from_yaml(data_file: str, data_key: str) -> Dict[float, float]:
    """
    Load data from a YAML file and cache it.

    Args:
        data_file: Path to the YAML file.
        data_key: Key in the YAML file to extract data (e.g., 'CONSUMPTION_DATA' or 'PRODUCTION_DATA').

    Returns:
        Dictionary mapping time fractions to values.
    """
    # Check cache first
    if data_file in _data_cache:
        return _data_cache[data_file]

    # Load data from file
    try:
        with open(data_file, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load data from {data_file}: {e}")
        return {}

    # Extract values
    extracted_data = {}
    if data_key in data and "values" in data[data_key]:
        values = data[data_key]["values"]
        for time_str, value in values.items():
            try:
                time_fraction = _parse_time_to_fraction(time_str)
                extracted_data[time_fraction] = float(value)
            except ValueError as e:
                logging.warning(f"Skipping invalid time or value: {time_str} -> {value}. Error: {e}")

    # Sort by time fraction
    sorted_data = dict(sorted(extracted_data.items()))

    # Cache the data for future use
    _data_cache[data_file] = sorted_data

    return sorted_data


def interpolate_value(time_fraction: float, data: Dict[float, float]) -> float:
    """
    Interpolate a value based on time fraction.

    Args:
        time_fraction: Current time as a fraction of the day (0.0 to 1.0).
        data: Dictionary mapping time fractions to values.

    Returns:
        Interpolated value.
    """
    if not data:
        logging.warning("Empty data, returning default value of 100.0")
        return 100.0

    # Get time points
    time_points = list(data.keys())

    # Handle edge cases
    if time_fraction <= time_points[0]:
        prev_time = time_points[-1] - 1.0
        prev_value = data[time_points[-1]]
        next_time = time_points[0]
        next_value = data[time_points[0]]
    elif time_fraction >= time_points[-1]:
        prev_time = time_points[-1]
        prev_value = data[time_points[-1]]
        next_time = time_points[0] + 1.0
        next_value = data[time_points[0]]
    else:
        for i, t in enumerate(time_points):
            if t > time_fraction:
                prev_time = time_points[i - 1]
                prev_value = data[prev_time]
                next_time = t
                next_value = data[next_time]
                break

    # Linear interpolation
    if next_time == prev_time:
        return prev_value
    else:
        return prev_value + (next_value - prev_value) * (time_fraction - prev_time) / (next_time - prev_time)


def setup_logger(name: str, log_file: str, level=logging.DEBUG) -> logging.Logger:
    """
    Sets up a logger with the specified name and log file.

    Ensures that each logger has only one handler to prevent duplicate logs
    and unclosed file handles.

    Args:
        name (str): The name of the logger.
        log_file (str): The path to the log file.
        level (int): Logging level (default: logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers, do not add another one
    if not logger.handlers:
        logger.setLevel(level)

        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create a file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(fh)

        # Optionally, prevent log messages from being propagated to the root logger
        logger.propagate = False

    return logger

def _parse_time_to_fraction(time_str: str) -> float:
    """
    Convert a time string in HH:MM format to a fraction of a day.

    Args:
        time_str: Time string in HH:MM format (e.g., "08:30", "14:45")

    Returns:
        Float representing the fraction of a day (0.0 to 1.0)
    """
    # Handle both HH:MM and H:MM formats
    match = re.match(r"(\d{1,2}):(\d{2})", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM or H:MM")

    hours = int(match.group(1))
    minutes = int(match.group(2))

    # Validate hours and minutes
    if hours < 0 or hours >= 24:
        raise ValueError(f"Hours must be between 0 and 23, got {hours}")
    if minutes < 0 or minutes >= 60:
        raise ValueError(f"Minutes must be between 0 and 59, got {minutes}")

    # Convert to fraction of day
    return (hours * 60.0 + minutes) / (24.0 * 60.0)