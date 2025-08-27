from enum import Enum
import numpy as np
import yaml
import os
from typing import Dict, Any, Optional, Tuple
import logging
import re

class DemandPattern(Enum):
    SINUSOIDAL = "sinusoidal"
    CONSTANT = "constant" 
    DOUBLE_PEAK = "double_peak"
    TWO_PEAK = "two_peak"  # Added for completeness
    DATA_DRIVEN = "data_driven"  # New data-driven pattern

# Cache for loaded demand data to improve performance
_demand_data_cache = {}

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

def _load_demand_data(data_file: str) -> Dict[float, float]:
    """
    Load demand data from a YAML file.
    
    Args:
        data_file: Path to the YAML file containing demand data
        
    Returns:
        Dictionary mapping time fractions to demand values
    """
    # Check cache first
    if data_file in _demand_data_cache:
        return _demand_data_cache[data_file]
    
    # Load data from file
    try:
        with open(data_file, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load demand data from {data_file}: {e}")
        # Return empty data as fallback
        return {}
    
    # Extract demand values
    demand_data = {}
    if "demand_data" in data and "values" in data["demand_data"]:
        values = data["demand_data"]["values"]
        for time_str, demand in values.items():
            try:
                time_fraction = _parse_time_to_fraction(time_str)
                demand_data[time_fraction] = float(demand)
            except ValueError as e:
                logging.warning(f"Skipping invalid time or demand value: {time_str} -> {demand}. Error: {e}")
    
    # Sort by time fraction
    sorted_data = dict(sorted(demand_data.items()))
    
    # Cache the data for future use
    _demand_data_cache[data_file] = sorted_data
    
    return sorted_data

def _interpolate_demand(time_fraction: float, demand_data: Dict[float, float]) -> float:
    """
    Interpolate demand value based on time fraction.
    
    Args:
        time_fraction: Current time as fraction of day (0.0 to 1.0)
        demand_data: Dictionary mapping time fractions to demand values
    
    Returns:
        Interpolated demand value
    """
    if not demand_data:
        logging.warning("Empty demand data, returning default value of 100.0")
        return 100.0
    
    # Get time points
    time_points = list(demand_data.keys())
    
    # Handle edge cases
    if time_fraction <= time_points[0]:
        # Before first point, wrap around to end of day
        prev_time = time_points[-1] - 1.0
        prev_demand = demand_data[time_points[-1]]
        next_time = time_points[0]
        next_demand = demand_data[time_points[0]]
    elif time_fraction >= time_points[-1]:
        # After last point, wrap around to start of day
        prev_time = time_points[-1]
        prev_demand = demand_data[time_points[-1]]
        next_time = time_points[0] + 1.0
        next_demand = demand_data[time_points[0]]
    else:
        # Find surrounding time points
        for i, t in enumerate(time_points):
            if t > time_fraction:
                prev_time = time_points[i-1]
                prev_demand = demand_data[prev_time]
                next_time = t
                next_demand = demand_data[next_time]
                break
    
    # Linear interpolation
    if next_time == prev_time:  # Avoid division by zero
        return prev_demand
    else:
        return prev_demand + (next_demand - prev_demand) * (time_fraction - prev_time) / (next_time - prev_time)

def calculate_demand(time: float, pattern: DemandPattern, config: dict) -> float:
    """
    Calculate demand based on pattern type and configuration
    
    Args:
        time: Current time as fraction of day (0.0 to 1.0)
        pattern: Type of demand pattern to use
        config: Configuration dictionary containing pattern-specific parameters
    
    Returns:
        Demand value for the given time
    """
    # Handle data-driven pattern
    if pattern == DemandPattern.DATA_DRIVEN:
        data_file = config.get('data_file')
        if not data_file:
            logging.error("No data file specified for DATA_DRIVEN pattern")
            # Fall back to constant pattern
            return config.get('base_load', 100.0)
        
        # Load demand data
        demand_data = _load_demand_data(data_file)
        
        # Get demand scale factor (optional)
        scale_factor = config.get('scale_factor', 1.0)
        
        # Interpolate demand for current time
        return _interpolate_demand(time, demand_data) * scale_factor
    
    # Handle other patterns
    base_load = config.get('base_load', 100.0)
    amplitude = config.get('amplitude', 50.0)
    interval_multiplier = config.get('interval_multiplier', 1.0) 
    period_divisor = config.get('period_divisor', 12.0)
    phase_shift = config.get('phase_shift', 0.0)
    
    interval = time * interval_multiplier
    
    if pattern == DemandPattern.SINUSOIDAL:
        return base_load + amplitude * np.cos(
            (interval + phase_shift) * np.pi / period_divisor
        )
    elif pattern == DemandPattern.CONSTANT:
        return base_load
    elif pattern == DemandPattern.DOUBLE_PEAK:
        # Create a double peak pattern with morning and evening peaks
        morning_peak = 0.1  # 6 AM
        evening_peak = 0.75  # 6 PM
        morning_factor = np.exp(-20 * ((interval - morning_peak) ** 2))
        evening_factor = np.exp(-20 * ((interval - evening_peak) ** 2))
        return base_load + amplitude * (morning_factor + evening_factor)
    elif pattern == DemandPattern.TWO_PEAK:
        return base_load + base_load*interval*0.5
    else:
        raise ValueError(f"Unknown demand pattern: {pattern}")
    
# Add a new function to get raw demand data for a file
def get_raw_demand_data(data_file: str) -> Dict[float, float]:
    """
    Get raw demand data from a file without any processing.
    This is useful for visualization purposes.
    
    Args:
        data_file: Path to the YAML file containing demand data
        
    Returns:
        Dictionary mapping time fractions to demand values
    """
    return _load_demand_data(data_file)
    
