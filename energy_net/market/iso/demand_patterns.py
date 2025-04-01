from enum import Enum
import numpy as np

class DemandPattern(Enum):
    SINUSOIDAL = "sinusoidal"
    CONSTANT = "constant" 
    DOUBLE_PEAK = "double_peak"

def calculate_demand(time: float, pattern: DemandPattern, config: dict) -> float:
    """
    Calculate demand based on pattern type and configuration
    
    Args:
        time: Current time as fraction of day (0.0 to 1.0)
        pattern: Type of demand pattern to use
        config: Configuration dictionary containing:
            - base_load: Base demand level
            - amplitude: Maximum deviation from base load
            - interval_multiplier: Time scaling factor
            - period_divisor: Controls frequency of oscillation
            - phase_shift: Shifts the pattern in time
    """
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
        morning_peak = 0.25  # 6 AM
        evening_peak = 0.75  # 6 PM
        morning_factor = np.exp(-20 * ((interval - morning_peak) ** 2))
        evening_factor = np.exp(-20 * ((interval - evening_peak) ** 2))
        return base_load + amplitude * (morning_factor + evening_factor)
    else:
        raise ValueError(f"Unknown demand pattern: {pattern}")
