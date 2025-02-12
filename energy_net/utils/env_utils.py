
import numpy as np
from gymnasium.spaces import Box
from typing import List

from ..defs import Bounds

def assign_indexes(dict):
    """
    Assigns an index to each key in the dictionary and saves the mapping.

    Args:
        box_dict (dict): The dictionary of box objects.

    Returns:
        dict: A dictionary mapping each key to an index.
    """
    index_mapping = {key: idx for idx, key in enumerate(dict.keys())}
    return index_mapping




def observation_seperator(observation:dict[str, np.ndarray]):
    """
    Seperates the observation into the agents's observation.

    Parameters:
    observation (dict): The observation of all agents.
    agents (str): The agents to get the observation for.

    Returns:
    dict: The observation of the agents.
    """

    return [observation[name] for name in observation.keys()]


def bounds_to_gym_box(dict_bounds: dict[str, Bounds]) -> Box:
    lows = [bound.low for bound in dict_bounds.values()]
    highs = [bound.high for bound in dict_bounds.values()]
    
    # Concatenate the lower and upper bounds to form the complete space
    low = np.concatenate(lows) if isinstance(lows, list) and all(isinstance(i, np.ndarray) for i in lows) else np.array(lows)
    high = np.concatenate(highs) if isinstance(highs, list) and all(isinstance(i, np.ndarray) for i in highs) else np.array(highs)

    # Create and return the gym.spaces.Box object
    return Box(low=low, high=high, dtype=np.float32)

def dict_to_numpy_array(dict, index_mapping=None):
    """
    Converts the dictionary of box objects to a numpy array using the index mapping.

    Args:
        box_dict (dict): The dictionary of box objects.
        index_mapping (dict): The dictionary mapping each key to an index.

    Returns:
        np.ndarray: A numpy array representing the combined box objects.
    """
    if index_mapping is None:
        index_mapping = assign_indexes(dict)
    # Determine the total size of the numpy array
    total_size = sum(np.prod(box['shape']) for box in dict.values())

    # Create an empty numpy array of the appropriate size
    result_array = np.empty(total_size, dtype=np.float32)

    current_position = 0
    for key, idx in index_mapping.items():
        box = dict[key]
        size = np.prod(box['shape'])
        result_array[current_position:current_position + size] = np.full(box['shape'], box['low']).flatten()
        current_position += size

    return result_array





