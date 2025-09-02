import numpy as np
from gymnasium.spaces import Box
from energy_net.defs import Bounds
DEFAULT_BOUND_VALUE = 1e10

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



def bounds_to_gym(bounds:Bounds) -> Box:
    # Check if dtype is compatible
    dtype = bounds.low.dtype

    # Gymnasium doesn't accept infinite bounds — optionally clip if needed
    low = np.where(np.isfinite(bounds.low), bounds.low, -DEFAULT_BOUND_VALUE)
    high = np.where(np.isfinite(bounds.high), bounds.high, DEFAULT_BOUND_VALUE)

    return Box(low=low, high=high, dtype=dtype)



def gym_to_bounds(box:Box) -> Bounds:
    low = np.where(np.isclose(box.low, -DEFAULT_BOUND_VALUE),-np.inf,box.low)
    high = np.where(np.isclose(box.high, DEFAULT_BOUND_VALUE),np.inf,box.high)


    return Bounds(low=low,high=high)







