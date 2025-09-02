from typing import Callable, Any, TypedDict, List, Dict, Tuple  # Add Tuple import

import numpy as np
import yaml
import os

from ..envs.env_utils import assign_indexes
from ..model.state import State

AggFunc = Callable[[List[Dict[str, Any]]], Dict[str, Any]]


def agg_func_sum(element_arr:List[Dict[str, Any]])-> Dict[str, Any]:
    sum_dict = {}
    for element in element_arr:
        for entry in element:
            if entry in sum_dict.keys():
                sum_dict[entry] += element[entry]
            else:
                sum_dict[entry] = element[entry]
    return sum_dict


def convert_hour_to_int(hour_str):
    # Split the time string to get the hour part
    hour_part = hour_str.split(':')[0]
    # Convert the hour part to an integer
    return int(hour_part)

def condition(state:State):
    pass


def get_predicted_state(cur_state:State, horizon:float)->State:
    state = State({'time':cur_state['time']+horizon})
    return state


def get_value_by_type(dict, wanted_type):
    print(dict)
    print(wanted_type)
    for value in dict.values():
        if type(value) is wanted_type:
            return value
    
    return None


def move_time_tick(cur_time, cur_hour):
    new_time = cur_time + 1
    if new_time % 2 == 0:
        cur_hour += 1
    if cur_hour == 24:
        cur_hour = 0
    return new_time, cur_hour 




def load_config(config_path: str, bValidate:bool= False) -> Dict[str, Any]:
    """
    Loads and validates a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    if not os.path.exists(config_path):
        # try the path from the project's root
        from pathlib import Path
        root_dir = Path(config_path).resolve().parent.parent.parent.parent  # adjust as needed
        config_path = root_dir / config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Example validation
    if bValidate:
        required_energy_params = ['min', 'max', 'init', 'charge_rate_max', 'discharge_rate_max', 'charge_efficiency', 'discharge_efficiency']
        for param in required_energy_params:
            if param not in config.get('energy', {}):
                raise ValueError(f"Missing energy parameter in config: {param}")


    return config


def dict_level_alingment(d, key1, key2):
    return d[key1] if key2 not in d[key1] else d[key1][key2]


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
