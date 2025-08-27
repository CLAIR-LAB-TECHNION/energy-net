import copy
import numpy as np


class State:
    def __init__(self, current_time_step: int = 0, hour: int = 0):
        self.hour = hour
        self.current_time_step = current_time_step

    def get_timedelta_state(self, delta_hours):
        # Create a deep copy of the state
        timedelta_state = self.copy()
        # Update the hour, ensuring it wraps around correctly
        timedelta_state.hour = (self.hour + delta_hours) % 24
        timedelta_state.current_time_step += delta_hours
        return timedelta_state

    def to_numpy(self):
        # Convert current_time_step and hour to a NumPy array
        return np.array([self.current_time_step, self.hour], dtype=np.float32)

    @classmethod
    def from_numpy(cls, array):
        current_time_step = int(array[0])
        hour = int(array[1])
        return cls(current_time_step, hour)

    def copy(self):
        return copy.deepcopy(self)
    
    def get_hour(self):
        return self.hour
    

    
