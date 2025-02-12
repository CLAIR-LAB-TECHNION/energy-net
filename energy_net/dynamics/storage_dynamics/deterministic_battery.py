# dynamics/deterministic_battery.py

from typing import Any, Dict
from energy_net.dynamics.energy_dynamcis import ModelBasedDynamics
import math



class DeterministicBattery(ModelBasedDynamics):
    """
    Deterministic Battery Dynamics.
    
    This class models the dynamics of a battery within the smart grid, handling charging
    and discharging actions, applying efficiencies, and accounting for natural decay losses.
    """

    def __init__(self, model_parameters: Dict[str, Any]):
        """
        Initializes the DeterministicBattery with specific model parameters.

        Args:
            model_parameters (Dict[str, Any]):
                - charge_efficiency (float): Efficiency factor for charging (0 < charge_efficiency <= 1).
                - discharge_efficiency (float): Efficiency factor for discharging (0 < discharge_efficiency <= 1).
                - lifetime_constant (float): Lifetime constant representing the rate of decay (lifetime_constant > 0).

        Raises:
            AssertionError: If any required parameter is missing or invalid.
        """
        super().__init__(model_parameters)
        
        # Ensure all required parameters are provided
        required_params = ['charge_efficiency', 'discharge_efficiency', 'lifetime_constant']
        for param in required_params:
            assert param in model_parameters, f"Missing required parameter '{param}' for DeterministicBattery."

        # Validate parameter values
        charge_efficiency = model_parameters['charge_efficiency']
        discharge_efficiency = model_parameters['discharge_efficiency']
        lifetime_constant = model_parameters['lifetime_constant']

        assert 0 < charge_efficiency <= 1, "charge_efficiency must be in the range (0, 1]."
        assert 0 < discharge_efficiency <= 1, "discharge_efficiency must be in the range (0, 1]."
        assert lifetime_constant > 0, "lifetime_constant must be positive."

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.lifetime_constant = lifetime_constant

    def get_value(self, **kwargs) -> float:
        """
        Calculates the updated energy level of the battery based on the action and current state.

        Args:
            **kwargs:
                - time (float): Current time as a fraction of the day (0 to 1).
                - action (float): Charging (+) or discharging (-) power (MW).
                - current_energy (float): Current energy level (MWh).
                - min_energy (float): Minimum energy level (MWh).
                - max_energy (float): Maximum energy level (MWh).
                - charge_rate_max (float): Maximum charge rate (MW).
                - discharge_rate_max (float): Maximum discharge rate (MW).

        Returns:
            float: Updated energy level in MWh.

        Raises:
            AssertionError: If required arguments are missing.
            ValueError: If the adjusted action value is invalid.
        """
        # Extract required parameters from kwargs
        required_kwargs = [
            'time', 'action', 'current_energy', 'min_energy',
            'max_energy', 'charge_rate_max', 'discharge_rate_max'
        ]
        for kw in required_kwargs:
            assert kw in kwargs, f"Missing required argument '{kw}' for DeterministicBattery.get_value."

        time = kwargs['time']
        action = kwargs['action']
        current_energy = kwargs['current_energy']
        min_energy = kwargs['min_energy']
        max_energy = kwargs['max_energy']
        charge_rate_max = kwargs['charge_rate_max']
        discharge_rate_max = kwargs['discharge_rate_max']

        # Validate state parameters
        assert max_energy > min_energy, "max_energy must be greater than min_energy."
        assert min_energy <= current_energy <= max_energy, "current_energy must be within [min_energy, max_energy]."
        # Apply charging or discharging efficiency
        if action > 0:
            assert action <= charge_rate_max, "Charging action exceeds maximum charge rate."
            # Charging
            charge_power = min(action, charge_rate_max)
            energy_change = charge_power * self.charge_efficiency
            new_energy = min(current_energy + energy_change, max_energy)
        elif action < 0:
            assert abs(action) <= discharge_rate_max, "Discharging action exceeds maximum discharge rate."
            # Discharging
            discharge_power = min(abs(action), discharge_rate_max)
            energy_change = discharge_power * self.discharge_efficiency
            new_energy = max(current_energy - energy_change, min_energy)
        else:
            # No action
            new_energy = current_energy

        # Apply natural decay losses to energy capacity
        # Assuming energy_capacity is part of the state, decay can be applied if needed
        # For simplicity, assuming energy_capacity remains constant in this method

        # Advance time (handled externally or within the GridEntity)
        # If decay affects energy_capacity, implement it here or adjust as per project design

        return new_energy

    @staticmethod
    def exp_mult(x: float, lifetime_constant: float, current_time_step: int) -> float:
        """
        Apply exponential decay to a value based on the lifetime constant and current time step.

        This function ensures that the exponent is clamped within a safe range to prevent overflow.

        Parameters
        ----------
        x : float
            The original value to be decayed.
        lifetime_constant : float
            The lifetime constant representing the rate of decay.
        current_time_step : int
            The current time step in the simulation.

        Returns
        -------
        float
            The decayed value.

        Raises
        ------
        ValueError
            If `lifetime_constant` is non-positive.
        """
        if lifetime_constant <= 0:
            raise ValueError("Lifetime constant must be positive.")

        # Calculate the exponent and clamp it to prevent overflow
        exponent = current_time_step / float(lifetime_constant)
        exponent = max(-100, min(100, exponent))  # Clamp to prevent overflow
        return x * math.exp(-exponent)
