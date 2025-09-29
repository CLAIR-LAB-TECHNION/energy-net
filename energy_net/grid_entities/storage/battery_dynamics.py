from typing import Any, Dict
from energy_net.dynamics import ModelBasedDynamics
import math
from energy_net.model.action import Action

class BatteryDynamics(ModelBasedDynamics):
    """Base class for battery dynamics implementations."""
    
    def __init__(self, model_parameters: Dict[str, Any]):
        super().__init__(model_parameters)
        self.model_parameters = self._validate_and_prepare_config(model_parameters)

    def _validate_and_prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate battery configuration parameters."""
        required_params = ['charge_efficiency', 'discharge_efficiency']
        for param in required_params:
            assert param in config, f"Missing required parameter '{param}'"

        # Validate efficiency values
        for name, value in [
            ('charge_efficiency', config['charge_efficiency']),
            ('discharge_efficiency', config['discharge_efficiency'])
        ]:
            assert 0 < value <= 1, f"{name} must be in range (0, 1]"

        return config

    def get_value(self, **kwargs) -> float:
        """Get new energy value based on action and current state."""
        required_kwargs = [
            'time', 'action', 'current_energy', 'min_energy',
            'max_energy', 'charge_rate_max', 'discharge_rate_max'
        ]
        for kw in required_kwargs:
            assert kw in kwargs, f"Missing required argument '{kw}'"

        return self.calculate(**kwargs)

    def calculate(self, **kwargs) -> float:
        """Calculate the new energy value. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement calculate method")

class DeterministicBatteryDynamics(BatteryDynamics):
    """Deterministic implementation of battery dynamics.
    
    This class models the dynamics of a storage within the smart grid, handling charging
    and discharging actions, applying efficiencies, and accounting for natural decay losses.
    """

    def __init__(self, model_parameters: Dict[str, Any]):
        """
        Initialize the DeterministicBatteryDynamics with model parameters.

        Args:
            model_parameters (Dict[str, Any]):
                - charge_efficiency (float): Efficiency factor for charging (0 < charge_efficiency <= 1).
                - discharge_efficiency (float): Efficiency factor for discharging (0 < discharge_efficiency <= 1).
        """
        super().__init__(model_parameters)
        self.charge_efficiency = self.model_parameters['charge_efficiency']
        self.discharge_efficiency = self.model_parameters['discharge_efficiency']

    def calculate(self, **kwargs) -> float:
        """
        Calculate the updated energy level based on action and current state.

        Args:
            **kwargs:
                - time (float): Current time as a fraction of the day (0 to 1).
                - action (Action): Action containing charging/discharging amount.
                - current_energy (float): Current energy level (MWh).
                - min_energy (float): Minimum energy level (MWh).
                - max_energy (float): Maximum energy level (MWh).
                - charge_rate_max (float): Maximum charge rate (MW).
                - discharge_rate_max (float): Maximum discharge rate (MW).

        Returns:
            float: Updated energy level in MWh.

        Raises:
            ValueError: If the action is not an Action instance or amount is invalid.
        """
        action = kwargs['action']
        current_energy = kwargs['current_energy']
        charge_rate_max = kwargs['charge_rate_max']
        discharge_rate_max = kwargs['discharge_rate_max']

        if not isinstance(action, Action):
            raise ValueError("action must be an Action dataclass instance")
        
        amount = action.amount

        if amount > 0:
            assert amount <= charge_rate_max, "Charging action exceeds maximum charge rate"
            charge_power = min(amount, charge_rate_max)
            energy_change = charge_power * self.charge_efficiency
            new_energy = current_energy + energy_change
        elif amount < 0:
            assert abs(amount) <= discharge_rate_max, "Discharging action exceeds maximum discharge rate"
            discharge_power = min(abs(amount), discharge_rate_max)
            energy_change = discharge_power * self.discharge_efficiency
            new_energy = current_energy - energy_change
        else:
            new_energy = current_energy

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
