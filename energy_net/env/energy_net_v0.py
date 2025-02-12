
from .EnergyNetEnv import EnergyNetEnv
from energy_net.env.wrappers.order_enforcing_parallel import  OrderEnforcingParallelWrapper


def parallel_env(*args, **kwargs):
    """
    Creates and returns a parallel environment for the EnergyNet simulation.

    This function initializes an instance of the EnergyNetEnv environment with the provided arguments,
    and then wraps it with the OrderEnforcingParallelWrapper to ensure proper order enforcement in a parallel setting.

    Args:
        *args: Variable length argument list to be passed to the EnergyNetEnv constructor.
        **kwargs: Arbitrary keyword arguments to be passed to the EnergyNetEnv constructor.

    Returns:
        OrderEnforcingParallelWrapper: A wrapped instance of EnergyNetEnv that enforces order in a parallel environment.
    """
    p_env = EnergyNetEnv(*args, **kwargs)
    p_env = OrderEnforcingParallelWrapper(p_env)
    return p_env






