from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env_logger import EnvLogger
from typing import Any

class OrderEnforcingParallelWrapper(BaseParallelWrapper):

    def __init__(self, env):
        self._has_reset = False
        super().__init__(env)

    def __getattr__(self, value: str) -> Any:
        """Raises an error message when data is gotten from the env.

        Should only be gotten after reset
        """
        if value == "unwrapped":
            return self.env.unwrapped
        
        elif value == "possible_agents":
            try:
                return self.env.possible_agents
            except AttributeError:
                EnvLogger.error_possible_agents_attribute_missing("possible_agents")
        elif value == "observation_spaces":
            raise AttributeError(
                "The base environment does not have an possible_agents attribute. Use the environments `observation_space` method instead"
            )
        elif value == "action_spaces":
            raise AttributeError(
                "The base environment does not have an possible_agents attribute. Use the environments `action_space` method instead"
            )
        elif value == "agent_order":
            raise AttributeError(
                "agent_order has been removed from the API. Please consider using agent_iter instead."
            )
        elif (
            value
            in {
                "rewards",
                "terminations",
                "truncations",
                "infos",
                "agent_selection",
                "num_agents",
                "agents",
            }
            and not self._has_reset
        ):
            raise AttributeError(f"{value} cannot be accessed before reset")
        else:
            return super().__getattr__(value)

    def step(self, action):
        if not self._has_reset:
            EnvLogger.error_step_before_reset()
        elif not self.agents:
            EnvLogger.warn_step_after_terminated_truncated()
        else:
            return super().step(action)

    def state(self):
        if not self._has_reset:
            EnvLogger.error_state_before_reset()
        return super().state()

    def reset(self, **kwargs):
        self._has_reset = True
        return super().reset(**kwargs)

    def __str__(self):
        str(self.unwrapped)

    def seed(self, seed=None):
        return self.env.seed(seed)