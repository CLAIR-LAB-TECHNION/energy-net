"""
Reward functions: the agent's learning objective.

Rewards are computed from a StepResult and never change the physics or the
money actually settled. Each reward reports named components so shaping
terms can be tracked separately from money.
"""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Protocol, Sequence

from energy_net.simulation.contracts import StepResult


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: Mapping[str, float] = field(default_factory=dict)


class RewardFunction(Protocol):
    def __call__(self, result: StepResult) -> RewardResult:
        ...


class NetMeteringReward:
    """Reward = money settled under net metering."""

    def __call__(self, result: StepResult) -> RewardResult:
        return RewardResult(result.money, {"money": result.money})


class GridReliancePenalty:
    """
    Fixed penalty in every interval where the grid supplied household load.

    This is reward shaping, not unserved demand: the grid did supply it.
    """

    def __init__(self, weight: float = 1.0):
        weight = float(weight)
        if weight < 0:
            raise ValueError("shortage_penalty must be nonnegative")
        self.weight = weight

    def __call__(self, result: StepResult) -> RewardResult:
        penalty = -self.weight if result.grid_supplied else 0.0
        return RewardResult(penalty, {"grid_reliance": penalty})


class CompositeReward:
    """Sum of several rewards; components are merged by name."""

    def __init__(self, parts: Sequence[RewardFunction]):
        if not parts:
            raise ValueError("CompositeReward needs at least one part")
        self.parts = list(parts)

    def __call__(self, result: StepResult) -> RewardResult:
        total = None
        components: Dict[str, float] = {}
        for part in self.parts:
            outcome = part(result)
            total = outcome.total if total is None else total + outcome.total
            for name, value in outcome.components.items():
                components[name] = components.get(name, 0.0) + value
        return RewardResult(total, components)


def make_reward(include_shortage_penalty: bool = False,
                shortage_penalty: float = 1.0) -> RewardFunction:
    """The two legacy reward variants: plain money, or money minus a penalty."""
    penalty = GridReliancePenalty(shortage_penalty)  # validates the weight
    if not include_shortage_penalty:
        return NetMeteringReward()
    return CompositeReward([NetMeteringReward(), penalty])
