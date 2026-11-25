"""
Gymnasium adapter for the prosumer simulator.

This module only converts between gym and the simulation: it decodes the
action, calls the simulator, evaluates the injected reward function and
encodes the observation. All physics, data handling, tariffs and
settlement live in ``energy_net.simulation``.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from energy_net.gym_envs.prosumer_observation import ProsumerObservationEncoder
from energy_net.gym_envs.prosumer_render import RenderView, render_prosumer
from energy_net.simulation import (
    ProsumerSimulator,
    RewardFunction,
    make_prosumer_simulator,
    make_reward,
)


class ProsumerEnv(gym.Env):
    """
    One prosumer (battery + load + generator) as a gym environment.

    The action is one value in ``[-action_bound, action_bound]``; the
    battery command is ``action * action_scale`` (positive charges).
    See ``observation_layout`` for the observation vector's named slices.

    Args:
        simulator: A ProsumerSimulator. Built with
            ``make_prosumer_simulator(**simulator_kwargs)`` when omitted.
        reward_fn: Reward function of a StepResult. Built with
            ``make_reward(include_shortage_penalty, shortage_penalty)`` when
            omitted.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, simulator: Optional[ProsumerSimulator] = None,
                 reward_fn: Optional[RewardFunction] = None, *,
                 action_scale: float = 1.0,
                 action_bound: float = 10.0,
                 render_mode: Optional[str] = None,
                 verbosity: int = 2,
                 include_shortage_penalty: bool = False,
                 shortage_penalty: float = 1.0,
                 **simulator_kwargs):
        super().__init__()
        # render_mode='human' maps to verbosity 2.
        self.verbosity = 2 if render_mode == "human" else verbosity
        self.render_mode = render_mode

        if simulator is None:
            simulator = make_prosumer_simulator(verbosity=self.verbosity, **simulator_kwargs)
        elif simulator_kwargs:
            raise ValueError("Pass either a simulator or the arguments to build one, not both")
        self.simulator = simulator
        self.reward_fn = reward_fn or make_reward(include_shortage_penalty, shortage_penalty)

        self.action_scale = float(action_scale)
        self.action_bound = float(action_bound)
        self.action_space = spaces.Box(
            low=-self.action_bound, high=self.action_bound, shape=(1,), dtype=np.float32
        )
        self.encoder = ProsumerObservationEncoder(
            simulator.forecast_horizon, storage_scale=simulator.prosumer.storage.energy_max
        )
        self.observation_space = self.encoder.space
        self.observation_layout = self.encoder.layout

        self.last_action: Optional[float] = None
        self._decision_obs: Optional[np.ndarray] = None
        self._last_price: Optional[float] = None
        self._reset_reward_totals()

        if self.verbosity > 0:
            print(f"Observation space dimension: {self.encoder.size}")

    # ------------------------------------------------------------ helpers

    def _reset_reward_totals(self):
        self.total_reward = 0.0
        self.total_shortage_penalty = 0.0

    def _encode_decision(self) -> np.ndarray:
        observation = self.simulator.observe()
        self._last_price = observation.buy_tariff
        self._decision_obs = self.encoder.encode(observation)
        return self._decision_obs.copy()

    @property
    def current_step(self) -> int:
        return self.simulator.step_index

    @property
    def current_datetime(self):
        return self.simulator.current_time

    @property
    def max_steps(self) -> int:
        return self.simulator.episode_steps

    @property
    def dt(self) -> float:
        """Interval length in days."""
        return self.simulator.dt.total_seconds() / 86400.0

    # ------------------------------------------------------------ gym API

    def reset(self, *, seed=None, options=None):
        """
        Options: ``start_date`` and ``initial_storage`` (otherwise the
        battery's configured level).

        Without ``start_date`` the episode continues from where the last one
        ended, except when a ``seed`` is given: then it starts from the
        beginning of the data, so a seeded reset is reproducible.
        """
        super().reset(seed=seed)
        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a dictionary or None")
        options = dict(options or {})
        start = options.pop("start_date", None)
        initial_storage = options.pop("initial_storage", None)
        if options:
            raise ValueError(f"Unknown reset options: {sorted(options)}")

        self.simulator.reset(start, initial_storage=initial_storage,
                             rng=self.np_random, rewind=seed is not None)
        self.last_action = None
        self._reset_reward_totals()
        return self._encode_decision(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 1:
            raise ValueError(f"Expected a single battery action, got {action.size} values")
        agent_action = float(action[0])
        command = agent_action * self.action_scale

        decision_obs = self._decision_obs
        result = self.simulator.step(command)
        reward_result = self.reward_fn(result)
        reward = reward_result.total
        shaping = result.money - reward
        self.last_action = command
        self.total_reward += reward
        self.total_shortage_penalty += shaping

        if result.terminated:
            observation = self.encoder.terminal()
            self._decision_obs = None
        else:
            observation = self._encode_decision()

        info = result.to_info()
        info.update({
            "agent_action": agent_action,
            "observation_at_decision": decision_obs.tolist(),
            "step_reward": reward,
            "reward_components": dict(reward_result.components),
            "shortage_penalty": shaping,
            "total_reward_so_far": self.total_reward,
            "total_shortage_penalty_so_far": self.total_shortage_penalty,
        })
        return observation, reward, result.terminated, False, info

    def set_price_strategy(self, strategy):
        """
        Replace the pricing strategy. During an episode, return the
        refreshed decision observation, which must replace the previous one.
        """
        if self.simulator.set_price_strategy(strategy):
            return self._encode_decision()
        return None

    def ground_truth(self, unit: str, times):
        """Privileged query of what ``unit`` ('load' or 'generator') realizes."""
        return self.simulator.ground_truth(unit, times)

    def render(self, action=None, verbosity=None):
        simulator = self.simulator
        prosumer = simulator.prosumer
        metrics = simulator.metrics
        snapshot = simulator.observe() if simulator.is_active else None
        display_action = (
            float(np.asarray(action).reshape(-1)[0]) * self.action_scale
            if action is not None else self.last_action
        )
        view = RenderView(
            step=simulator.step_index,
            max_steps=simulator.episode_steps,
            datetime=simulator.current_time,
            dt_days=self.dt,
            storage=prosumer.storage_level(),
            capacity=prosumer.storage.energy_max,
            consumption=prosumer.load.get_state(),
            generation=prosumer.generator.get_state(),
            price=snapshot.buy_tariff if snapshot is not None else self._last_price,
            strategy=type(simulator.price_strategy).__name__,
            last_action=self.last_action,
            display_action=display_action,
            action_bound=self.action_bound,
            total_money=metrics.total_money,
            total_penalty=self.total_shortage_penalty,
            total_reward=self.total_reward,
            shortage_count=metrics.grid_supplied_count,
            available_charge=prosumer.available_charge(),
            available_discharge=prosumer.available_discharge(),
            features=snapshot.calendar_dict() if snapshot is not None else {},
            next_load_forecasts=tuple(snapshot.load_forecast[:5]) if snapshot is not None else (),
        )
        return render_prosumer(view, self.verbosity if verbosity is None else verbosity)
