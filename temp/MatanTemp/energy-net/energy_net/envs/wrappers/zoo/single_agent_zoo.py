import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper


class SingleAgentParallelEnvToGymWrapper(BaseParallelWrapper, gym.Env):
    """
    A wrapper for single-agents parallel environments aligning the environments'
    API with OpenAI Gym.
    """
    # gym API class variables
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, env):
        super().__init__(env)

        # assert single agents environment
        assert len(env.possible_agents) == 1

    
    def reset(self, **kwargs):
        # get pettingzoo specific reset arguments
        seed = kwargs.pop('seed', None)  # random seed (also common in gym)
        return_info = kwargs.pop('return_info', False)  # pettingzoo exclusive
        # run `reset` as usual.
        out = self.env.reset(seed=seed,
                             options=kwargs or None)
        
        
       
        

        # check if infos are a part of the reset return
        if out and isinstance(out, tuple):
            obs, infos = out
            
        else:
            obs = out
            infos = {k: {} for k in obs.keys()}

        # return the single entry value as is.
        # no need for the key (only one agents)
        return next(iter(obs.values())), next(iter(infos.values()))

    def step(self, action):
        # step using "joint action" of a single agnet as a dictionary
        step_rets = self.env.step({self.env.agents[0]: action})
        
        
        # unpack step return values from their dictionaries
        return tuple(next(iter(ret.values())) for ret in step_rets)

    @property  # make property for gym-like access
    def action_space(self, _=None):  # ignore second argument in API
        # get action space of the single agents
        return self._flatten_action_space(self.env.action_space(self.env.possible_agents[0]))

    @property  # make property for gym-like access
    def observation_space(self, _=None):  # ignore second argument in API
        # get observation space of the single agents
        return self.env.observation_space(self.env.possible_agents[0])

    def seed(self, seed=None):
        return self.env.seed(seed)
    

    def _flatten_action_space(self, action_space):
        if isinstance(action_space, spaces.Dict):
            flat_action_space = spaces.Box(
                low=np.concatenate([space.low for space in action_space.values()]),
                high=np.concatenate([space.high for space in action_space.values()]),
                dtype=np.float32,
            )
            return flat_action_space
        else:
            return action_space
        

    def action(self, action):
        if isinstance(self.env.action_space, spaces.Dict):
            split_actions = np.split(action, len(self.env.action_space.spaces))
            return {key: split_action for key, split_action in zip(self.env.action_space.spaces.keys(), split_actions)}
        else:
            return action   
        
    @staticmethod
    def _flatten_obs(obs_space):
        if isinstance(obs_space, gym.spaces.Dict):
            flattened_obs_space = gym.spaces.Box(
                low=np.concatenate([space.low for space in obs_space.spaces.values()]),
                high=np.concatenate([space.high for space in obs_space.spaces.values()]),
                dtype=np.float32,
            )
            
            def flatten_dict_obs(obs_dict):
                flattened_obs = np.concatenate([obs_dict[key] for key in obs_space.spaces.keys()])
                return flattened_obs.astype(np.float32)
            
            return flattened_obs_space, flatten_dict_obs
        else:
            return obs_space


class SingleEntityWrapper(SingleAgentParallelEnvToGymWrapper):
    def __init__(self, env):
        super().__init__(env)

        # override environment extra APIs

        # rename original functions
        self.unwrapped.observe_all_ = self.unwrapped.observe_all
        

        # set original name to wrapper overrides
        self.unwrapped.observe_all = self.__observe_all
        

    def __observe_all(self):
        return next(iter(self.unwrapped.observe_all_().values()))

