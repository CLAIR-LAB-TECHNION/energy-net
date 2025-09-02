from energy_net.grid_entity import GridEntity
from energy_net.agents.agent import Agent
from energy_net.model.rewards.base_reward import BaseReward


class StrategicEntity():

    def __init__(self, name, agent: Agent, reward_function: BaseReward):
        """
               Initializes a StrategicEntity instance.
               Args:
                   grid_entity (GridEntity): The associated network entity.
                   agent (Agent): The agent responsible for decision-making.
                   reward_function (RewardFunction): The reward function to evaluate performance.
               """
        self.agent = agent
        self.reward_function = reward_function
        self.name = name


class StrategicGridEntity(StrategicEntity):
    """
    """
    
    def __init__(self, name, agent: Agent, reward_function: BaseReward, grid_entity:GridEntity):
        """
        Initializes a StrategicEntity instance.
        Args:
            agent (Agent): The agent responsible for decision-making.
            reward_function (RewardFunction): The reward function to evaluate performance.
            grid_entity (GridEntity): The associated network entity.
        """
        super().__init__(self, name = name, agent= agent, reward_function= reward_function)
        self.grid_entity = grid_entity
