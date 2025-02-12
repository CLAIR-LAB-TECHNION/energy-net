from abc import abstractmethod

from energy_net.components import pcsunit
from energy_net.model.state import State
from energy_net.model.action import EnergyAction
from energy_net.rewards.base_reward import BaseReward
from energy_net.defs import Bid
from energy_net.components.grid_entity import GridEntity
from energy_net.stratigic_entity import StrategicEntity


class MarketEntity(StrategicEntity):
    def __init__(self, name, grid_entity:GridEntity):
        self.name = name
        self.network_entity = grid_entity

    @abstractmethod
    def step(self, action: EnergyAction) -> [State, BaseReward]:
        return self.network_entity.step(action)

    def predict(self, action: EnergyAction, state: State) -> [State, BaseReward]:
        return self.network_entity.predict(action, state)

class ControlledProducer(MarketEntity):
    """
    Producer that can only produce electricity.
    """
    def __init__(self, prodcution_unit, predicted_demand, predicted_prices, production_capacity=100):
        self.production_capacity = production_capacity
        self.predicted_demand = predicted_demand
        self.predicted_prices = predicted_prices
        self.prodcution_unit  = prodcution_unit

    def decide_action(self, timestamp):
        # Produce up to the production capacity
        production = min(self.production_capacity, self.predicted_demand[timestamp])
        return production

class MarketStorage(MarketEntity):
    """
    Storage that can store and release energy.
    """
    def __init__(self, predicted_demand, predicted_prices, storage_capacity=30, initial_storage=0, charge_rate=30, discharge_rate=30):
        self.storage_capacity = storage_capacity
        self.current_storage = initial_storage
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.predicted_demand = predicted_demand
        self.predicted_prices = predicted_prices

        # Determine when to charge and discharge
        self.plan_actions()

    def plan_actions(self):
        # Find the index of minimum price (time to charge)
        self.charge_time = self.predicted_prices.index(min(self.predicted_prices))
        # Find the index of maximum price (time to discharge)
        self.discharge_time = self.predicted_prices.index(max(self.predicted_prices))

        # Create an action schedule over time
        self.action_schedule = [0]*len(self.predicted_prices)
        # At charge_time, plan to charge
        self.action_schedule[self.charge_time] = -self.charge_rate  # Negative means consume energy
        # At discharge_time, plan to discharge
        self.action_schedule[self.discharge_time] = self.discharge_rate  # Positive means produce energy

    def decide_action(self, timestamp):
        """
        Decide action based on precomputed schedule.
        """
        action = self.action_schedule[timestamp]

        # Update storage based on action
        if action < 0:
            # Charging
            amount_to_charge = min(-action, self.storage_capacity - self.current_storage)
            self.current_storage += amount_to_charge
            return -amount_to_charge  # Return negative consumption
        elif action > 0:
            # Discharging
            amount_to_discharge = min(action, self.current_storage)
            self.current_storage -= amount_to_discharge
            return amount_to_discharge  # Return positive production
        else:
            # No action
            return 0

class NonControlledMarketEntity(MarketEntity):

    def __init__(self,PCSUnit:pcsunit):
        self.pcsunit = pcsunit

