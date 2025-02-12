from copy import deepcopy

from energy_net.market_entity import MarketEntity
from energy_net.model.action import ConsumeAction
from energy_net.model.state import State
from energy_net.defs import Bid

class NDAMarket():
    def __init__(self, production_entities:list[MarketEntity], consumption_entities:list[MarketEntity], horizons:list[float] = [24, 48], intervals:list[float] = [0.5,0.5]):
        self.production_entities = production_entities
        self.consumption_entities = consumption_entities
        self.horizons = horizons
        self.intervals = intervals


    def step(self, cur_state: State):
        market_results = {}
        for horizon in self.horizons:
            try:
                future_state = cur_state.get_timedelta_state(delta_hours=horizon)
            except TypeError:
                future_state = deepcopy(cur_state)
            [demand, bids, workloads, price] = self.do_market_clearing(future_state)
            market_results[horizon] = [demand, bids, workloads, price]

        return market_results

    def do_market_clearing(self, state:State):
        demand = self.collect_demand(state)
        bids = self.collect_production_bids(state, demand)
        workloads, price = self.market_clearing_merit_order(demand, bids)
        return [demand,bids,workloads,price]

    def collect_demand(self, future_state:State)->float:
        total_demand = 0
        for consumer in self.consumption_entities:
            cur_prediction = consumer.predict(state=future_state, action=ConsumeAction)
            if cur_prediction:
                total_demand += cur_prediction
        return total_demand

    def collect_production_bids(self, state:State, demand:float) -> dict[str, Bid]:
        bids = {}
        for producer in self.production_entities:
            bid = producer.get_bid('production', state, demand)
            if bid:
                bids[producer.name] = bid

        return bids

    def dispatch(self, consumption_demand, bids)-> tuple[dict[MarketEntity,float], float]:
        sorted_bidders = sorted(bids.keys(), key=lambda k: bids[k][1])
        workloads = {}
        last_bid = 0
        for bidder in sorted_bidders:
            workloads[bidder] = min(bids[bidder][0], consumption_demand)
            consumption_demand -= workloads[bidder]
            last_bid = bids[bidder][1]
            if consumption_demand == 0:
                break

        return [workloads, last_bid]

    def set_price(self, workloads, last_bid):
        return last_bid

    def market_clearing(self, method: str, consumption_demand, bids):
        if method == 'merit_order':
            return self.market_clearing_merit_order(consumption_demand, bids)
        else:
            raise NotImplementedError

    def market_clearing_merit_order(self, consumption_demand, bids):
        workloads, last_bid = self.dispatch(consumption_demand, bids)
        price = self.set_price(workloads, last_bid)
        return workloads, price



    def step(self, cur_state:State):
        market_results = self.market.step(cur_state)
        return market_results

