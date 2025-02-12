from energy_net.market_entity import MarketEntity
from energy_net.model.state import State


class SimpleMarketManager():
    def __init__(self, name: str, market):
        self.mame = name
        self.market = market

    def step(self, cur_state:State):
        market_results = self.market.step(cur_state)
        return market_results



class NetworkManager:
    def __init__(self, market_entities:list[MarketEntity]):
        self.market_entities = market_entities

    def do_market_clearing(self, state:State):
        demand = self.collect_demand(state)
        bids = self.collect_production_bids(state, demand)
        workloads, price = self.market_clearing_merit_order(demand, bids)
        return [demand,bids,workloads,price]

    def collect_demand(self, state:State)->float:
        total_demand = 0
        for ma in self.market_entities:
            cur_prediction = ma.predict({'consumption': {}}, state)
            if cur_prediction:
                total_demand += cur_prediction
        return total_demand

    def collect_production_bids(self, state:State, demand:float) -> dict[str, Bid]:
        bids = {}
        for ma in self.market_entities:
            bid = ma.get_bid('production',state, demand)
            if bid:
                bids[ma.name] = bid

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


    def run(self, initial_state:State, stop_criteria:condition, horizons:list[float]=[24,48]):
        cur_state = initial_state
        while not stop_criteria(cur_state):
            for horizon in horizons:
                predicted_state = get_predicted_state(cur_state,horizon)
                [demand, bids, workloads, price] = self.do_market_clearing(predicted_state)
                #check solution validity

                #send solution



