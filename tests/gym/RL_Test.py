import gym
from gym import spaces
import numpy as np
import pandas as pd
from energy_net.grid_entities.consumption.consumption_dynamics import CSV_DataConsumptionDynamics
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.production.production_unit import ProductionUnit
from energy_net.grid_entities.production.production_dynamics import GMMProductionDynamics
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit


class PCSGymEnv(gym.Env):
    """
    Generic Gym wrapper for ANY PCSUnit instance.
    You fully control how PCS is constructed.
    """

    def __init__(self, pcs_unit, dt=1.0 / 24):
        super().__init__()

        # --- New battery ---
        battery_dynamics = DeterministicBattery(model_parameters={"charge_efficiency": 0.95,
                                                                      "discharge_efficiency": 0.9,
                                                                      "lifetime_constant": 1e6})
        battery_config = {
            "min": 0.0,
            "max": 2e6,
            "charge_rate_max": 2e5,
            "discharge_rate_max": 2e5,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.9,
            "init": 1e6
        }
        battery = Battery(dynamics=battery_dynamics, config=battery_config)

        # --- New consumption unit ---
        data_file = 'SystemDemand_30min_2023-2025.csv'
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if df.empty:
            raise ValueError(f"The data file {data_file} is empty or could not be loaded.")

        consumption_dynamics = CSV_DataConsumptionDynamics(params={'data_file': data_file})
        consumption_unit_config = {
            'data_file': data_file,
            'consumption_capacity': 12000.0
        }
        consumption_unit = ConsumptionUnit(dynamics=consumption_dynamics, config=consumption_unit_config)


        # --- New production unit ---
        production_dynamics = GMMProductionDynamics(params={
            "peak_production1": 1500.0,
            "peak_time1": 0.3,
            "width1": 0.05,
            "peak_production2": 2000.0,
            "peak_time2": 0.8,
            "width2": 0.1
        })
        production_unit = ProductionUnit(
            dynamics=production_dynamics,
            config={"production_capacity": 8000.0}
        )

        # --- Build PCSUnit ---
        pcs_unit = PCSUnit(
            storage_units=[battery],
            production_units=[production_unit],
            consumption_units=[consumption_unit]
        )


        self.pcs = pcs_unit
        self.dt = dt
        self.time = 0.0

        # ✅ ONLY control battery for now
        self.action_space = spaces.Box(
            low=-1e5,
            high=1e5,
            shape=(1,),
            dtype=np.float32
        )

        # ✅ Observation = [storage, production, consumption]
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

    def reset(self):
        # IMPORTANT: user recreates PCS manually outside
        self.time = 0.0
        return self._get_obs()

    def step(self, action):
        battery_action = float(action)

        actions = {
            "Battery_0": battery_action,
        }

        self.pcs.update(state=self.time, actions=actions)
        self.time += self.dt

        obs = self._get_obs()
        reward = -abs(self.pcs.get_energy_change())
        done = self.time >= 1.0
        info = {}

        return obs, reward, done, info

    def _get_obs(self):
        return np.array([
            self.pcs.get_total_storage(),
            self.pcs.get_production(),
            self.pcs.get_consumption(),
        ], dtype=np.float32)

