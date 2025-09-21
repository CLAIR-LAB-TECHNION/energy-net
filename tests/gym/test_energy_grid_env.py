import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class MockBattery:
    def __init__(self, energy_min, energy_max, charge_rate_max, discharge_rate_max):
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.charge_rate_max = charge_rate_max
        self.discharge_rate_max = discharge_rate_max
        self.current_energy = (energy_min + energy_max) / 2

    def reset(self):
        self.current_energy = (self.energy_min + self.energy_max) / 2

    def get_state(self):
        return self.current_energy

    def update(self, time, action):
        self.current_energy = np.clip(
            self.current_energy + action, self.energy_min, self.energy_max
        )


class MockDynamics:
    def get_value(self, time):
        return 10.0  # Constant value for simplicity
class RealisticDynamics:
    def __init__(self, base_value, amplitude, frequency, noise_std=0.0):
        """
        Initializes the dynamics model.

        Args:
            base_value (float): The average value of the dynamics.
            amplitude (float): The amplitude of the sinusoidal variation.
            frequency (float): The frequency of the sinusoidal variation.
            noise_std (float): Standard deviation of Gaussian noise to add variability.
        """
        self.base_value = base_value
        self.amplitude = amplitude
        self.frequency = frequency
        self.noise_std = noise_std

    def get_value(self, time):
        """
        Calculates the dynamic value based on time.

        Args:
            time (float): Current time as a fraction of the day (0 to 1).

        Returns:
            float: The dynamic value.
        """
        import numpy as np
        # Sinusoidal variation with optional noise
        value = self.base_value + self.amplitude * np.sin(2 * np.pi * self.frequency * time)
        noise = np.random.normal(0, self.noise_std)
        return max(0, value + noise)  # Ensure non-negative values

class EnergyGridEnv(gym.Env):
    """
    Gymnasium-compatible environment for simulating energy grid dynamics.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, battery, production_dynamics, consumption_dynamics, time_steps=24):
        super().__init__()
        self.battery = battery
        self.production_dynamics = production_dynamics
        self.consumption_dynamics = consumption_dynamics
        self.time_steps = time_steps

        # Define action space: Continuous values for charging/discharging
        self.action_space = spaces.Box(low=-self.battery.discharge_rate_max,
                                       high=self.battery.charge_rate_max,
                                       shape=(1,), dtype=np.float32)

        # Define observation space: Battery level, production, consumption, and time
        self.observation_space = spaces.Dict({
            "battery_level": spaces.Box(low=self.battery.energy_min,
                                        high=self.battery.energy_max,
                                        shape=(1,), dtype=np.float32),
            "production": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "consumption": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "time": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        self.current_time = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.battery.reset()
        self.state = {
            "battery_level": np.array([self.battery.get_state()], dtype=np.float32),
            "production": np.array([0.0], dtype=np.float32),
            "consumption": np.array([0.0], dtype=np.float32),
            "time": np.array([0.0], dtype=np.float32),
        }
        return self.state, {}

    def step(self, action):
        time_fraction = self.current_time / self.time_steps
        production = self.production_dynamics.get_value(time=time_fraction)
        consumption = self.consumption_dynamics.get_value(time=time_fraction)

        self.battery.update(time=time_fraction, action=action[0])
        battery_level = self.battery.get_state()

        net_energy = production - consumption
        reward = -abs(net_energy)  # Penalize imbalance
        if battery_level < self.battery.energy_min or battery_level > self.battery.energy_max:
            reward -= 1000  # Heavy penalty for exceeding battery limits
        else:
            reward += 10  # Small reward for staying within limits

        self.state = {
            "battery_level": np.array([battery_level], dtype=np.float32),
            "production": np.array([production], dtype=np.float32),
            "consumption": np.array([consumption], dtype=np.float32),
            "time": np.array([time_fraction], dtype=np.float32),
        }

        self.current_time += 1
        done = self.current_time >= self.time_steps

        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        print(f"Time: {self.state['time'][0]:.2f}, "
              f"Battery: {self.state['battery_level'][0]:.2f} MWh, "
              f"Production: {self.state['production'][0]:.2f} MW, "
              f"Consumption: {self.state['consumption'][0]:.2f} MW")

    def close(self):
        pass


# Train an RL agent (PPO)
def train_rl_agent(env):
    vec_env = make_vec_env(lambda: env, n_envs=1)  # Vectorized environment for stable-baselines3
    model = PPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=10000)  # Train for 10,000 timesteps
    return model


# Evaluate an agent
def evaluate_agent(env, agent=None, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if agent is None:
                # Random agent
                action = env.action_space.sample()
            else:
                # RL agent
                action, _ = agent.predict(state)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


if __name__ == "__main__":
    # Initialize the environment with realistic dynamics
    battery = MockBattery(energy_min=0, energy_max=100, charge_rate_max=10, discharge_rate_max=10)
    production_dynamics = RealisticDynamics(base_value=50, amplitude=20, frequency=1, noise_std=5)
    consumption_dynamics = RealisticDynamics(base_value=40, amplitude=15, frequency=1, noise_std=5)
    env = EnergyGridEnv(battery=battery, production_dynamics=production_dynamics, consumption_dynamics=consumption_dynamics)

    # Train and evaluate
    rl_agent = train_rl_agent(env)
    random_agent_reward = evaluate_agent(env, agent=None)
    rl_agent_reward = evaluate_agent(env, agent=rl_agent)

    print(f"Average reward of random agent: {random_agent_reward}")
    print(f"Average reward of RL agent: {rl_agent_reward}")