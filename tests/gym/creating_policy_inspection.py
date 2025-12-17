import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from PCSGymEnv import PCSGymEnv

# -----------------------------
# Environment creation
# -----------------------------
def make_env():
    """Create one environment for testing - MATCHES PCSGymEnv parameters"""
    return PCSGymEnv(
        data_file='synthetic_household_consumption.csv',
        dt=0.5 / 24,  # 30 minutes in days
        episode_length_days=1,
        train_test_split=0.8,  # 80% train, 20% test
        prediction_horizon=48,
        shortage_penalty=5.0,
        base_price=0.10,
        price_volatility=0.15
    )

# -----------------------------
# Policy rollout
# -----------------------------
def rollout_policy(model, env, n_episodes=3):
    trajectories = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        traj = {"obs": [], "action": [], "reward": [], "info": []}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Ensure correct batch handling
            if isinstance(env, DummyVecEnv) and action.ndim == 1:
                action = action.reshape(1, -1)
            obs_new, reward, done, info = env.step(action)

            traj["obs"].append(obs.copy())
            traj["action"].append(action.copy())
            traj["reward"].append(float(reward[0] if isinstance(reward, np.ndarray) else reward))
            traj["info"].append(info)

            obs = obs_new

        trajectories.append(traj)
    return trajectories

# -----------------------------
# Save trajectories to CSV
# -----------------------------
def save_trajectories_csv(trajectories, model_name):
    os.makedirs("policy_inspection", exist_ok=True)
    for ep_idx, traj in enumerate(trajectories):
        data = []
        for t, (obs, act, rew, info) in enumerate(zip(traj["obs"], traj["action"], traj["reward"], traj["info"])):
            data.append({
                "t": t,
                "obs": obs.tolist() if isinstance(obs, np.ndarray) else obs,
                "action": act.tolist() if isinstance(act, np.ndarray) else act,
                "reward": rew,
                "info": info
            })
        df = pd.DataFrame(data)
        file_path = f"policy_inspection/{model_name}_episode{ep_idx+1}.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved {file_path}")

# -----------------------------
# Plot actions & rewards
# -----------------------------
def plot_actions_rewards(trajectories, model_name):
    plt.figure(figsize=(12, 6))
    for ep_idx, traj in enumerate(trajectories):
        actions = [a[0] if isinstance(a, np.ndarray) else a for a in traj["action"]]
        rewards = traj["reward"]

        plt.subplot(2, 1, 1)
        plt.plot(actions, label=f"Episode {ep_idx+1}")
        plt.ylabel("Action")
        plt.title(f"{model_name} Actions over Time")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(rewards, label=f"Episode {ep_idx+1}")
        plt.ylabel("Reward")
        plt.xlabel("Step")
        plt.title(f"{model_name} Rewards over Time")
        plt.legend()

    plt.tight_layout()
    os.makedirs("policy_inspection", exist_ok=True)
    plot_path = f"policy_inspection/{model_name}_actions_rewards.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    # Make environment
    env = DummyVecEnv([make_env])

    # Models to inspect
    model_files = {
        "PPO": "models/PPO",
        "A2C": "models/A2C",
        "SAC": "models/SAC",
        "TD3": "models/TD3"
    }

    # Mapping names to classes
    model_classes = {
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC,
        "TD3": TD3
    }

    for model_name, path in model_files.items():
        if not os.path.exists(f"{path}.zip"):
            print(f"Model {path} not found, skipping {model_name}")
            continue

        print(f"\n=== Inspecting {model_name} ===")
        model_class = model_classes.get(model_name)
        model = model_class.load(path, env=env)

        trajectories = rollout_policy(model, env, n_episodes=3)
        save_trajectories_csv(trajectories, model_name)
        plot_actions_rewards(trajectories, model_name)

if __name__ == "__main__":
    main()
