import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import os
from typing import Dict, List, Tuple
import pandas as pd
from energy_net.gym_envs.pcs_env import PCSEnv

# -----------------------------
# Environment creation
# -----------------------------
def make_env():
    """
    Create one environment for testing.
    Updated to match the new PCSGymEnv parameter names.
    MATCHES the updated PCSGymEnv parameters.
    """
    return PCSEnv(
        test_data_file='../../energy_net/gym_envs/data_for_tests/synthetic_household_consumption_test.csv',
        predictions_file='../../energy_net/gym_envs/data_for_tests/consumption_predictions.csv',
        dt=0.5 / 24,  # 30 minutes in days
        episode_length_days=1,
        prediction_horizon=48,
        shortage_penalty=5.0,
        base_price=0.10,
        price_volatility=0.15,
        log_path = 'logs'
    )


# -----------------------------
# Agent Classes & Evaluation (File 1)
# -----------------------------
class RandomAgent:
    """Random baseline agent adapted for the [-10, 10] action space"""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        action = self.action_space.sample()
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        elif action.ndim == 0:
            action = np.array([action])
        return action, None


def evaluate_agent(agent, env, n_episodes=10) -> Tuple[float, float, List[float]]:
    """
    Evaluate an agent over multiple episodes.
    Stable Baselines3 DummyVecEnv handles the Gymnasium 5-tuple
    conversion back to the 4-tuple (obs, reward, done, info).
    """
    episode_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            # Add batch dimension for VecEnv
            if action.ndim == 1:
                action = action.reshape(1, -1)

            obs, reward, done, info = env.step(action)
            # Reward is returned as a numpy array [reward]
            total_reward += reward[0]

        episode_rewards.append(total_reward)

    return np.mean(episode_rewards), np.std(episode_rewards), episode_rewards


def train_and_evaluate_algorithm(
        algo_class,
        algo_name: str,
        env,
        training_steps: int,
        eval_episodes: int = 10,
        **algo_kwargs
) -> Dict:
    """Train and evaluate an RL algorithm"""
    print(f"\n{'=' * 60}")
    print(f"Training {algo_name}...")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        os.makedirs("models", exist_ok=True)

        model = algo_class(
            "MlpPolicy",
            env,
            verbose=0,
            **algo_kwargs
        )

        model.learn(total_timesteps=training_steps)
        training_time = time.time() - start_time

        model_path = f"models/{algo_name}"
        model.save(model_path)

        mean_reward, std_reward, episode_rewards = evaluate_agent(
            model, env, n_episodes=eval_episodes
        )

        print(f"✓ {algo_name} completed:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

        return {
            'name': algo_name,
            'model_path': model_path,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'episode_rewards': episode_rewards,
            'training_time': training_time,
            'success': True
        }

    except Exception as e:
        print(f"✗ {algo_name} failed: {str(e)}")
        return {
            'name': algo_name,
            'success': False,
            'mean_reward': -np.inf,
            'error': str(e)
        }


# -----------------------------
# Policy rollout & Inspection (File 2)
# -----------------------------
def rollout_policy(model, env, n_episodes=3):
    """Rollout the model and capture the detailed info from the new environment."""
    trajectories = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        # We use 'info' to track internal state like storage levels and price
        traj = {"obs": [], "action": [], "reward": [], "info": []}

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # DummyVecEnv expects and returns batched data
            if action.ndim == 1:
                action = action.reshape(1, -1)

            obs_new, reward, done, info = env.step(action)

            traj["obs"].append(obs.copy())
            traj["action"].append(action.copy())
            # reward is a numpy array [val]
            traj["reward"].append(float(reward[0]))
            # info is a list of dicts: [ {info_env_1} ]
            traj["info"].append(info[0])

            obs = obs_new

        trajectories.append(traj)
    return trajectories


def save_trajectories_csv(trajectories, model_name):
    """Saves flattened trajectory data including specific PCS metrics."""
    os.makedirs("policy_inspection", exist_ok=True)

    for ep_idx, traj in enumerate(trajectories):
        rows = []
        for t in range(len(traj["reward"])):
            info = traj["info"][t]

            # Combine basic step data with the detailed 'info' dict from PCSGymEnv
            row = {
                "step": t,
                "reward": traj["reward"][t],
                "action_raw": traj["action"][t][0][0],
                "storage_after": info.get('storage_after_units'),
                "consumption": info.get('consumption_units'),
                "energy_moved": info.get('energy_sold_or_bought_units'),
                "is_shortage": info.get('shortage'),
                "step_money": info.get('step_money')
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        file_path = f"policy_inspection/{model_name}_episode{ep_idx + 1}.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved CSV: {file_path}")


# -----------------------------
# Visualization & Plotting
# -----------------------------
def plot_results(results: List[Dict], save_path='algorithm_comparison.png'):
    """Create visualization of algorithm comparison (from File 1)"""
    successful_results = [r for r in results if r['success']]

    if not successful_results:
        print("No successful results to plot!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    names = [r['name'] for r in successful_results]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(names)))

    # 1. Mean reward comparison
    ax1 = axes[0, 0]
    means = [r['mean_reward'] for r in successful_results]
    stds = [r['std_reward'] for r in successful_results]
    bars = ax1.bar(range(len(names)), means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45)
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Performance Comparison')

    # 2. Training time
    ax2 = axes[0, 1]
    times = [r['training_time'] for r in successful_results]
    ax2.bar(range(len(names)), times, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45)
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Training Time')

    # 3. Episode reward distribution
    ax3 = axes[1, 0]
    episode_data = [r['episode_rewards'] for r in successful_results]
    ax3.boxplot(episode_data, tick_labels=names)
    ax3.set_xticklabels(names, rotation=45)
    ax3.set_ylabel('Episode Reward')
    ax3.set_title('Reward Distribution')

    # 4. Ranking table
    ax4 = axes[1, 1]
    ax4.axis('off')
    sorted_results = sorted(successful_results, key=lambda x: x['mean_reward'], reverse=True)
    table_data = [[r['name'], f"{r['mean_reward']:.1f}", f"{r['training_time']:.1f}s"] for r in sorted_results]
    ax4.table(cellText=table_data, colLabels=['Algo', 'Mean Reward', 'Time'], loc='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Results saved to {save_path}")


def plot_actions_rewards(trajectories, model_name):
    """Creates a 3-panel plot to see rewards, actions, and storage levels (from File 2)."""
    plt.figure(figsize=(12, 10))

    for ep_idx, traj in enumerate(trajectories):
        steps = range(len(traj["reward"]))
        actions = [a[0][0] for a in traj["action"]]
        rewards = traj["reward"]
        storage = [info.get('storage_after_units') for info in traj["info"]]

        # 1. Actions Panel
        plt.subplot(3, 1, 1)
        plt.plot(steps, actions, label=f"Ep {ep_idx + 1}")
        plt.ylabel("Action (-10 to 10)")
        plt.title(f"{model_name} Policy Inspection")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # 2. Rewards Panel
        plt.subplot(3, 1, 2)
        plt.plot(steps, rewards, label=f"Ep {ep_idx + 1}")
        plt.ylabel("Step Reward ($)")
        plt.grid(True, alpha=0.3)

        # 3. Storage Level Panel (Crucial for PCS logic)
        plt.subplot(3, 1, 3)
        plt.plot(steps, storage, label=f"Ep {ep_idx + 1}")
        plt.ylabel("Battery Level")
        plt.xlabel("Timesteps (30-min intervals)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("policy_inspection", exist_ok=True)
    plot_path = f"policy_inspection/{model_name}_analysis.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved Plot: {plot_path}")


# -----------------------------
# Integrated Main
# -----------------------------
def main():
    print("=" * 60)
    print("RL ALGORITHM TRAINING AND DETAILED INSPECTION")
    print("=" * 60)

    # 1. Verify data files exist (Check from File 2)
    # Note: Paths adjusted to match the make_env definition
    data_files = [
        '../../energy_net/gym_envs/data_for_tests/synthetic_household_consumption_test.csv',
        '../../energy_net/gym_envs/data_for_tests/consumption_predictions.csv'
    ]
    for f in data_files:
        if not os.path.exists(f):
            print(f"Warning: Data file not found at {f}. Ensure paths are correct relative to execution.")

    # 2. Initialize environment
    env = DummyVecEnv([make_env])

    TRAINING_STEPS = 15000
    EVAL_EPISODES = 10

    algorithms = [
        {'class': PPO, 'name': 'PPO', 'kwargs': {'learning_rate': 3e-4}},
        {'class': A2C, 'name': 'A2C', 'kwargs': {'learning_rate': 7e-4}},
        {'class': SAC, 'name': 'SAC', 'kwargs': {'learning_rate': 3e-4, 'buffer_size': 20000}},
        {'class': TD3, 'name': 'TD3', 'kwargs': {'learning_rate': 3e-4, 'buffer_size': 20000}}
    ]

    results = []

    # 3. Baseline Evaluation
    print("\nEvaluating Random Agent...")
    temp_env = make_env()
    random_agent = RandomAgent(temp_env.action_space)
    mean_r, std_r, ep_rs = evaluate_agent(random_agent, env, n_episodes=EVAL_EPISODES)
    results.append({
        'name': 'Random', 'mean_reward': mean_r, 'std_reward': std_r,
        'episode_rewards': ep_rs, 'training_time': 0, 'success': True
    })

    # 4. RL Training Loop
    for algo in algorithms:
        res = train_and_evaluate_algorithm(
            algo['class'], algo['name'], env, TRAINING_STEPS, EVAL_EPISODES, **algo['kwargs']
        )
        results.append(res)

    # 5. Summary and Plotting Comparison (File 1 functionality)
    plot_results(results)

    # 6. Detailed Policy Inspection (File 2 functionality)
    print("\n" + "=" * 60)
    print("RUNNING DETAILED POLICY INSPECTION")
    print("=" * 60)

    for res in results:
        if res['name'] == 'Random' or not res['success']:
            continue

        model_name = res['name']
        model_path = res['model_path']

        print(f"\n>>> Analyzing {model_name}...")
        try:
            # Find the correct class for loading
            algo_entry = next(a for a in algorithms if a['name'] == model_name)
            model_class = algo_entry['class']
            model = model_class.load(model_path, env=env)

            # Generate 3 test episodes for inspection
            trajectories = rollout_policy(model, env, n_episodes=3)

            # Export data and plots
            save_trajectories_csv(trajectories, model_name)
            plot_actions_rewards(trajectories, model_name)

        except Exception as e:
            print(f"Failed to inspect {model_name}: {e}")

    # Final Output Table
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'episode_rewards'} for r in results])
    print("\nFinal Results Table:")
    print(df.sort_values(by='mean_reward', ascending=False))


if __name__ == "__main__":
    main()