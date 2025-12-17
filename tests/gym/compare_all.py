import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import time
from typing import Dict, List, Tuple
import pandas as pd
from PCSGymEnv import PCSGymEnv


def make_env():
    """Create one environment for testing"""
    return PCSGymEnv(
        data_file='synthetic_household_consumption.csv',
        dt=0.5 / 24,  # 30 minutes in days
        episode_length_days=1,
        train_test_split=0.8,  # 80% train, 20% test
        prediction_horizon=48,  # Number of future time steps to predict
        shortage_penalty=5.0,  # ✅ REDUCED: from 20.0 to make positive rewards achievable
        base_price=0.10,  # Base electricity price $/kWh
        price_volatility=0.15  # ✅ INCREASED: from 0.05 for more trading opportunities
    )


class RandomAgent:
    """Random baseline agent"""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        action = self.action_space.sample()
        # Ensure action is always returned as array for consistency
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        elif action.ndim == 0:  # scalar ndarray
            action = np.array([action])
        return action, None


def evaluate_agent(agent, env, n_episodes=10, agent_name="Agent") -> Tuple[float, float, List[float]]:
    """
    Evaluate an agent over multiple episodes.
    Returns: (mean_reward, std_reward, episode_rewards)
    """
    episode_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            # DummyVecEnv expects actions with shape (n_envs, action_dim)
            # So we need to add the batch dimension
            if action.ndim == 1:
                action = action.reshape(1, -1)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward

        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, episode_rewards


def train_and_evaluate_algorithm(
        algo_class,
        algo_name: str,
        env,
        training_steps: int,
        eval_episodes: int = 10,
        **algo_kwargs
) -> Dict:
    """
    Train an algorithm and evaluate it.
    Returns dictionary with results.
    """
    print(f"\n{'=' * 60}")
    print(f"Training {algo_name}...")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Create model
        model = algo_class(
            "MlpPolicy",
            env,
            verbose=0,
            **algo_kwargs
        )

        # ---- TRAIN ONCE ----
        model.learn(total_timesteps=training_steps)
        training_time = time.time() - start_time

        # ---- SAVE MODEL ----
        model_path = f"models/{algo_name}"
        model.save(model_path)

        # ---- EVALUATE ----
        mean_reward, std_reward, episode_rewards = evaluate_agent(
            model, env, n_episodes=eval_episodes, agent_name=algo_name
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
            'model_path': None,
            'mean_reward': -np.inf,
            'std_reward': 0,
            'episode_rewards': [],
            'training_time': 0,
            'success': False,
            'error': str(e)
        }


def plot_results(results: List[Dict], save_path='algorithm_comparison.png'):
    """Create visualization of algorithm comparison"""
    # Filter successful results
    successful_results = [r for r in results if r['success']]

    if not successful_results:
        print("No successful results to plot!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Mean reward comparison with error bars
    ax1 = axes[0, 0]
    names = [r['name'] for r in successful_results]
    means = [r['mean_reward'] for r in successful_results]
    stds = [r['std_reward'] for r in successful_results]

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(names)))
    bars = ax1.bar(range(len(names)), means, yerr=stds, capsize=5,
                   color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{mean:.0f}',
                 ha='center', va='bottom', fontsize=9)

    # 2. Training time comparison
    ax2 = axes[0, 1]
    times = [r['training_time'] for r in successful_results]
    bars = ax2.bar(range(len(names)), times, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Episode reward distributions (box plot)
    ax3 = axes[1, 0]
    episode_data = [r['episode_rewards'] for r in successful_results]
    bp = ax3.boxplot(episode_data, tick_labels=names, patch_artist=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('Episode Reward')
    ax3.set_title('Reward Distribution Across Episodes', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 4. Ranking table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    # Sort by mean reward
    sorted_results = sorted(successful_results, key=lambda x: x['mean_reward'], reverse=True)

    table_data = []
    for i, r in enumerate(sorted_results):
        table_data.append([
            f"{i + 1}",
            r['name'],
            f"{r['mean_reward']:.1f}",
            f"±{r['std_reward']:.1f}",
            f"{r['training_time']:.1f}s"
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Rank', 'Algorithm', 'Mean', 'Std', 'Time'],
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.3, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color first place
    if len(table_data) > 0:
        for i in range(5):
            table[(1, i)].set_facecolor('#FFD700')

    ax4.set_title('Algorithm Rankings', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Results saved to {save_path}")


def main():
    print("=" * 60)
    print("RL ALGORITHM COMPARISON FOR ELECTRICITY GRID")
    print("=" * 60)

    # Create environment
    env = DummyVecEnv([make_env])

    # Configuration - REDUCED for faster testing
    TRAINING_STEPS = 15000  # Reduced from 20k
    EVAL_EPISODES = 10  # Reduced from 15

    # Define algorithms to test - STREAMLINED to best performers only
    algorithms = [
        # Best on-policy: PPO
        {
            'class': PPO,
            'name': 'PPO',
            'kwargs': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'gamma': 0.99,
            }
        },
        # Fast baseline: A2C
        {
            'class': A2C,
            'name': 'A2C',
            'kwargs': {
                'learning_rate': 7e-4,
                'n_steps': 5,
                'gamma': 0.99,
            }
        },
        # Best off-policy: SAC (usually best for continuous)
        {
            'class': SAC,
            'name': 'SAC',
            'kwargs': {
                'learning_rate': 3e-4,
                'buffer_size': 20000,  # Reduced from 50k
                'batch_size': 128,  # Reduced from 256
                'gamma': 0.99,
                'tau': 0.005,
                'ent_coef': 'auto',
                'train_freq': 1,
                'gradient_steps': 1,
            }
        },
        # Robust alternative: TD3
        {
            'class': TD3,
            'name': 'TD3',
            'kwargs': {
                'learning_rate': 3e-4,
                'buffer_size': 20000,  # Reduced from 50k
                'batch_size': 128,  # Reduced from 256
                'gamma': 0.99,
                'tau': 0.005,
                'train_freq': 1,
                'gradient_steps': 1,
            }
        },
    ]

    # Store results
    results = []

    # Evaluate random agent FIRST for debugging
    print(f"\n{'=' * 60}")
    print("Evaluating Random Agent (Baseline)...")
    print(f"{'=' * 60}")

    # Create a test environment to get action space
    temp_env = make_env()

    random_agent = RandomAgent(temp_env.action_space)
    test_action, _ = random_agent.predict(None)

    mean_reward, std_reward, episode_rewards = evaluate_agent(
        random_agent, env, n_episodes=EVAL_EPISODES, agent_name="Random"
    )

    print(f"✓ Random Agent completed:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    results.append({
        'name': 'Random',
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards,
        'training_time': 0,
        'success': True
    })

    print("\n⏸️  Random agent test complete. Continuing with RL algorithms...\n")

    # Train and evaluate each algorithm
    for algo_config in algorithms:
        result = train_and_evaluate_algorithm(
            algo_config['class'],
            algo_config['name'],
            env,
            TRAINING_STEPS,
            EVAL_EPISODES,
            **algo_config['kwargs']
        )
        results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    successful_results = [r for r in results if r['success']]
    sorted_results = sorted(successful_results, key=lambda x: x['mean_reward'], reverse=True)

    print(f"\n{'Rank':<6} {'Algorithm':<20} {'Mean Reward':<15} {'Std Dev':<12} {'Training Time'}")
    print("-" * 70)

    for i, r in enumerate(sorted_results):
        print(
            f"{i + 1:<6} {r['name']:<20} {r['mean_reward']:>10.2f}     ±{r['std_reward']:>6.2f}     {r['training_time']:>8.2f}s")

    # Calculate improvement over random
    if sorted_results[0]['name'] != 'Random':
        random_result = next((r for r in results if r['name'] == 'Random'), None)
        if random_result:
            improvement = ((sorted_results[0]['mean_reward'] - random_result['mean_reward']) /
                           abs(random_result['mean_reward']) * 100)
            print(
                f"\n🏆 Best algorithm ({sorted_results[0]['name']}) shows {improvement:.1f}% improvement over random baseline")

    # Plot results
    plot_results(results)

    # Export to CSV
    df = pd.DataFrame([
        {
            'Algorithm': r['name'],
            'Mean_Reward': r['mean_reward'],
            'Std_Reward': r['std_reward'],
            'Training_Time': r['training_time'],
            'Success': r['success']
        }
        for r in results
    ])

if __name__ == "__main__":
    main()