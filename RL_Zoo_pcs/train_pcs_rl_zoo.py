import sys
sys.path.append("../")  # ensure the root is in your PYTHONPATH
import RL_Zoo_pcs.register_pcsenv as register_pcsenv  # Register PCSUnitEnv

from rl_zoo3.utils import create_test_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import yaml
import argparse
import os

def train():
    with open("../energy_net/config/rl_zoo_pcs.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create vectorized training environment
    train_env = make_vec_env(
        env_id=config["env"]["env_id"],
        n_envs=config["train"]["n_envs"],
        env_kwargs=config["env"]["env_kwargs"],
        vec_env_cls=DummyVecEnv
    )

    if config["train"]["normalize"]:
        train_env = VecNormalize(train_env, **config["train"]["normalize_kwargs"])

    # Initialize model
    model = PPO(
        config["algo"]["policy"],
        train_env,
        learning_rate=config["algo"]["learning_rate"],
        n_steps=config["algo"]["n_steps"],
        batch_size=config["algo"]["batch_size"],
        n_epochs=config["algo"]["n_epochs"],
        gamma=config["algo"]["gamma"],
        gae_lambda=config["algo"]["gae_lambda"],
        clip_range=config["algo"]["clip_range"],
        ent_coef=config["algo"]["ent_coef"],
        vf_coef=config["algo"]["vf_coef"],
        max_grad_norm=config["algo"]["max_grad_norm"],
        tensorboard_log=config["logging"]["tensorboard_log"],
        verbose=1
    )

    # Create log directory if it doesn't exist
    os.makedirs(config["logging"]["log_path"], exist_ok=True)
    os.makedirs(os.path.dirname(config["logging"]["save_path"]), exist_ok=True)

    # Training loop with periodic checkpoints
    for iter_num in range(config["train"]["n_timesteps"] // config["eval"]["eval_freq"]):
        model.learn(
            total_timesteps=config["eval"]["eval_freq"],
            reset_num_timesteps=False,
            tb_log_name=f"PPO_PCS_iter_{iter_num}"
        )
        if (iter_num + 1) % (config["logging"]["save_freq"] // config["eval"]["eval_freq"]) == 0:
            model.save(f"{config['logging']['save_path']}_{iter_num}")
            if config["train"]["normalize"]:
                train_env.save(f"{config['logging']['save_path']}_{iter_num}_vecnormalize.pkl")

    # Save final model
    model.save(f"{config['logging']['save_path']}_final")
    if config["train"]["normalize"]:
        train_env.save(f"{config['logging']['save_path']}_final_vecnormalize.pkl")

def evaluate():
    with open("../energy_net/config/rl_zoo_pcs.yaml", "r") as f:
        config = yaml.safe_load(f)

    eval_env = create_test_env(
        config["env"]["env_id"],
        n_envs=1,
        env_kwargs=config["env"]["env_kwargs"]
    )

    # Load normalization stats if used
    if config["train"]["normalize"]:
        eval_env = VecNormalize.load(
            f"{config['logging']['save_path']}_final_vecnormalize.pkl",
            eval_env
        )
        eval_env.training = False
        eval_env.norm_reward = False

    model = PPO.load(f"{config['logging']['save_path']}_final.zip", env=eval_env)

    # Evaluate over several episodes
    rewards = []
    for _ in range(config["eval"]["n_eval_episodes"]):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=config["eval"]["deterministic"])
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    mean_reward = sum(rewards) / len(rewards)
    sq_diffs = [(r - mean_reward) ** 2 for r in rewards]
    std_reward = (sum(sq_diffs) / len(sq_diffs)) ** 0.5
    print(f"PCS Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    if args.evaluate:
        evaluate()
