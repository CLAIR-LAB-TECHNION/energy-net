from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from iso_env import ISOEnv


def load_expected_realized_days(pred_csv: str, real_csv: str) -> tuple[np.ndarray, np.ndarray]:
    # Read CSVs directly
    pred = pd.read_csv(pred_csv)
    real = pd.read_csv(real_csv)

    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    real["Datetime"] = pd.to_datetime(real["Datetime"])

    pred = pred[["timestamp", "predicted_consumption"]].rename(
        columns={"predicted_consumption": "expected"}
    )
    real = real[["Datetime", "Consumption"]].rename(
        columns={"Datetime": "timestamp", "Consumption": "realized"}
    )

    df = pred.merge(real, on="timestamp", how="inner").sort_values("timestamp")
    df = df.dropna(subset=["expected", "realized"])

    df["date"] = df["timestamp"].dt.date
    df["slot"] = df["timestamp"].dt.hour * 2 + (df["timestamp"].dt.minute // 30)

    expected_wide = df.pivot_table(index="date", columns="slot", values="expected", aggfunc="mean").sort_index()
    realized_wide = df.pivot_table(index="date", columns="slot", values="realized", aggfunc="mean").sort_index()

    expected_wide = expected_wide.reindex(columns=range(48))
    realized_wide = realized_wide.reindex(columns=range(48))

    # Drop incomplete days
    good_days = (~expected_wide.isna().any(axis=1)) & (~realized_wide.isna().any(axis=1))
    expected_wide = expected_wide.loc[good_days]
    realized_wide = realized_wide.loc[good_days]

    expected_demands = expected_wide.to_numpy(dtype=np.float32)
    realized_demands = realized_wide.to_numpy(dtype=np.float32)

    return expected_demands, realized_demands


def main():
    # CSV files in the current working directory
    pred_csv = "consumption_predictions.csv"
    real_csv = "synthetic_household_consumption_test.csv"

    expected_demands, realized_demands = load_expected_realized_days(pred_csv, real_csv)
    print("Final arrays:", expected_demands.shape, realized_demands.shape)

    max_dispatch = 1000.0

    def make_env():
        return ISOEnv(
            expected_demands=expected_demands,
            realized_demands=realized_demands,
            min_price=-500.0,
            max_price=500.0,
            max_dispatch=max_dispatch,
            seed=0,
        )

    vec_env = make_vec_env(make_env, n_envs=8)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        n_steps=256,
        batch_size=256,
    )

    model.learn(total_timesteps=200_000)

    # Save the model in the same folder as this script
    here = Path(__file__).resolve().parent
    model_path = here / "ppo_isoenv"
    model.save(str(model_path))
    print(f"Saved model to: {model_path}.zip")

    env = make_env()
    maes = []
    for _ in range(10):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        maes.append(info["mean_abs_error"])
    print("Quick eval MAE (10 eps):", float(np.mean(maes)))


if __name__ == "__main__":
    main()