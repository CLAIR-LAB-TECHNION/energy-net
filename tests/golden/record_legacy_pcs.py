"""Record golden trajectories from the legacy PCSEnv.

This script was run once, against the pre-refactor code base (the commit is
stored in the fixture metadata), to pin the behavior the Prosumer refactor
must preserve. It is kept for provenance; after the legacy PCSEnv is
deleted it can no longer be executed.

Usage (from the repository root):
    python -m tests.golden.record_legacy_pcs
"""

import gzip
import json
import subprocess

import numpy as np

from energy_net.gym_envs.pcs_env import PCSEnv
from tests.golden.scenarios import (
    FIXTURE_JSON,
    FIXTURE_NPZ,
    SCENARIOS,
    RecordingDataDrivenStrategy,
    legacy_layout,
)


def _plain(value):
    """Convert numpy scalars/arrays recursively into JSON-safe values."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def record():
    metadata = {"recorded_at_commit": _commit(), "scenarios": {}}
    arrays = {}

    for name, spec in SCENARIOS.items():
        strategy = spec["strategy"]()
        env = PCSEnv(
            test_data_file=spec["load"],
            predictions_file=spec["forecast"],
            price_strategy=strategy,
            verbosity=0,
            **spec["env_kwargs"],
        )
        rng = np.random.default_rng(spec["seed"])
        horizon = env.prediction_horizon
        episodes = []

        for episode_index, start in enumerate(spec["starts"]):
            if start is None:
                obs, _ = env.reset()
            else:
                obs, _ = env.reset(options={"start_date": start})

            episode_start = env.current_datetime.isoformat()
            actions = spec["actions"](rng, env.max_steps)
            observations = [obs.copy()]
            steps = []

            for action_value in actions:
                action = np.array([action_value], dtype=np.float32)
                decision_obs = observations[-1]
                obs, reward, terminated, truncated, info = env.step(action)

                # observation_at_decision duplicates the previous obs;
                # verify instead of storing it.
                assert np.array_equal(
                    np.asarray(info["observation_at_decision"], np.float32),
                    decision_obs,
                )
                info = {
                    key: value for key, value in info.items()
                    if key != "observation_at_decision"
                }
                observations.append(obs.copy())
                steps.append({
                    "action": float(action_value),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _plain(info),
                })

            key = f"{name}/{episode_index}"
            arrays[key] = np.stack(observations)
            episodes.append({
                "requested_start": start,
                "start": episode_start,
                "steps": steps,
            })

        scenario_meta = {
            "horizon": horizon,
            "max_steps": env.max_steps,
            "layout": legacy_layout(horizon),
            "episodes": episodes,
        }
        if isinstance(strategy, RecordingDataDrivenStrategy):
            arrays[f"{name}/strategy_inputs"] = np.stack(strategy.received)
        metadata["scenarios"][name] = scenario_meta
        env.close()

    # mtime=0 keeps the gzip bytes reproducible across re-recordings.
    with open(FIXTURE_JSON, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(
                json.dumps(metadata, sort_keys=True).encode("utf-8")
            )
    np.savez_compressed(FIXTURE_NPZ, **arrays)
    print(f"Recorded {len(SCENARIOS)} scenarios to {FIXTURE_JSON}")


if __name__ == "__main__":
    record()
