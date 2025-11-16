#!/usr/bin/env python3
"""
Run both ISO and PCS against predefined action sequences in EnergyNetV0.
"""
import argparse
import numpy as np
import logging
import os

from energy_net import EnergyNetV0
from original.dynamics.consumption_dynamics.demand_patterns import DemandPattern
from original.market import PricingPolicy
from original.market import CostType
from original.controllers import PlotCallback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dual_run")


def parse_args():
    parser = argparse.ArgumentParser(description="Execute predefined ISO and PCS sequences in EnergyNetV0")
    parser.add_argument("--iso-action-file", type=str, required=True,
                        help=".npy file of ISO actions (normalized [-1,1])")
    parser.add_argument("--pcs-action-file", type=str, required=True,
                        help=".npy file of PCS actions (unnormalized in PCS space)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Number of steps to run (default: min(len(iso), len(pcs)))")
    parser.add_argument("--demand-pattern", type=str, default="CONSTANT",
                        choices=["CONSTANT","SINUSOIDAL","DOUBLE_PEAK","TWO_PEAK","DATA_DRIVEN"],
                        help="Demand pattern for environment")
    parser.add_argument("--demand-data", type=str, default=None,
                        help="Path to demand data YAML file (required for DATA_DRIVEN pattern)")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE",
                        choices=["ONLINE","CONSTANT","QUADRATIC","INTERVALS"],
                        help="Pricing policy for environment")
    parser.add_argument("--cost-type", type=str, default="CONSTANT",
                        choices=["CONSTANT","VARIABLE","TIME_OF_USE"],
                        help="Cost type for environment")
    parser.add_argument("--use-dispatch", action="store_true",
                        help="Include dispatch in ISO action space")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for environment reset")
    parser.add_argument("--plot-dir", type=str, default="plots/dual",
                        help="Directory to save episode plots")
    return parser.parse_args()


def main():
    args = parse_args()

    # Enforce data-driven demand file requirement
    if args.demand_pattern.upper() == "DATA_DRIVEN" and not args.demand_data:
        logger.error("DATA_DRIVEN pattern requires --demand-data to specify demand YAML file")
        return

    # Ensure plot directory exists
    os.makedirs(args.plot_dir, exist_ok=True)

    # Load sequences
    iso_seq = np.load(args.iso_action_file)
    pcs_seq = np.load(args.pcs_action_file)
    length = args.timesteps or min(len(iso_seq), len(pcs_seq))
    logger.info(f"Loaded ISO seq ({iso_seq.shape}), PCS seq ({pcs_seq.shape}), running {length} steps")

    # Convert string flags to enums
    demand_pattern = DemandPattern[args.demand_pattern]
    pricing_policy = PricingPolicy[args.pricing_policy]
    cost_type = CostType[args.cost_type]

    # Create env
    env = EnergyNetV0(
        cost_type=cost_type,
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        num_pcs_agents=1,
        dispatch_config={"use_dispatch_action": args.use_dispatch, "dispatch_strategy": "PROPORTIONAL"},
        demand_data_path=args.demand_data
    )
    # Precompute ISO action space bounds for unnormalization
    iso_space = env.action_space["iso"]
    iso_low = iso_space.low
    iso_high = iso_space.high

    # Reset
    obs_dict, info = env.reset(seed=args.seed)
    total_iso_reward = 0.0
    total_pcs_reward = 0.0

    # Prepare storage for one episode's data
    episode_data = []

    for t in range(length):
        # Determine ISO action: raw dispatch or normalized prices
        if args.use_dispatch:
            # iso_seq contains raw MWh dispatch values in the last dimension
            iso_action = iso_seq[t % len(iso_seq)]
        else:
            # Unnormalize normalized price actions from [-1,1] to original bounds
            norm_iso = iso_seq[t % len(iso_seq)]
            iso_action = iso_low + (norm_iso + 1.0) * 0.5 * (iso_high - iso_low)
        # PCS action is already in PCS space
        pcs_action = pcs_seq[t % len(pcs_seq)]

        # Step
        obs_dict, rewards, terminated, truncated, info = env.step({"iso": iso_action, "pcs": pcs_action})
        iso_r = rewards[0] if isinstance(rewards, (list, tuple)) else rewards.get('iso', 0.0)
        pcs_r = rewards[1] if isinstance(rewards, (list, tuple)) else rewards.get('pcs', 0.0)
        total_iso_reward += iso_r
        total_pcs_reward += pcs_r

        # Collect step metrics for plotting
        step_info = info
        step_data = {
            'step': t,
            'production': step_info.get('production', 0.0),
            'consumption': step_info.get('consumption', 0.0),
            'predicted_demand': step_info.get('predicted_demand', 0.0),
            'realized_demand': step_info.get('realized_demand', 0.0),
            'battery_level': step_info.get('battery_level', 0.0),
            'net_exchange': step_info.get('net_exchange', 0.0),
            'iso_sell_price': step_info.get('iso_sell_price', 0.0),
            'iso_buy_price': step_info.get('iso_buy_price', 0.0),
            'dispatch': step_info.get('dispatch', 0.0),
            'dispatch_cost': step_info.get('dispatch_cost', 0.0),
            'reserve_cost': step_info.get('reserve_cost', 0.0),
            'pcs_actions': pcs_action,
            'iso_reward': iso_r,
            'pcs_reward': pcs_r
        }
        episode_data.append(step_data)

        logger.debug(f"Step {t}: iso_r={iso_r:.3f}, pcs_r={pcs_r:.3f}")

        # Termination
        done = False
        if isinstance(terminated, dict):
            done = terminated.get('iso', False) or terminated.get('pcs', False)
        else:
            done = bool(terminated)
        if done:
            logger.info(f"Episode finished at step {t}")
            break

    logger.info(f"Run complete: ISO total reward={total_iso_reward:.2f}, PCS total reward={total_pcs_reward:.2f}")
    env.close()

    # Generate detailed plots using the existing callback
    callback = PlotCallback(verbose=0)
    callback.agent_name = "dual_run"
    callback.save_path = args.plot_dir
    # Provide our collected data as a single-episode list
    callback.all_episodes_actions = [episode_data]
    callback.plot_episode_results(0, args.plot_dir)


if __name__ == "__main__":
    main() 