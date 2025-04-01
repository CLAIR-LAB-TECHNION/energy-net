from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle  # Import pickle to save data
import json


class ActionTrackingCallback(BaseCallback):
    """
    A custom callback for tracking actions during training.
    """
    def __init__(self, agent_name: str, env_config=None, verbose=0, is_training=True, save_path="training_plots"):
        super().__init__(verbose)
        """
        Initializes the ActionTrackingCallback.

        Args:
            agent_name (str): Name of the agent being tracked.
            env_config (dict, optional): Environment configuration. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
            is_training (bool, optional): Flag to distinguish between training and evaluation. Defaults to True.
            save_path (str, optional): Directory to save plots. Defaults to "training_plots".
        """
        self.agent_name = agent_name
        self.env_config = env_config or {'Dispatch_price': 5.0}  # Default if not provided
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # Basic tracking
        self.episode_actions = []
        self.all_episodes_actions = []
        self.current_step = 0
        self.steps_in_episode = 0  
        self.max_steps = 48  # Maximum steps per episode (24 hours)
        
        # Extended tracking for detailed visualization
        self.timestamps = []
        self.predicted_demands = []
        self.realized_demands = []
        self.productions = []
        self.consumptions = []
        self.battery_levels = []
        self.net_exchanges = []
        self.iso_sell_prices = [] 
        self.iso_buy_prices = []
        self.dispatch = []   
        self.is_training = is_training 
        self.pcs_actions = [] 
        
    def _on_step(self) -> bool:
        """
        This method is called once per step during training.
        It tracks actions, observations, and other relevant information.

        Returns:
            bool: True to continue training, False to stop.
        """
        infos = self.locals.get('infos')
        # If infos is not a non-empty list, warn
        if not (isinstance(infos, list) and len(infos) > 0):
            if self.verbose > 0:
                print("Callback Warning - 'infos' is empty or not a list; using {}")
            info = {}
        else:
            info = infos[0]
            
        obs = self.locals.get('new_obs')
        action = self.locals.get('actions')
        done = self.locals.get('dones', [False])[0]  # Check if episode is done
        
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]
        
        # Debug info structure in first step of first episode
        if self.steps_in_episode == 0 and len(self.all_episodes_actions) == 0 and self.verbose > 0:
            print(f"\nDEBUG - Info structure in callback ({self.agent_name}):")
            print(f"Info keys: {list(info.keys())}")
            for key in info.keys():
                print(f"  {key} type: {type(info[key])}")
                if isinstance(info[key], dict):
                    print(f"    {key} keys: {list(info[key].keys())}")
        
        # Get agent-specific info - supports both nested and flat structures
        iso_info = info.get('iso', {})
        pcs_info = info.get('pcs', {})
                
        # Track steps and store data
        step_data = {
            'step': self.steps_in_episode,
            'action': float(action) if action is not None else 0.0,
            'observation': obs.tolist() if isinstance(obs, np.ndarray) else obs,
            'predicted_demand': info.get('predicted_demand', iso_info.get('predicted_demand', 0.0)),
            'realized_demand': info.get('realized_demand', iso_info.get('realized_demand', 0.0)),
            'production': info.get('production', pcs_info.get('production', 0.0)),
            'consumption': info.get('consumption', pcs_info.get('consumption', 0.0)),
            'battery_level': info.get('battery_level', pcs_info.get('battery_level', 0.0)),
            'net_exchange': info.get('net_exchange', pcs_info.get('net_exchange', 0.0)),
            'iso_sell_price': info.get('iso_sell_price', iso_info.get('sell_price', iso_info.get('sell_prices', [0.0])[-1] if iso_info.get('sell_prices') else 0.0)),
            'iso_buy_price': info.get('iso_buy_price', iso_info.get('buy_price', iso_info.get('buy_prices', [0.0])[-1] if iso_info.get('buy_prices') else 0.0)),
            'dispatch_cost': info.get('dispatch_cost', iso_info.get('dispatch_cost', iso_info.get('dispatch_costs', [0.0])[-1] if iso_info.get('dispatch_costs') else 0.0)),
            'reserve_cost': info.get('reserve_cost', iso_info.get('reserve_cost', iso_info.get('reserve_costs', [0.0])[-1] if iso_info.get('reserve_costs') else 0.0)),
            'shortfall': info.get('shortfall', iso_info.get('shortfall', iso_info.get('shortfalls', [0.0])[-1] if iso_info.get('shortfalls') else 0.0)),
            'dispatch': info.get('dispatch', iso_info.get('dispatch', 0.0)),
            'net_demand': info.get('net_demand', iso_info.get('net_demand', 0.0)),
            'pcs_cost': info.get('pcs_exchange_cost', pcs_info.get('cost', 0.0)),
            'pcs_actions': info.get('pcs_action', pcs_info.get('actions', [])),
            'iso_reward': info.get('iso_reward', info.get('episode_iso_reward', 0.0)),
            'pcs_reward': info.get('pcs_reward', info.get('episode_pcs_reward', 0.0))
        }
        
        # Debug battery level
        if self.verbose > 0 and self.steps_in_episode % 10 == 0:
            print(f"\nStep {self.steps_in_episode} Battery Level: {step_data['battery_level']}")
            if 'pcs' in info:
                print(f"  From PCS info: {pcs_info.get('battery_level')}")
            print(f"  From flat info: {info.get('battery_level')}")
        
        self.episode_actions.append(step_data)
        self.steps_in_episode += 1
        
        # Reset step counter when episode ends
        if done or self.steps_in_episode >= self.max_steps:
            if len(self.episode_actions) > 0:
                self.all_episodes_actions.append(self.episode_actions)
                # Save last episode data to JSON for reference
                ep_num = len(self.all_episodes_actions) - 1
                # Plot the last episode data
                if self.is_training:
                    self.plot_episode_results(ep_num, self.save_path)
            self.episode_actions = []
            self.steps_in_episode = 0
            
        return True

    def plot_episode_results(self, episode_num: int, save_path: str):
        """
        Generate visualization similar to simple_market_simulation_test
        """
        if episode_num >= len(self.all_episodes_actions):
            print(f"No data for episode {episode_num}")
            return
            
        episode_data = self.all_episodes_actions[episode_num]
        if not episode_data:
            return
            
        # Extract data using dict.get with defaults
        steps = [d.get('step', 0) for d in episode_data]
        production = [d.get('production', 0.0) for d in episode_data]
        net_exchange = [d.get('net_exchange', 0.0) for d in episode_data]
        battery_level = [d.get('battery_level', 0.0) for d in episode_data]
        predicted_demand = [d.get('predicted_demand', 0.0) for d in episode_data]
        realized_demand = [d.get('realized_demand', 0.0) for d in episode_data]
        iso_sell_prices = [d.get('iso_sell_price', 0.0) for d in episode_data]  # safe extraction
        iso_buy_prices = [d.get('iso_buy_price', 0.0) for d in episode_data]    # safe extraction
        dispatch = [d.get('dispatch', 0.0) for d in episode_data]
        
        # Compute net demand as realized_demand + net_exchange
        net_demand = [r + n for r, n in zip(realized_demand, net_exchange)]
        
        # Use pre-calculated costs from controller instead of recalculating
        dispatch_costs = [d.get('dispatch_cost', 0.0) for d in episode_data]
        pcs_costs = [
            d.get('net_exchange', 0.0) * (d.get('iso_sell_price', 0.0) if d.get('net_exchange', 0.0) > 0 
                                          else d.get('iso_buy_price', 0.0))
            for d in episode_data
        ]  # updated pricing lookup
        reserve_costs = [d.get('reserve_cost', 0.0) for d in episode_data]

        # ===== Figure 1: Energy flows + Battery levels and Prices =====
        fig = plt.figure(figsize=(15, 12))  # Back to original height
        
        ax1 = plt.subplot(2, 1, 1)  # Back to 2 rows
        
        # Dispatch bar as originally defined
        ax1.bar(steps, dispatch, width=0.8, color='lightblue', label='dispatch')
        
        # Plot demand lines on top of dispatch:
        ax1.plot(steps, predicted_demand, 'k--', linewidth=2, label='Predicted Demand')
        ax1.plot(steps, realized_demand, 'b-', linewidth=2, label='Non Strategic Demand')
        ax1.plot(steps, net_demand, 'r-', linewidth=2, label='Total Demand')
        
        ax1.set_ylabel('Energy (MWh)', fontsize=12)
        ax1.set_title(f'{self.agent_name} Energy Flows - Episode {episode_num}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)

        # Plot 2: Battery Levels and Prices (Bottom)
        ax2 = plt.subplot(2, 1, 2)
        
        # Battery levels for all PCS agents
        pcs_battery_levels = []
        for d in episode_data:
            levels = d.get('battery_level', [])
            if not isinstance(levels, list):
                levels = [levels]
            pcs_battery_levels.append(levels)
            
        if pcs_battery_levels and len(pcs_battery_levels[0]) > 0:
            for agent_idx in range(len(pcs_battery_levels[0])):
                agent_levels = [step_levels[agent_idx] for step_levels in pcs_battery_levels]
                ax2.plot(steps, agent_levels, '-', linewidth=2, label=f'PCS {agent_idx + 1} Battery')
        else:
            # For single battery level (non-list)
            ax2.plot(steps, battery_level, '-', linewidth=2, label='Battery Level')

        # Prices on secondary y-axis
        ax3 = ax2.twinx()
        ax3.plot(steps, iso_sell_prices, 'r--', linewidth=2, label='ISO Sell Price')
        ax3.plot(steps, iso_buy_prices, 'g--', linewidth=2, label='ISO Buy Price')
        ax3.set_ylabel('Price ($/MWh)', color='black', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='black')
        
        # Set labels and grid for battery axis
        ax2.set_ylabel('Battery Level (MWh)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        plt.tight_layout()
        fig_path = os.path.join(save_path, f'episode_{episode_num}_detail.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved detailed plot to {fig_path}")
        
        # ===== Figure 2: Cost components only =====
        fig2 = plt.figure(figsize=(10, 6))
        ax4 = fig2.add_subplot(1, 1, 1)

        # Separate positive costs (dispatch, reserve) from negative/positive exchange costs
        # Create separate bars for each cost component
        ax4.bar(steps, dispatch_costs, label='Dispatch Cost', color='lightblue')
        ax4.bar(steps, reserve_costs, label='Reserve Cost', color='salmon')
        
        # PCS exchange costs can be positive (cost to grid) or negative (revenue)
        # Plot them separately with different colors based on sign
        pos_pcs_costs = [max(0, cost) for cost in pcs_costs]  # Only positive values
        neg_pcs_costs = [min(0, cost) for cost in pcs_costs]  # Only negative values
        
        if any(pos_pcs_costs):
            ax4.bar(steps, pos_pcs_costs, label='PCS Exchange Cost (Grid Paying)', color='lightgreen')
        if any(neg_pcs_costs):
            ax4.bar(steps, neg_pcs_costs, label='PCS Exchange Revenue (Grid Earning)', color='darkgreen')
        
        ax4.set_ylabel('Cost ($)', fontsize=12)
        ax4.set_title('Cost Components Over Time', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right', fontsize=10)
        
        fig2.tight_layout()
        fig_path_2 = os.path.join(save_path, f'episode_{episode_num}_cost_components.png')
        plt.savefig(fig_path_2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cost components plot to {fig_path_2}")
        
        # ===== Figure 3: Single bar to show final cost distribution =====
        fig3 = plt.figure(figsize=(6, 8))
        ax5 = fig3.add_subplot(1, 1, 1)
        
        total_dispatch = sum(dispatch_costs)
        total_pcs = sum(pcs_costs)
        total_reserve = sum(reserve_costs)
        
        # Plot individual components with different x positions
        bar_positions = [0, 1, 2]  # Different x positions for each bar
        width = 0.7  # Width of the bars
        
        bar_dispatch = ax5.bar([bar_positions[0]], [total_dispatch], width=width, color='lightblue', label='Dispatch Cost')
        bar_reserve = ax5.bar([bar_positions[1]], [total_reserve], width=width, color='salmon', label='Reserve Cost')
        
        # Handle PCS exchange differently based on sign
        if total_pcs >= 0:
            bar_pcs = ax5.bar([bar_positions[2]], [total_pcs], width=width, color='lightgreen', label='PCS Exchange Cost')
        else:
            bar_pcs = ax5.bar([bar_positions[2]], [total_pcs], width=width, color='darkgreen', label='PCS Exchange Revenue')
        
        # Add text labels with values
        def add_value_label(bar, value):
            """Add a label with the value above each bar"""
            height = bar[0].get_height()
            ax5.text(bar[0].get_x() + bar[0].get_width()/2., 
                    height if height >= 0 else height - 1000000,
                    f'${abs(value):,.0f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, color='black', fontweight='bold')
        
        add_value_label(bar_dispatch, total_dispatch)
        add_value_label(bar_reserve, total_reserve)
        add_value_label(bar_pcs, total_pcs)
        
        # Add total cost label
        total_cost = total_dispatch + total_reserve + total_pcs
        ax5.text(0.5, 0.02, f'Total: ${total_cost:,.0f}', 
                 ha='center', transform=ax5.transAxes,
                 fontsize=12, fontweight='bold')
        
        ax5.set_ylabel('Total Cost ($)', fontsize=12)
        ax5.set_title('Episode Final Cost Distribution', fontsize=14)
        
        # Set x-ticks and labels for the bars
        ax5.set_xticks(bar_positions)
        ax5.set_xticklabels(['Dispatch Cost', 'Reserve Cost', 'PCS Exchange'])
        
        # Ensure y-axis limits accommodate all values
        y_min = min(0, total_pcs, total_dispatch, total_reserve) * 1.1  # Add 10% margin
        y_max = max(0, total_pcs, total_dispatch, total_reserve) * 1.1
        ax5.set_ylim(y_min, y_max)
        
        # Place legend outside the plot area
        ax5.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        fig3.tight_layout()
        final_cost_path = os.path.join(save_path, f'episode_{episode_num}_final_cost_distribution.png')
        plt.savefig(final_cost_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved final cost distribution plot to {final_cost_path}")
        
        if episode_data and any('pcs_actions' in d for d in episode_data):
            plt.figure(figsize=(10, 6))
            steps = range(len(episode_data))
            pcs_actions = [d.get('pcs_actions', []) for d in episode_data]
            
            # Check if pcs_actions is a list of lists (multiple agents)
            if pcs_actions and isinstance(pcs_actions[0], list) and len(pcs_actions[0]) > 0:
                for agent_idx in range(len(pcs_actions[0])):
                    agent_actions = [step_actions[agent_idx] if isinstance(step_actions, list) and len(step_actions) > agent_idx else 0.0 
                                   for step_actions in pcs_actions]
                    plt.plot(steps, agent_actions, label=f'PCS Agent {agent_idx + 1}')
            else:
                # For single action (not a list or empty list)
                # First try to get the pcs_actions directly if it's not a list
                actions = []
                for d in episode_data:
                    action = d.get('pcs_actions', None)
                    if action is None or (isinstance(action, list) and len(action) == 0):
                        # Fallback to using the raw action
                        action = d.get('action', 0.0)
                    elif isinstance(action, list) and len(action) > 0:
                        action = action[0]  # Take the first element if it's a list
                    actions.append(action)
                plt.plot(steps, actions, label='PCS Agent Action')
            
            plt.xlabel('Step')
            plt.ylabel('Battery Action')
            plt.title(f'PCS Agents Actions - Episode {episode_num}')
            plt.legend()
            plt.grid(True)
            pcs_actions_path = os.path.join(save_path, f'episode_{episode_num}_pcs_actions.png')
            plt.savefig(pcs_actions_path)
            plt.close()
            print(f"Saved PCS actions plot to {pcs_actions_path}")

    def _on_rollout_end(self) -> bool:
        """
        This method is called once per rollout.
        """
        # Don't clear episode actions here anymore
        return True
    
    def _on_training_end(self) -> None:
        """
        This method is called at the end of training or evaluation.
        It saves all runtime information (all episodes' actions) to a file.
        """
        mode = "training" if self.is_training else "evaluation"
        file_path = os.path.join(f"runtime_info_{mode}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.all_episodes_actions, f)
        print(f"{mode.capitalize()} runtime info saved to {file_path}")

# The method below matches the format from eval_iso_zoo.py and eval_pcs_zoo.py
def plot_episode_detail(episode_data, episode_num, output_dir):
    """
    Plot detailed metrics for a single episode.
    """
    # Extract data
    steps = range(len(episode_data))
    
    # Get ISO metrics
    predicted_demands = [d.get('info', {}).get('iso', {}).get('predicted_demands', [0.0])[0] 
                        for d in episode_data]
    realized_demands = [d.get('info', {}).get('iso', {}).get('realized_demands', [0.0])[0] 
                        for d in episode_data]
    pcs_demands = [d.get('info', {}).get('iso', {}).get('pcs_demands', [0.0])[0] 
                    for d in episode_data]
    net_demands = [d.get('info', {}).get('iso', {}).get('net_demands', [0.0])[0] 
                    for d in episode_data]
    buy_prices = [d.get('info', {}).get('iso', {}).get('buy_prices', [0.0])[0] 
                for d in episode_data]
    sell_prices = [d.get('info', {}).get('iso', {}).get('sell_prices', [0.0])[0] 
                    for d in episode_data]
    
    # Get PCS metrics
    battery_levels = [d.get('info', {}).get('pcs', {}).get('battery_levels', [0.0])[0] 
                    for d in episode_data]
    energy_exchanges = [d.get('info', {}).get('pcs', {}).get('energy_exchanges', [0.0])[0] 
                        for d in episode_data]
    
    # Get dispatch and costs
    dispatch_costs = [d.get('info', {}).get('iso', {}).get('dispatch_costs', [0.0])[0] 
                    for d in episode_data]
    reserve_costs = [d.get('info', {}).get('iso', {}).get('reserve_costs', [0.0])[0] 
                    for d in episode_data]
    dispatch = [d.get('info', {}).get('shared', {}).get('dispatch', 0.0)
                for d in episode_data]
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))
    
    # Plot 1: Demand and generation (top subplot)
    ax1 = axs[0]
    
    # Dispatch as bars
    ax1.bar(steps, dispatch, color='lightblue', alpha=0.7, label='Dispatch')
    
    # Demands as lines
    ax1.plot(steps, predicted_demands, 'k--', linewidth=2, label='Predicted Demand')
    ax1.plot(steps, realized_demands, 'b-', linewidth=2, label='Realized Demand')
    ax1.plot(steps, pcs_demands, 'g-', linewidth=2, label='PCS Demand')
    ax1.plot(steps, net_demands, 'r-', linewidth=2, label='Net Demand')
    
    ax1.set_ylabel('Power (MW)', fontsize=12)
    ax1.set_title(f'Episode {episode_num} - Energy Flows', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Battery and exchange (middle subplot)
    ax2 = axs[1]
    
    # Battery levels on left y-axis
    ax2.plot(steps, battery_levels, 'g-', linewidth=2, label='Battery Level')
    ax2.set_ylabel('Battery Level (MWh)', color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Energy exchange on right y-axis
    ax2_right = ax2.twinx()
    ax2_right.plot(steps, energy_exchanges, 'b-', linewidth=2, label='Energy Exchange')
    ax2_right.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2_right.set_ylabel('Energy Exchange (MWh)', color='b', fontsize=12)
    ax2_right.tick_params(axis='y', labelcolor='b')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax2.set_title('Battery Levels and Energy Exchange', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prices (bottom subplot)
    ax3 = axs[2]
    
    ax3.plot(steps, buy_prices, 'r-', linewidth=2, label='Buy Price')
    ax3.plot(steps, sell_prices, 'g-', linewidth=2, label='Sell Price')
    
    # Fill the price spread
    ax3.fill_between(steps, buy_prices, sell_prices, 
                    where=[b > s for b, s in zip(buy_prices, sell_prices)], 
                    color='gray', alpha=0.3)
    
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Price ($/MWh)', fontsize=12)
    ax3.set_title('ISO Prices', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_detail.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_iso_prices(episode_data, episode_num, output_dir):
    """
    Plot ISO prices and dispatch costs.
    """
    # Extract data
    steps = range(len(episode_data))
    
    # Get prices
    buy_prices = [d.get('info', {}).get('iso', {}).get('buy_prices', [0.0])[0] 
                for d in episode_data]
    sell_prices = [d.get('info', {}).get('iso', {}).get('sell_prices', [0.0])[0] 
                    for d in episode_data]
    
    # Get costs
    dispatch_costs = [d.get('info', {}).get('iso', {}).get('dispatch_costs', [0.0])[0] 
                    for d in episode_data]
    reserve_costs = [d.get('info', {}).get('iso', {}).get('reserve_costs', [0.0])[0] 
                    for d in episode_data]
    total_costs = [d.get('info', {}).get('iso', {}).get('total_costs', [0.0])[0] 
                    for d in episode_data]
    
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Prices
    ax1 = axs[0]
    ax1.plot(steps, buy_prices, 'r-', linewidth=2, label='Buy Price')
    ax1.plot(steps, sell_prices, 'g-', linewidth=2, label='Sell Price')
    
    # Fill the price spread
    ax1.fill_between(steps, buy_prices, sell_prices, 
                    where=[b > s for b, s in zip(buy_prices, sell_prices)], 
                    color='gray', alpha=0.3)
    
    ax1.set_ylabel('Price ($/MWh)', fontsize=12)
    ax1.set_title(f'Episode {episode_num} - ISO Prices', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Costs
    ax2 = axs[1]
    ax2.bar(steps, dispatch_costs, color='lightblue', label='Dispatch Cost')
    ax2.bar(steps, reserve_costs, bottom=dispatch_costs, color='salmon', label='Reserve Cost')
    ax2.plot(steps, total_costs, 'k-', linewidth=2, label='Total Cost')
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Cost ($)', fontsize=12)
    ax2.set_title('ISO Costs', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_iso_prices.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_pcs_actions(episode_data, episode_num, output_dir):
    """
    Plot PCS actions and battery levels.
    """
    # Extract data
    steps = range(len(episode_data))
    
    # Get PCS metrics
    battery_levels = [d.get('info', {}).get('pcs', {}).get('battery_levels', [0.0])[0] 
                    for d in episode_data]
    pcs_actions = [d.get('actions', {}).get('pcs', 0.0) for d in episode_data]
    energy_exchanges = [d.get('info', {}).get('pcs', {}).get('energy_exchanges', [0.0])[0] 
                        for d in episode_data]
    
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Battery levels
    ax1 = axs[0]
    ax1.plot(steps, battery_levels, 'g-', linewidth=2, label='Battery Level')
    
    ax1.set_ylabel('Battery Level (MWh)', fontsize=12)
    ax1.set_title(f'Episode {episode_num} - PCS Battery Levels', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Actions and exchanges
    ax2 = axs[1]
    
    # Actions on left y-axis
    ax2.plot(steps, pcs_actions, 'b-', linewidth=2, label='PCS Action')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Action Value', color='b', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Energy exchange on right y-axis
    ax2_right = ax2.twinx()
    ax2_right.plot(steps, energy_exchanges, 'r-', linewidth=2, label='Energy Exchange')
    ax2_right.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2_right.set_ylabel('Energy Exchange (MWh)', color='r', fontsize=12)
    ax2_right.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_title('PCS Actions and Energy Exchange', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_pcs_actions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

