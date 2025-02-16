from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle  # Import pickle to save data


class ActionTrackingCallback(BaseCallback):
    """
    A custom callback for tracking actions during training.
    """
    def __init__(self, agent_name: str, env_config=None, verbose=0):
        super().__init__(verbose)
        """
        Initializes the ActionTrackingCallback.

        Args:
            agent_name (str): Name of the agent being tracked.
            env_config (dict, optional): Environment configuration. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        self.agent_name = agent_name
        self.env_config = env_config or {'Dispatch_price': 5.0}  # Default if not provided
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
            print("Callback Warning - 'infos' is empty or not a list; using {}")
            info = {}
        else:
            info = infos[0]
            
        obs = self.locals.get('new_obs')
        action = self.locals.get('actions')
        done = self.locals.get('dones', [False])[0]  # Check if episode is done
        
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]
                
        # Track steps and store data
        step_data = {
            'step': self.steps_in_episode,
            'action': float(action) if action is not None else 0.0,
            'observation': obs.tolist() if isinstance(obs, np.ndarray) else obs,
            'predicted_demand': info.get('predicted_demand', 0.0),
            'realized_demand': info.get('realized_demand', 0.0),
            'production': info.get('production', 0.0),
            'consumption': info.get('consumption', 0.0),
            'battery_level': info.get('battery_level', 0.0),
            'net_exchange': info.get('net_exchange', 0.0),
            'iso_sell_price': info.get('iso_sell_price', 0.0), 
            'iso_buy_price': info.get('iso_buy_price', 0.0),    
            'dispatch_cost': info.get('dispatch_cost', 0.0),
            'reserve_cost': info.get('reserve_cost', 0.0),
            'shortfall': info.get('shortfall', 0.0),
            'dispatch': info.get('dispatch', 0.0),
            'net_demand': info.get('net_demand', 0.0)
        }
        
        
        
        self.episode_actions.append(step_data)
        self.steps_in_episode += 1
        
        # Reset step counter when episode ends
        if done or self.steps_in_episode >= self.max_steps:
            if len(self.episode_actions) >= self.max_steps:
                self.all_episodes_actions.append(self.episode_actions)
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
        dispatch_costs = [d['dispatch_cost'] for d in episode_data]
        pcs_costs = [
            d.get('net_exchange', 0.0) * (d.get('iso_sell_price', 0.0) if d.get('net_exchange', 0.0) > 0 
                                          else d.get('iso_buy_price', 0.0))
            for d in episode_data
        ]  # updated pricing lookup
        reserve_costs = [d['reserve_cost'] for d in episode_data]

        # ===== Figure 1: Energy flows + Battery levels and Prices =====
        fig = plt.figure(figsize=(15, 12))
        
        ax1 = plt.subplot(2, 1, 1)
        
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

        # Plot 2: Battery Level and Prices (Middle)
        ax2 = plt.subplot(2, 1, 2)
        
        # Battery level on primary y-axis
        color = 'blue'
        ax2.plot(steps, battery_level, color=color, linewidth=2, label='Battery Level')
        ax2.set_ylabel('Battery Level (MWh)', color=color, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(True, alpha=0.3)
        
        # Prices on secondary y-axis
        ax3 = ax2.twinx()
        ax3.plot(steps, iso_sell_prices, 'r-', linewidth=2, label='ISO Sell Price')
        ax3.plot(steps, iso_buy_prices, 'g-', linewidth=2, label='ISO Buy Price')
        ax3.set_ylabel('Price ($/MWh)', color='black', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='black')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        fig_path_1 = os.path.join(save_path, f'episode_{episode_num}_flows_prices.png')
        plt.savefig(fig_path_1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved flows and prices plot to {fig_path_1}")
        
        # ===== Figure 2: Cost components only =====
        fig2 = plt.figure(figsize=(10, 6))
        ax4 = fig2.add_subplot(1, 1, 1)

        # Create stacked bar chart
        ax4.bar(steps, dispatch_costs, label='Dispatch Cost', color='lightblue')
        ax4.bar(steps, pcs_costs, bottom=dispatch_costs, label='PCS Exchange Cost', color='lightgreen')
        ax4.bar(steps, reserve_costs, bottom=[sum(x) for x in zip(dispatch_costs, pcs_costs)], 
                label='Reserve Cost', color='salmon')
        
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
        
        fig3 = plt.figure(figsize=(6, 6))
        ax5 = fig3.add_subplot(1, 1, 1)
        
        total_dispatch = sum(dispatch_costs)
        total_pcs = sum(pcs_costs)
        total_reserve = sum(reserve_costs)
        
        ax5.bar([0], [total_dispatch], color='lightblue', label='Dispatch Cost')
        ax5.bar([0], [total_pcs], bottom=[total_dispatch], color='lightgreen', label='PCS Exchange Cost')
        ax5.bar([0], [total_reserve], bottom=[total_dispatch + total_pcs], color='salmon', label='Reserve Cost')
        
        ax5.set_ylabel('Total Cost ($)', fontsize=12)
        ax5.set_title('Episode Final Cost Distribution', fontsize=14)
        ax5.set_xticks([])
        
        # Place legend on the right side, out of the bar area
        ax5.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        fig3.tight_layout()
        final_cost_path = os.path.join(save_path, f'episode_{episode_num}_final_cost_distribution.png')
        plt.savefig(final_cost_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved final cost distribution plot to {final_cost_path}")

    def _on_rollout_end(self) -> bool:
        """
        This method is called once per rollout.
        """
        # Don't clear episode actions here anymore
        return True
    
    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        It saves all runtime information (all episodes' actions) to a file.
        """
        file_path = os.path.join("runtime_info.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.all_episodes_actions, f)
        print(f"Runtime info saved to {file_path}")

