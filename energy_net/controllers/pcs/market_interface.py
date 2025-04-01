from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import logging
from stable_baselines3 import PPO
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO

class MarketInterface:
    """
    Handles interactions with the energy market for the PCS controller.
    
    Responsibilities:
    1. Communicating with ISO for price determination
    2. Tracking market prices (buy/sell)
    3. Calculating energy exchange with the grid
    4. Computing financial results of market interactions
    5. Managing trained ISO agent interactions
    """
    
    def __init__(
        self,
        env_config: Dict[str, Any],
        iso_config: Dict[str, Any],
        pcs_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the market interface.
        
        Args:
            env_config: Environment configuration including demand settings
            iso_config: ISO configuration including pricing parameters
            pcs_config: PCS unit configuration
            logger: Optional logger for tracking operations
        """
        self.logger = logger
        self.env_config = env_config
        self.iso_config = iso_config
        self.pcs_config = pcs_config
        
        # Extract pricing configuration
        pricing_config = self.iso_config.get('pricing', {})
        price_params = pricing_config.get('parameters', {})
        self.min_price = price_params.get('min_price', pricing_config.get('default_sell_price', 1.0))
        self.max_price = price_params.get('max_price', pricing_config.get('default_buy_price', 10.0))
        
        # Extract reserve parameters
        self.reserve_price = self.env_config.get('reserve_price', 0.0)
        self.dispatch_price = self.env_config.get('dispatch_price', 0.0)
        
        # Initialize market state variables
        self.iso_buy_price = self.min_price
        self.iso_sell_price = self.min_price
        self.net_exchange = 0.0
        self.revenue = 0.0
        self.dispatch_cost = 0.0
        self.reserve_cost = 0.0
        self.dispatch = 0.0
        self.shortfall = 0.0
        self.net_demand = 0.0
        
        # Initialize demand tracking
        self.predicted_demand = 0.0
        self.realized_demand = 0.0
        self.sigma = self.env_config.get('demand_uncertainty', {}).get('sigma', 0.0)
        
        # Initialize ISO agent
        self.trained_iso_agent = None
        
        # Initialize default pricing mechanism for when no agent is available
        default_iso_params = self.pcs_config.get('default_iso_params', {}).get('quadratic', {})
       
        self.default_pricing_model = QuadraticPricingISO(
            buy_a=default_iso_params.get('buy_a', 1.0),
            buy_b=default_iso_params.get('buy_b', 2.0),
            buy_c=default_iso_params.get('buy_c', 5.0)
        )
        
        if self.logger:
            self.logger.info("Market Interface initialized")
    
    def set_trained_iso_agent(self, iso_agent: PPO) -> bool:
        """
        Set the trained ISO agent for price determination.
        
        Args:
            iso_agent: Trained PPO agent for ISO price decisions
            
        Returns:
            Success status of setting the agent
            
        Raises:
            Exception: If the agent fails validation
        """
        self.trained_iso_agent = iso_agent
        
        # Test that the agent works with a dummy observation
        test_obs = np.array([0.5, 100.0, 0.0], dtype=np.float32)  # time, predicted_demand, pcs_demand
        
        try:
            prices = self.trained_iso_agent.predict(test_obs, deterministic=True)[0]
            if self.logger:
                self.logger.info(f"ISO agent test successful - got prices: {prices}")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"ISO agent validation failed: {e}")
            self.trained_iso_agent = None  # Reset if validation fails
            raise e
    
    def update_market_prices(self, time: float, predicted_demand: float, pcs_demand: float) -> Tuple[float, float]:
        """
        Update market prices based on current state using ISO pricing.
        
        Args:
            time: Current time as fraction of day
            predicted_demand: Predicted system demand
            pcs_demand: Current PCS unit net demand
            
        Returns:
            Tuple of (sell_price, buy_price)
        """
        # Update internal tracking
        self.predicted_demand = predicted_demand
        
        # Create ISO observation
        iso_observation = np.array([
            time,
            predicted_demand,
            pcs_demand
        ], dtype=np.float32)
        
        # Determine prices using trained agent or default pricing model
        if self.trained_iso_agent is not None:
            try:
                prices = self.trained_iso_agent.predict(iso_observation, deterministic=True)[0]
                self.iso_sell_price, self.iso_buy_price = prices
                
                if self.iso_sell_price == 0 and self.iso_buy_price == 0:
                    if self.logger:
                        self.logger.warning("ISO agent returned zero prices - this might indicate an issue")
                
                if self.logger:
                    self.logger.info(
                        f"Using ISO agent prices:\n"
                        f"  - ISO Sell Price: {self.iso_sell_price:.2f} $/MWh\n"
                        f"  - ISO Buy Price: {self.iso_buy_price:.2f} $/MWh"
                    )
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to get prices from ISO agent: {e}")
                self.iso_sell_price = self.iso_config.get('pricing', {}).get('default_buy_price', 50.0)
                self.iso_buy_price = self.iso_config.get('pricing', {}).get('default_sell_price', 45.0)
                
                if self.logger:
                    self.logger.warning(
                        f"Falling back to default prices:\n"
                        f"  - ISO Sell Price: {self.iso_sell_price:.2f} $/MWh\n"
                        f"  - ISO Buy Price: {self.iso_buy_price:.2f} $/MWh"
                    )
        else:
            # Use quadratic pricing model when no agent is available
            buy_pricing_fn = self.default_pricing_model.get_pricing_function({'demand': predicted_demand})
            self.iso_sell_price = 50.0
            self.iso_buy_price = 0.85 * self.iso_sell_price
            if (0.5 < time < 4.5):
                self.iso_sell_price = 30.0
                self.iso_buy_price = 0.85 * self.iso_sell_price
            #self.iso_sell_price = max(buy_pricing_fn(1.0), 0)
            #self.iso_buy_price = 0.85 * self.iso_sell_price
            
            if self.logger:
                self.logger.info(
                    f"Using quadratic pricing (no ISO agent):\n"
                    f"  - ISO Sell Price: {self.iso_sell_price:.2f} $/MWh\n"
                    f"  - ISO Buy Price: {self.iso_buy_price:.2f} $/MWh"
                )
                
        return self.iso_sell_price, self.iso_buy_price
    
    def update_realized_demand(self) -> float:
        """
        Generate realized demand from predicted demand with noise.
        
        Returns:
            Realized demand value
        """
        noise = np.random.normal(0, self.sigma)
        self.realized_demand = float(self.predicted_demand + noise)
        return self.realized_demand
    
    def calculate_market_position(
        self, 
        production: float, 
        consumption: float, 
        energy_change: float
    ) -> Dict[str, float]:
        """
        Calculate market position and financial results based on energy balance.
        
        Args:
            production: Current energy production
            consumption: Current energy consumption
            energy_change: Actual energy change in battery (positive for charging)
                          This should be the validated energy change after constraints
        """
        # Calculate net exchange with grid (+ = buying, - = selling)
        # This formula is physically correct and uses validated energy_change value
        self.net_exchange = consumption - production + energy_change
        
        # VALIDATION: Log detailed values to debug reward issues
        self.logger.debug(
            f"Market position calculation:\n"
            f"  - Consumption: {consumption:.4f} MWh\n"
            f"  - Production: {production:.4f} MWh\n" 
            f"  - Energy Change (validated): {energy_change:.4f} MWh\n"
            f"  = Net Exchange: {self.net_exchange:.4f} MWh"
        )
        
        # ADDITIONAL CHECK: Ensure net_exchange is physically valid
        # When selling (net_exchange < 0), verify we can't sell more than available
        if self.net_exchange < 0:
            max_sellable = production + max(0, -energy_change)
            if -self.net_exchange > max_sellable + 1e-6:  # Small epsilon for floating point
                self.logger.warning(
                    f"Physics violation: Attempting to sell {-self.net_exchange:.4f} MWh "
                    f"with only {max_sellable:.4f} MWh available (production: {production:.4f}, "
                    f"discharge: {-min(0, energy_change):.4f})"
                )
                # Correct to physically possible value
                self.net_exchange = -max_sellable
        
        # Calculate financial impact based on physically valid net_exchange
        if self.net_exchange > 0:  # Buying from grid
            self.revenue = -self.net_exchange * self.iso_sell_price  # Negative revenue (cost)
            self.logger.debug(
                f"BUYING: {self.net_exchange:.4f} MWh × ${self.iso_sell_price:.2f}/MWh = ${self.revenue:.2f}"
            )
        else:  # Selling to grid
            self.revenue = -self.net_exchange * self.iso_buy_price  # Positive revenue
            self.logger.debug(
                f"SELLING: {-self.net_exchange:.4f} MWh × ${self.iso_buy_price:.2f}/MWh = ${self.revenue:.2f}"
            )
        
        # Calculate grid impact
        self.net_demand = self.realized_demand + self.net_exchange
        
        # Calculate dispatch cost using thresholds from config
        self.dispatch = self.predicted_demand
        self.dispatch_cost = self.dispatch * self.dispatch_price
        
        # Calculate reserve costs for shortfall
        self.shortfall = max(0.0, self.net_demand - self.dispatch)
        self.reserve_cost = self.reserve_price * self.shortfall
        
        # Return complete market position
        return {
            'net_exchange': self.net_exchange,
            'revenue': self.revenue,
            'predicted_demand': self.predicted_demand,
            'realized_demand': self.realized_demand,
            'dispatch': self.dispatch,
            'dispatch_cost': self.dispatch_cost,
            'shortfall': self.shortfall,
            'reserve_cost': self.reserve_cost,
            'net_demand': self.net_demand,
            'iso_buy_price': self.iso_buy_price,
            'iso_sell_price': self.iso_sell_price
        }
    
    def get_state(self) -> Dict[str, float]:
        """
        Get current market state.
        
        Returns:
            Dictionary with market state information
        """
        return {
            'iso_buy_price': self.iso_buy_price,
            'iso_sell_price': self.iso_sell_price,
            'net_exchange': self.net_exchange,
            'revenue': self.revenue,
            'predicted_demand': self.predicted_demand,
            'realized_demand': self.realized_demand,
            'dispatch': self.dispatch,
            'dispatch_cost': self.dispatch_cost,
            'shortfall': self.shortfall,
            'reserve_cost': self.reserve_cost,
            'net_demand': self.net_demand
        }
    
    def reset(self) -> None:
        """
        Reset market interface state.
        """
        # Reset market prices
        self.iso_buy_price = self.min_price
        self.iso_sell_price = self.min_price
        
        # Reset market position
        self.net_exchange = 0.0
        self.revenue = 0.0
        self.dispatch_cost = 0.0
        self.reserve_cost = 0.0
        self.dispatch = 0.0
        self.shortfall = 0.0
        self.net_demand = 0.0
        
        # Reset demand
        self.predicted_demand = 0.0
        self.realized_demand = 0.0
        
        if self.logger:
            self.logger.info("Market interface reset to initial state") 