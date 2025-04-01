# iso_factory.py

from typing import Dict, Any
from energy_net.market.iso.hourly_pricing_iso import HourlyPricingISO
from energy_net.market.iso.dynamic_pricing_iso import DynamicPricingISO
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO
from energy_net.market.iso.random_pricing_iso import RandomPricingISO
from energy_net.market.iso.time_of_use_pricing_iso import TimeOfUsePricingISO
from energy_net.market.iso.iso_base import ISOBase


def iso_factory(iso_type: str, iso_parameters: Dict[str, Any]) -> ISOBase:
    """
    Factory function to create ISO instances based on the iso_type.
    
    Args:
        iso_type (str): The type of ISO to create.
        iso_parameters (Dict[str, Any]): Parameters required to instantiate the ISO.
    
    Returns:
        ISOBase: An instance of the specified ISO.
    
    Raises:
        ValueError: If the iso_type is unknown.
    """
    iso_type = iso_type.strip()
    iso_type_mapping = {
        'HourlyPricingISO': HourlyPricingISO,
        'DynamicPricingISO': DynamicPricingISO,
        'QuadraticPricingISO': QuadraticPricingISO,
        'RandomPricingISO': RandomPricingISO,
        'TimeOfUsePricingISO': TimeOfUsePricingISO
    }
    
    if iso_type in iso_type_mapping:
        iso_class = iso_type_mapping[iso_type]
        try:
            return iso_class(**iso_parameters)
        except TypeError as e:
            raise TypeError(f"Error initializing {iso_type}: {e}") from e
    else:
        raise ValueError(f"Unknown ISO type: {iso_type}")
