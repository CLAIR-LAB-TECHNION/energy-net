"""Target-neutral energy forecasting tools."""

import sys

from . import energy_predictor as _energy_predictor
from .energy_predictor import (
    EnergyPredictor,
    advanced_time_features,
    create_energy_predictor,
    default_time_features,
    detect_features,
    generate_day_predictions,
    load_data,
    predict_energy,
    prediction_column_name,
    save_energy_predictions_with_train_test_split,
    train_gradient_boosting,
)

# Import-only compatibility for callers using the removed module path.
sys.modules[f"{__name__}.predicting_consumption_model"] = _energy_predictor

__all__ = [
    "EnergyPredictor",
    "advanced_time_features",
    "create_energy_predictor",
    "default_time_features",
    "detect_features",
    "generate_day_predictions",
    "load_data",
    "predict_energy",
    "prediction_column_name",
    "save_energy_predictions_with_train_test_split",
    "train_gradient_boosting",
]
