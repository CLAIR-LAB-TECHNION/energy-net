import importlib
from pathlib import Path

import numpy as np
import pytest

import prediction
from prediction import energy_predictor


class ConstantModel:
    def predict(self, features):
        return np.full(len(features), 12.5)


def hour_feature(frame):
    frame["hour"] = frame["Datetime"].dt.hour
    return frame


def test_predict_energy_is_primary_generic_function():
    value = energy_predictor.predict_energy(
        ConstantModel(),
        ["hour"],
        hour_feature,
        "2025-01-15",
        "08:00",
    )

    assert value == 12.5
    assert prediction.predict_energy is energy_predictor.predict_energy
    assert "predict_consumption" not in prediction.__all__


@pytest.mark.parametrize(
    ("target_col", "expected_column"),
    [
        ("Consumption", "Predicted_Consumption"),
        ("Production", "Predicted_Production"),
        ("Net energy", "Predicted_Net_energy"),
    ],
)
def test_day_predictions_use_target_specific_output_name(
        target_col,
        expected_column,
):
    result = energy_predictor.generate_day_predictions(
        ConstantModel(),
        ["hour"],
        hour_feature,
        start_date="2025-01-15",
        target_col=target_col,
    )

    assert expected_column in result.columns
    assert len(result) == 24
    assert result[expected_column].eq(12.5).all()


def test_primary_factory_requires_explicit_target():
    with pytest.raises(TypeError, match="target_col"):
        energy_predictor.create_energy_predictor("energy.csv")


def test_primary_factory_passes_selected_target(monkeypatch):
    captured = {}

    class StubPredictor:
        def __init__(self, csv_path, feature_engineering_fn=None, target_col=None):
            captured.update(
                csv_path=csv_path,
                feature_engineering_fn=feature_engineering_fn,
                target_col=target_col,
            )
            self.default_include_features = False

    monkeypatch.setattr(energy_predictor, "EnergyPredictor", StubPredictor)

    predictor = energy_predictor.create_energy_predictor(
        "generation.csv",
        target_col="Production",
        include_features=True,
    )

    assert captured["target_col"] == "Production"
    assert predictor.default_include_features is True


def test_legacy_predict_consumption_warns_and_delegates():
    with pytest.deprecated_call(match="predict_energy"):
        value = energy_predictor.predict_consumption(
            ConstantModel(),
            ["hour"],
            hour_feature,
            "2025-01-15",
            "08:00",
        )

    assert value == 12.5


def test_legacy_factory_warns_and_keeps_consumption_default(monkeypatch):
    captured = {}

    def fake_factory(csv_path, *, target_col, feature_engineering_fn, include_features):
        captured.update(
            csv_path=csv_path,
            target_col=target_col,
            feature_engineering_fn=feature_engineering_fn,
            include_features=include_features,
        )
        return object()

    monkeypatch.setattr(
        energy_predictor,
        "create_energy_predictor",
        fake_factory,
    )

    with pytest.deprecated_call(match="create_energy_predictor"):
        energy_predictor.create_predictor("consumption.csv")

    assert captured["target_col"] == "Consumption"


def test_removed_module_path_remains_import_compatible():
    legacy_module = importlib.import_module(
        "prediction.predicting_consumption_model"
    )

    assert legacy_module is energy_predictor
    assert not (
        Path(energy_predictor.__file__).with_name(
            "predicting_consumption_model.py"
        )
    ).exists()
