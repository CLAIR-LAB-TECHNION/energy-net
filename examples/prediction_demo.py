from pathlib import Path

from prediction.energy_predictor import (
    create_energy_predictor,
    prediction_column_name,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    REPOSITORY_ROOT
    / "tests"
    / "gym"
    / "data_for_tests"
    / "synthetic_household_consumption.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    print("=" * 70)
    print("ENERGY TARGET FORECASTING - DEMO")
    print("=" * 70)

    target_col = "Consumption"
    predictor = create_energy_predictor(DATA_FILE, target_col=target_col)
    output_col = prediction_column_name(target_col)

    print("\n--- Example 1: Single Prediction ---")
    single_prediction = predictor.predict("2025-12-15", "14:00")
    print(
        f"Predicted {target_col} for 2025-12-15 at 14:00: "
        f"{single_prediction:.2f}"
    )

    print("\n--- Example 2: One Day of Predictions ---")
    one_day_df = predictor.predict_days(
        start_date="2025-12-10",
        num_days=1,
    )
    print(one_day_df.head(5).to_string(index=False))

    print("\n--- Example 3: One Month of Predictions ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "energy_predictions_january.csv"
    month_df = predictor.predict_days(
        start_date="2026-01-01",
        num_days=31,
        output_csv=output_file,
    )

    print(f"\nTotal predictions: {len(month_df)}")
    print(f"Average predicted {target_col}: {month_df[output_col].mean():.2f}")
    print(f"Minimum predicted {target_col}: {month_df[output_col].min():.2f}")
    print(f"Maximum predicted {target_col}: {month_df[output_col].max():.2f}")
    print(f"Saved predictions to: {output_file}")


if __name__ == "__main__":
    main()
