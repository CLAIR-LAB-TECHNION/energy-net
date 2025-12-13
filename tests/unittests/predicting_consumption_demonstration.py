from energy_net.consumption_prediction.predicting_consumption_model import (
    create_predictor,
    predict_consumption,
    generate_day_predictions
)


def main():
    print("=" * 70)
    print("ENERGY CONSUMPTION FORECASTING - DEMO")
    print("=" * 70)

    # =========================
    # STEP 1: CREATE PREDICTOR (ONE LINE!)
    # =========================
    # This handles everything: loading data, feature engineering, and training
    csv_path = "/Users/michaelwein/EnergyNetClean/tests/gym/SystemDemand_30min_2023-2025.csv"
    predictor = create_predictor(csv_path)

    # =========================
    # STEP 2: MAKE PREDICTIONS
    # =========================
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS")
    print("=" * 70)

    # Example 1: Single prediction for a specific date and time
    print("\n--- Example 1: Single Prediction ---")
    single_prediction = predict_consumption(predictor, "2025-12-15", "14:00")
    print(f"Predicted consumption for 2025-12-15 at 14:00: {single_prediction:.2f}")

    # Example 2: Generate predictions for one day (without saving to CSV)
    print("\n--- Example 2: One Day of Predictions (no CSV) ---")
    one_day_df = generate_day_predictions(
        model=predictor,
        start_date="2025-12-10",
        num_days=1
    )
    print("\nFirst 5 hours:")
    print(one_day_df.head(5).to_string(index=False))


    # Example 3: Generate predictions for a month
    print("\n--- Example 3: One Month of Predictions ---")
    month_df = generate_day_predictions(
        model=predictor,
        start_date="2026-01-01",
        num_days=31,
        output_csv="logs/energy_predictions_january.csv"
    )

    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"  Total predictions: {len(month_df)}")
    print(f"  Average predicted consumption: {month_df['Predicted_Consumption'].mean():.2f}")
    print(f"  Minimum predicted consumption: {month_df['Predicted_Consumption'].min():.2f}")
    print(f"  Maximum predicted consumption: {month_df['Predicted_Consumption'].max():.2f}")

    # =========================
    # SUMMARY
    # =========================
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nSimple 3-step usage:")
    print("  1. predictor = create_predictor('your_data.csv')")
    print("  2. consumption = predict_consumption(predictor, '2025-12-15', '14:00')")
    print("\nCSV files created:")
    print("  • energy_predictions_january.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()