import pandas as pd
import numpy as np
from datetime import timedelta
from energy_net.consumption_prediction.predicting_consumption_model import (
    create_predictor,
    predict_consumption,
)


def generate_and_save_predictions(
        data_file='synthetic_household_consumption.csv',
        train_test_split=0.8,
        dt=0.5 / 24,  # 30 minutes in days
        prediction_horizon=48,
        output_file='consumption_predictions.csv'
):
    """
    Generate consumption predictions for the test period and save to CSV.

    Parameters:
    -----------
    data_file : str
        Path to the full consumption data CSV
    train_test_split : float
        Fraction of data to use for training (rest is test)
    dt : float
        Timestep size in days (default 0.5/24 = 30 minutes)
    prediction_horizon : int
        Number of steps ahead to predict
    output_file : str
        Path to save the predictions CSV
    """

    print("Loading data and splitting into train/test sets...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"The data file {data_file} is empty or could not be loaded.")

    # Split data
    split_idx = int(len(df) * train_test_split)
    initial_test_start = df.index[split_idx]

    # Round up to the next day at 00:00:00 if not already at midnight
    if initial_test_start.hour != 0 or initial_test_start.minute != 0 or initial_test_start.second != 0:
        test_start_date = (initial_test_start + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        print(f"Rounded test start from {initial_test_start} to {test_start_date}")
    else:
        test_start_date = initial_test_start

    # Split with the rounded date - partial day rows go to training
    train_df = df[df.index < test_start_date]
    test_df = df[df.index >= test_start_date]

    # Save training data (now includes the partial day)
    train_file = data_file.replace('.csv', '_train.csv')
    train_df.to_csv(train_file)
    print(f"Training data: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Testing data: {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")

    # Save testing data
    test_file = data_file.replace('.csv', '_test.csv')
    test_df.to_csv(test_file)

    test_end_date = test_df.index[-1]

    # Create predictor on training data only
    print("Creating consumption predictor on training data...")
    predictor = create_predictor(train_file)

    # Generate predictions for test period
    print(f"Generating predictions from {test_start_date} to {test_end_date}...")
    total_days = (test_end_date - test_start_date).days + 1
    total_steps = int(total_days / dt) + prediction_horizon

    predictions = []
    timestamps = []
    current_dt = test_start_date

    for step in range(total_steps):
        date_str = current_dt.strftime("%Y-%m-%d")
        time_str = current_dt.strftime("%H:%M")
        try:
            pred = predict_consumption(predictor, date_str, time_str)
        except Exception as e:
            pred = 0.0
            if step < 10:
                print(f"Warning: Prediction failed for {date_str} {time_str}: {e}")

        predictions.append(pred)
        timestamps.append(current_dt)
        current_dt += timedelta(days=dt)

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_consumption': predictions
    })
    predictions_df.to_csv(output_file, index=False)
    print(f"Saved {len(predictions)} predictions to {output_file}")

    return output_file, test_file


if __name__ == "__main__":
    # Generate predictions with default parameters
    predictions_file, test_file = generate_and_save_predictions()
    print(f"\nPredictions saved to: {predictions_file}")
    print(f"Test data saved to: {test_file}")