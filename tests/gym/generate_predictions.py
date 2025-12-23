import pandas as pd
from energy_net.consumption_prediction.predicting_consumption_model import (
    create_predictor,
    predict_consumption,
)


def generate_and_save_predictions(
        data_file='synthetic_household_consumption.csv',
        train_test_split=0.8,
        dt=0.5 / 24,  # 30 minutes in days
        prediction_horizon=48,
        output_file='consumption_predictions.csv',
        mode='use_index'  # 'use_index' or 'grid'
):
    """
    Generate consumption predictions for the test period and save to CSV.

    mode:
      - 'use_index':  generate predictions for each timestamp present in test_df (recommended).
      - 'grid':       generate predictions on a uniform grid from test_start_date to test_end_date using dt.
    """

    print("Loading data and splitting into train/test sets...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"The data file {data_file} is empty or could not be loaded.")

    # Determine split point
    split_idx = int(len(df) * train_test_split)
    initial_test_start = df.index[split_idx]

    # Round up to the next day at 00:00:00 if not already at midnight
    # Use pandas normalize() which preserves tzinfo and sets time to 00:00
    # If initial_test_start is not exactly midnight, move to next day's midnight.
    if (initial_test_start.hour != 0 or
            initial_test_start.minute != 0 or
            initial_test_start.second != 0 or
            initial_test_start.microsecond != 0):
        # add one day then normalize -> next midnight
        test_start_date = (initial_test_start + pd.Timedelta(days=1)).normalize()
        print(f"Rounded test start from {initial_test_start} to {test_start_date}")
    else:
        test_start_date = initial_test_start.normalize()

    # Split data: partial day before test_start_date remains in training
    train_df = df[df.index < test_start_date]
    test_df = df[df.index >= test_start_date]

    # Save training and testing CSVs
    train_file = data_file.replace('.csv', '_train.csv')
    test_file = data_file.replace('.csv', '_test.csv')
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)

    print(f"Training data: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Testing data: {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")

    if len(test_df) == 0:
        raise ValueError("Test split is empty after rounding to midnight — reduce train_test_split or check data length.")

    test_end_date = test_df.index[-1]

    # Create predictor on training data only
    print("Creating consumption predictor on training data...")
    predictor = create_predictor(train_file)

    # Choose timestamps according to mode
    if mode == 'use_index':
        # Use exact timestamps from test_df (recommended). This ensures identical row counts & preserves tz.
        timestamps = test_df.index
        print(f"Using test_df.index timestamps (count={len(timestamps)})")
    elif mode == 'grid':
        # Create a uniform grid with spacing dt between start and end.
        # Use minutes to avoid floating-point freq issues:
        minutes = int(round(dt * 24 * 60))
        if minutes <= 0:
            raise ValueError("dt is too small -> minutes computed <= 0")

        freq_str = f"{minutes}T"  # e.g. "30T" for 30 minutes
        # Use inclusive='both' so that if end aligns it's included.
        timestamps = pd.date_range(start=test_start_date, end=test_end_date, freq=freq_str, inclusive='both')
        print(f"Using uniform grid from {test_start_date} to {test_end_date} with freq={freq_str} (count={len(timestamps)})")

        # If your original test_df had missing timestamps, this grid may be longer than test_df.
        if len(timestamps) != len(test_df):
            print("NOTE: grid length differs from test_df length. If you want exact match, use mode='use_index'.")
    else:
        raise ValueError("mode must be 'use_index' or 'grid'")

    # Generate predictions for each timestamp
    predictions = []
    failures = 0
    for i, ts in enumerate(timestamps):
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H:%M")
        try:
            pred = predict_consumption(predictor, date_str, time_str)
        except Exception as e:
            pred = 0.0
            failures += 1
            # keep a short log; you can expand if you want
            if failures <= 10:
                print(f"Warning: Prediction failed for {date_str} {time_str}: {e}")

        predictions.append(pred)

    # Save predictions
    predictions_df = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_consumption': predictions
    })
    predictions_df.to_csv(output_file, index=False)
    print(f"Saved {len(predictions)} predictions to {output_file} (failures={failures})")

    return output_file, test_file


if __name__ == "__main__":
    # Example: recommended
    generate_and_save_predictions(mode='use_index')

    # Example: uniform grid
    # generate_and_save_predictions(mode='grid')
