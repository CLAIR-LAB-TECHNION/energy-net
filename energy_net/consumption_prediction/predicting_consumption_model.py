import pandas as pd
import numpy as np
from math import sin, cos, pi
from sklearn.ensemble import GradientBoostingRegressor


# =========================
# 1. LOAD AND CLEAN DATA
# =========================

def load_data(csv_path):
    """
    Load energy consumption data from CSV file.

    This function reads the CSV, converts the Datetime and Consumption columns
    to appropriate data types, removes any rows with missing values, and sorts
    the data chronologically.

    Args:
        csv_path: Path to the CSV file containing energy consumption data

    Returns:
        DataFrame with cleaned and sorted energy consumption data
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Convert the Datetime column to proper datetime format
    # errors='coerce' will convert invalid dates to NaT (Not a Time)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # Convert the Consumption column to numeric format
    # errors='coerce' will convert invalid numbers to NaN (Not a Number)
    df["Consumption"] = pd.to_numeric(df["Consumption"], errors="coerce")

    # Remove any rows where Datetime or Consumption is missing
    # This ensures we only work with complete, valid data
    df = df.dropna(subset=["Datetime", "Consumption"])

    # Sort the data by date/time in ascending order
    # This is important for time-series modeling
    df = df.sort_values("Datetime").reset_index(drop=True)

    return df


# =========================
# 2. FEATURE ENGINEERING
# =========================

def add_time_features(df):
    """
    Create cyclical time-based features for the model.

    We use sine and cosine transformations to encode time features cyclically.
    This helps the model understand that 11:59 PM is close to 12:00 AM,
    Sunday is close to Monday, and December 31st is close to January 1st.

    Args:
        df: DataFrame with a 'Datetime' column

    Returns:
        DataFrame with additional time-based feature columns
    """
    dt = df["Datetime"]

    # --- TIME OF DAY FEATURES ---
    # Calculate which 30-minute slot of the day this is (0 to 47)
    # For example: 00:00 = 0, 00:30 = 1, 01:00 = 2, ..., 23:30 = 47
    slot = dt.dt.hour * 2 + (dt.dt.minute // 30)

    # Convert to sine and cosine to make it cyclical
    # This way, slot 47 (23:30) is numerically close to slot 0 (00:00)
    df["sin_time"] = np.sin(2 * pi * slot / 48)
    df["cos_time"] = np.cos(2 * pi * slot / 48)

    # --- DAY OF WEEK FEATURES ---
    # Get the day of week (0 = Monday, 6 = Sunday)
    dow = dt.dt.weekday

    # Convert to sine and cosine to make it cyclical
    # This way, Sunday (6) is numerically close to Monday (0)
    df["sin_week"] = np.sin(2 * pi * dow / 7)
    df["cos_week"] = np.cos(2 * pi * dow / 7)

    # --- DAY OF YEAR FEATURES ---
    # Get the day of year (1 to 365/366)
    doy = dt.dt.dayofyear

    # Convert to sine and cosine to make it cyclical
    # This way, December 31st is numerically close to January 1st
    df["sin_year"] = np.sin(2 * pi * doy / 365.25)
    df["cos_year"] = np.cos(2 * pi * doy / 365.25)

    # --- WEEKEND INDICATOR ---
    # Create a binary flag: 1 for weekends (Saturday/Sunday), 0 for weekdays
    # This captures the different energy consumption patterns on weekends
    df["is_weekend"] = (dow >= 5).astype(int)

    return df


# =========================
# 3. TRAIN GRADIENT BOOSTING MODEL
# =========================

def train_gradient_boosting(df, feature_cols, target_col="Consumption"):
    """
    Train a Gradient Boosting model on the entire dataset.

    Args:
        df: DataFrame with features and target variable
        feature_cols: List of column names to use as features
        target_col: Name of the target column to predict

    Returns:
        Trained Gradient Boosting model
    """
    # Separate features (X) and target variable (y)
    X = df[feature_cols]
    y = df[target_col]

    # Create and configure the Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=300,  # Build 300 sequential trees
        learning_rate=0.05,  # Learn slowly for better generalization
        max_depth=5,  # Limit tree depth to prevent overfitting
        random_state=42  # For reproducible results
    )

    # Train the model on all available data
    print("Training Gradient Boosting model...")
    model.fit(X, y)
    print("Model trained successfully!")

    return model


# =========================
# 4. PREDICTION FUNCTION
# =========================

def predict_consumption(model, date_str, time_str):
    """
    Predict energy consumption for a specific date and time.

    Args:
        model: Trained model to use for prediction
        date_str: Date in format 'YYYY-MM-DD' (e.g., '2025-06-15')
        time_str: Time in format 'HH:MM' (e.g., '14:30')

    Returns:
        Predicted energy consumption value
    """
    # Combine date and time into a datetime object
    dt = pd.to_datetime(f"{date_str} {time_str}")

    # Create a small DataFrame with this single datetime
    temp_df = pd.DataFrame({"Datetime": [dt]})

    # Generate the same time features we used for training
    temp_df = add_time_features(temp_df)

    # Extract just the features needed for prediction
    feature_cols = [
        "sin_time", "cos_time",
        "sin_week", "cos_week",
        "sin_year", "cos_year",
        "is_weekend"
    ]

    X = temp_df[feature_cols]

    # Make the prediction
    prediction = model.predict(X)[0]

    return prediction


# =========================
# 5. GENERATE PREDICTIONS FOR FULL DAYS
# =========================

def generate_day_predictions(model, start_date, num_days=1, output_csv=None):
    """
    Generate hourly predictions for one or more consecutive days.

    Args:
        model: Trained model to use for predictions
        start_date: Starting date in format 'YYYY-MM-DD'
        num_days: Number of consecutive days to predict (default 1)
        output_csv: Path to save the predictions CSV file (optional, default None)

    Returns:
        DataFrame with predictions for every hour
    """
    # Create a list to store all predictions
    predictions = []

    # Convert start date to datetime
    current_date = pd.to_datetime(start_date)

    # Loop through each day
    for day in range(num_days):
        date = current_date + pd.Timedelta(days=day)
        date_str = date.strftime('%Y-%m-%d')

        # Loop through each hour of the day (0-23)
        for hour in range(24):
            time_str = f"{hour:02d}:00"

            # Get prediction for this specific time
            consumption = predict_consumption(model, date_str, time_str)

            predictions.append({
                "Date": date_str,
                "Time": time_str,
                "Day_of_Week": date.strftime('%A'),
                "Day_of_Year": date.dayofyear,
                "Hour": hour,
                "Predicted_Consumption": round(consumption, 2)
            })

    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)

    # Save to CSV only if output path is provided
    if output_csv:
        pred_df.to_csv(output_csv, index=False)
        print(f"\nPredictions saved to: {output_csv}")

    print(f"Generated {len(pred_df)} hourly predictions for {num_days} day(s)")

    return pred_df


# =========================
# 6. SIMPLE API - CREATE PREDICTOR FROM FILE
# =========================

def create_predictor(csv_path):
    """
    One-stop function to create a trained predictor from a CSV file.

    This handles all the data loading, feature engineering, and model training
    so the user doesn't have to worry about the details.

    Args:
        csv_path: Path to the CSV file containing historical energy consumption data

    Returns:
        Trained model ready to make predictions

    Example:
        predictor = create_predictor("energy_data.csv")
        consumption = predictor.predict("2025-12-15", "14:00")
    """
    print("=" * 70)
    print("CREATING ENERGY CONSUMPTION PREDICTOR")
    print("=" * 70)

    # Load and prepare the data
    print("\n[1/3] Loading data...")
    df = load_data(csv_path)
    print(f"      Loaded {len(df)} records from {df['Datetime'].min()} to {df['Datetime'].max()}")

    # Create time-based features
    print("\n[2/3] Creating time-based features...")
    df = add_time_features(df)

    feature_cols = [
        "sin_time", "cos_time",
        "sin_week", "cos_week",
        "sin_year", "cos_year",
        "is_weekend"
    ]
    print(f"      Created {len(feature_cols)} features")

    # Train the model
    print("\n[3/3] Training Gradient Boosting model on full dataset...")
    model = train_gradient_boosting(df, feature_cols)

    print("\n" + "=" * 70)
    print("PREDICTOR READY!")
    print("=" * 70)

    return model
def save_predictions_with_train_test_split(
        data_file,
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