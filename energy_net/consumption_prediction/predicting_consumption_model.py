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