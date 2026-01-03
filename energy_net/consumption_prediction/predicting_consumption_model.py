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
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Consumption"] = pd.to_numeric(df["Consumption"], errors="coerce")
    df = df.dropna(subset=["Datetime", "Consumption"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


# =========================
# 2. DEFAULT FEATURE ENGINEERING
# =========================

def default_time_features(df):
    """
    Default feature engineering function that creates cyclical time-based features.

    This is the baseline feature set, but users can create their own functions
    following the same signature.

    Args:
        df: DataFrame with a 'Datetime' column

    Returns:
        DataFrame with additional feature columns
    """
    dt = df["Datetime"]

    # TIME OF DAY FEATURES (30-min slots)
    slot = dt.dt.hour * 2 + (dt.dt.minute // 30)
    df["sin_time"] = np.sin(2 * pi * slot / 48)
    df["cos_time"] = np.cos(2 * pi * slot / 48)

    # DAY OF WEEK FEATURES
    dow = dt.dt.weekday
    df["sin_week"] = np.sin(2 * pi * dow / 7)
    df["cos_week"] = np.cos(2 * pi * dow / 7)

    # DAY OF YEAR FEATURES
    doy = dt.dt.dayofyear
    df["sin_year"] = np.sin(2 * pi * doy / 365.25)
    df["cos_year"] = np.cos(2 * pi * doy / 365.25)

    # WEEKEND INDICATOR
    df["is_weekend"] = (dow >= 5).astype(int)

    return df


# =========================
# 3. FEATURE DETECTOR
# =========================

def detect_features(df, target_col="Consumption", exclude_cols=None):
    """
    Automatically detect which columns should be used as features.

    Excludes the target column, the Datetime column, and any specified columns.
    Returns all remaining numeric columns as feature columns.

    Args:
        df: DataFrame to analyze
        target_col: Name of the target column to exclude
        exclude_cols: Additional columns to exclude (list)

    Returns:
        List of column names to use as features
    """
    if exclude_cols is None:
        exclude_cols = []

    # Start with columns to exclude
    excluded = set([target_col, "Datetime"] + exclude_cols)

    # Get all numeric columns that aren't excluded
    feature_cols = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]

    return feature_cols


# =========================
# 4. TRAIN GRADIENT BOOSTING MODEL
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
    X = df[feature_cols]
    y = df[target_col]

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    print("Training Gradient Boosting model...")
    model.fit(X, y)
    print("Model trained successfully!")

    return model


# =========================
# 5. PREDICTION FUNCTION
# =========================

def predict_consumption(model, feature_cols, feature_engineering_fn, date_str, time_str):
    """
    Predict energy consumption for a specific date and time.

    Args:
        model: Trained model to use for prediction
        feature_cols: List of feature column names expected by the model
        feature_engineering_fn: Function to generate features from datetime
        date_str: Date in format 'YYYY-MM-DD' (e.g., '2025-06-15')
        time_str: Time in format 'HH:MM' (e.g., '14:30')

    Returns:
        Predicted energy consumption value
    """
    dt = pd.to_datetime(f"{date_str} {time_str}")
    temp_df = pd.DataFrame({"Datetime": [dt]})

    # Apply the same feature engineering used during training
    temp_df = feature_engineering_fn(temp_df)

    X = temp_df[feature_cols]
    prediction = model.predict(X)[0]

    return prediction


# =========================
# 6. GENERATE PREDICTIONS FOR FULL DAYS
# =========================

def generate_day_predictions(model, feature_cols, feature_engineering_fn,
                             start_date, num_days=1, output_csv=None, include_features=False):
    """
    Generate hourly predictions for one or more consecutive days.

    Args:
        model: Trained model to use for predictions
        feature_cols: List of feature column names
        feature_engineering_fn: Function to generate features
        start_date: Starting date in format 'YYYY-MM-DD'
        num_days: Number of consecutive days to predict (default 1)
        output_csv: Path to save the predictions CSV file (optional)
        include_features: If True, include all feature columns in output (default False)

    Returns:
        DataFrame with predictions (and optionally features)
    """
    predictions = []
    all_features = [] if include_features else None
    current_date = pd.to_datetime(start_date)

    for day in range(num_days):
        date = current_date + pd.Timedelta(days=day)
        date_str = date.strftime('%Y-%m-%d')

        for hour in range(24):
            time_str = f"{hour:02d}:00"
            consumption = predict_consumption(
                model, feature_cols, feature_engineering_fn,
                date_str, time_str
            )

            predictions.append({
                "Date": date_str,
                "Time": time_str,
                "Day_of_Week": date.strftime('%A'),
                "Day_of_Year": date.dayofyear,
                "Hour": hour,
                "Predicted_Consumption": round(consumption, 2)
            })

            # Extract features if requested
            if include_features:
                dt = pd.to_datetime(f"{date_str} {time_str}")
                temp_df = pd.DataFrame({"Datetime": [dt]})
                temp_df = feature_engineering_fn(temp_df)
                features = temp_df[feature_cols].iloc[0].to_dict()
                all_features.append(features)

    pred_df = pd.DataFrame(predictions)

    # Add feature columns if requested
    if include_features:
        features_df = pd.DataFrame(all_features)
        pred_df = pd.concat([pred_df, features_df], axis=1)

    if output_csv:
        pred_df.to_csv(output_csv, index=False)
        if include_features:
            print(f"\nPredictions with {len(feature_cols)} features saved to: {output_csv}")
            print(f"Feature columns: {feature_cols}")
        else:
            print(f"\nPredictions saved to: {output_csv}")

    print(f"Generated {len(pred_df)} hourly predictions for {num_days} day(s)")

    return pred_df


# =========================
# 7. FLEXIBLE PREDICTOR CLASS
# =========================

class EnergyPredictor:
    """
    A flexible energy consumption predictor that allows custom feature engineering.

    Example:
        # Using default features
        predictor = EnergyPredictor("energy_data.csv")

        # Using custom features
        def my_features(df):
            df["hour"] = df["Datetime"].dt.hour
            df["month"] = df["Datetime"].dt.month
            return df

        predictor = EnergyPredictor("energy_data.csv", feature_engineering_fn=my_features)

        # Make predictions
        consumption = predictor.predict("2025-12-15", "14:00")
    """

    def __init__(self, csv_path, feature_engineering_fn=None, target_col="Consumption"):
        """
        Initialize and train the predictor.

        Args:
            csv_path: Path to training data CSV
            feature_engineering_fn: Custom feature engineering function (optional)
            target_col: Name of the target column
        """
        self.target_col = target_col
        self.feature_engineering_fn = feature_engineering_fn or default_time_features
        self.default_include_features = False  # Default behavior for predict_days

        print("=" * 70)
        print("CREATING ENERGY CONSUMPTION PREDICTOR")
        print("=" * 70)

        # Load data
        print("\n[1/3] Loading data...")
        df = load_data(csv_path)
        print(f"      Loaded {len(df)} records from {df['Datetime'].min()} to {df['Datetime'].max()}")

        # Apply feature engineering
        print("\n[2/3] Creating features...")
        df = self.feature_engineering_fn(df)

        # Detect features automatically
        self.feature_cols = detect_features(df, target_col=self.target_col)
        print(f"      Detected {len(self.feature_cols)} features: {self.feature_cols}")

        # Train model
        print("\n[3/3] Training model...")
        self.model = train_gradient_boosting(df, self.feature_cols, target_col=self.target_col)

        print("\n" + "=" * 70)
        print("PREDICTOR READY!")
        print("=" * 70)

    def predict(self, date_str, time_str):
        """
        Predict consumption for a specific date and time.

        Args:
            date_str: Date in format 'YYYY-MM-DD'
            time_str: Time in format 'HH:MM'

        Returns:
            Predicted consumption value
        """
        return predict_consumption(
            self.model,
            self.feature_cols,
            self.feature_engineering_fn,
            date_str,
            time_str
        )

    def predict_days(self, start_date, num_days=1, output_csv=None, include_features=None):
        """
        Generate predictions for multiple days.

        Args:
            start_date: Starting date in format 'YYYY-MM-DD'
            num_days: Number of days to predict
            output_csv: Optional path to save results
            include_features: If True, include all feature columns in output.
                            If None, uses default_include_features set during creation.

        Returns:
            DataFrame with predictions (and optionally features)
        """
        # Use default if not explicitly specified
        if include_features is None:
            include_features = self.default_include_features

        return generate_day_predictions(
            self.model,
            self.feature_cols,
            self.feature_engineering_fn,
            start_date,
            num_days,
            output_csv,
            include_features
        )


# =========================
# 8. EXAMPLE CUSTOM FEATURES
# =========================

def advanced_time_features(df):
    """
    Example of more advanced feature engineering.

    Users can create their own functions like this to add custom features.
    """
    dt = df["Datetime"]

    # All the default features
    df = default_time_features(df)

    # Additional features
    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_morning_peak"] = ((dt.dt.hour >= 6) & (dt.dt.hour <= 9)).astype(int)
    df["is_evening_peak"] = ((dt.dt.hour >= 17) & (dt.dt.hour <= 21)).astype(int)
    df["season"] = (dt.dt.month % 12 + 3) // 3  # 1=winter, 2=spring, 3=summer, 4=fall

    return df


# =========================
# 9. Main Predictor Function
# =========================

def create_predictor(csv_path, feature_engineering_fn=None, include_features=False):
    """
    Backward compatible function that creates a predictor.

    Args:
        csv_path: Path to training data
        feature_engineering_fn: Optional custom feature function
        include_features: If True, predict_days will include features by default (default False)

    Returns:
        EnergyPredictor instance
    """
    predictor = EnergyPredictor(csv_path, feature_engineering_fn=feature_engineering_fn)
    predictor.default_include_features = include_features
    return predictor


def save_predictions_with_train_test_split(
        data_file,
        train_test_split=0.8,
        dt=0.5 / 24,  # 30 minutes in days
        prediction_horizon=48,
        output_file='consumption_predictions.csv',
        mode='use_index',  # 'use_index' or 'grid'
        feature_engineering_fn=None,
        target_col="Consumption",
        include_features=True
):
    """
    Generate consumption predictions for the test period and save to CSV.

    Args:
        data_file: Path to the data CSV file
        train_test_split: Fraction of data to use for training (default 0.8)
        dt: Time step in days for grid mode (default 0.5/24 = 30 minutes)
        prediction_horizon: Not currently used, kept for compatibility
        output_file: Path to save predictions CSV
        mode: 'use_index' (use test timestamps) or 'grid' (uniform grid)
        feature_engineering_fn: Custom feature engineering function (optional)
        target_col: Name of the target column (default "Consumption")
        include_features: If True, include all feature columns in output (default True)

    Returns:
        Tuple of (output_file path, test_file path)
    """

    print("Loading data and splitting into train/test sets...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"The data file {data_file} is empty or could not be loaded.")

    # Determine split point
    split_idx = int(len(df) * train_test_split)
    initial_test_start = df.index[split_idx]

    # Round up to the next day at 00:00:00 if not already at midnight
    if (initial_test_start.hour != 0 or
            initial_test_start.minute != 0 or
            initial_test_start.second != 0 or
            initial_test_start.microsecond != 0):
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
        raise ValueError(
            "Test split is empty after rounding to midnight — reduce train_test_split or check data length.")

    test_end_date = test_df.index[-1]

    # Create predictor on training data only using the flexible interface
    print("Creating consumption predictor on training data...")
    predictor = EnergyPredictor(
        train_file,
        feature_engineering_fn=feature_engineering_fn,
        target_col=target_col
    )

    # Choose timestamps according to mode
    if mode == 'use_index':
        timestamps = test_df.index
        print(f"Using test_df.index timestamps (count={len(timestamps)})")
    elif mode == 'grid':
        minutes = int(round(dt * 24 * 60))
        if minutes <= 0:
            raise ValueError("dt is too small -> minutes computed <= 0")

        freq_str = f"{minutes}T"
        timestamps = pd.date_range(start=test_start_date, end=test_end_date, freq=freq_str, inclusive='both')
        print(
            f"Using uniform grid from {test_start_date} to {test_end_date} with freq={freq_str} (count={len(timestamps)})")

        if len(timestamps) != len(test_df):
            print("NOTE: grid length differs from test_df length. If you want exact match, use mode='use_index'.")
    else:
        raise ValueError("mode must be 'use_index' or 'grid'")

    # Generate predictions (and optionally features) for each timestamp
    predictions = []
    all_features = [] if include_features else None
    failures = 0

    for i, ts in enumerate(timestamps):
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H:%M")
        try:
            pred = predictor.predict(date_str, time_str)

            # Also extract the features if requested
            if include_features:
                temp_df = pd.DataFrame({"Datetime": [ts]})
                temp_df = predictor.feature_engineering_fn(temp_df)
                features = temp_df[predictor.feature_cols].iloc[0].to_dict()
                all_features.append(features)

        except Exception as e:
            pred = 0.0
            if include_features:
                features = {col: 0.0 for col in predictor.feature_cols}
                all_features.append(features)
            failures += 1
            if failures <= 10:
                print(f"Warning: Prediction failed for {date_str} {time_str}: {e}")

        predictions.append(pred)

    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_consumption': predictions
    })

    # Add all feature columns if requested
    if include_features:
        features_df = pd.DataFrame(all_features)
        predictions_df = pd.concat([predictions_df, features_df], axis=1)
        print(
            f"Saved {len(predictions)} predictions with {len(predictor.feature_cols)} features to {output_file} (failures={failures})")
        print(f"Feature columns included: {predictor.feature_cols}")
    else:
        print(f"Saved {len(predictions)} predictions to {output_file} (failures={failures})")

    predictions_df.to_csv(output_file, index=False)

    return output_file, test_file