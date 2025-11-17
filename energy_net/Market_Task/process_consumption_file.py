import pandas as pd
import datetime
from pandas.tseries.frequencies import to_offset
from typing import Optional
def process_consumption_file(
    file_path: str,
    output_csv: Optional[str] = None,
    resample_freq: Optional[str] = '1T',
    agg_method: str = 'mean',
    interp_method: str = 'linear'
) -> pd.DataFrame:
    """
    Read an energy consumption Excel file and process it into a clean DataFrame.

    Args:
        file_path: Path to the Excel file (expects same layout you provided).
        output_csv: Optional path to save the processed CSV file.
        resample_freq: Pandas offset string for resampling (e.g. '1T', '30T', 'H').
                       If None, no resampling is performed (keeps original rows).
        agg_method: Method to aggregate when downsampling (e.g. 'mean', 'sum', etc.).
        interp_method: Interpolation method when upsampling (passed to .interpolate()).

    Returns:
        A DataFrame indexed by Datetime with a single 'Consumption' column.
    """
    # Step 1: Read Excel file and rename columns
    df = pd.read_excel(file_path, header=1)
    df.columns = ['Date', 'Time', 'Forecast_Day_Before', 'Forecast_Updated', 'Consumption']

    # Step 2: Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Step 3: Convert Time to Timedelta
    def time_to_timedelta(x):
        if pd.isna(x):
            return pd.NaT
        if isinstance(x, datetime.time):
            return pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
        if isinstance(x, (datetime.datetime, pd.Timestamp)):
            return pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second)
        return pd.to_timedelta(str(x))

    times = df['Time'].apply(time_to_timedelta)

    # Step 4: Combine Date + Time
    df['Datetime'] = df['Date'] + times

    # Step 5: Warn if any Datetime parsing failed
    if df['Datetime'].isna().any():
        print("Warning: Some Datetime values could not be parsed:")
        print(df[df['Datetime'].isna()][['Date', 'Time']].head())

    # Step 6: Set Datetime as index and keep only 'Consumption'
    df.set_index('Datetime', inplace=True)
    df = df[['Consumption']]
    df = df.sort_index()  # Ensure chronological order

    # Step 7: Resample if requested
    if resample_freq is not None:
        try:
            req_off = to_offset(resample_freq)

            if len(df.index) >= 2:
                median_diff = df.index.to_series().diff().median()
                median_nanos = int(median_diff.value)
            else:
                median_nanos = None

            if median_nanos is None:
                # Fallback: interpolate
                df = df.resample(resample_freq).interpolate(method=interp_method)
            else:
                try:
                    req_nanos = int(req_off.nanos)
                except Exception:
                    req_nanos = int(pd.to_timedelta(req_off.freqstr or resample_freq).value)

                if req_nanos >= median_nanos:
                    # Downsample: aggregate
                    df = df.resample(resample_freq).agg({'Consumption': agg_method})
                else:
                    # Upsample: interpolate
                    df = df.resample(resample_freq).interpolate(method=interp_method)
        except Exception:
            df = df.resample(resample_freq).interpolate(method=interp_method)

    # Step 8: Save CSV if requested
    if output_csv:
        df.to_csv(output_csv)
        print(f"Processed data saved to {output_csv}")

    return df

if __name__ == "__main__":
    # Example 1: Process and resample to 1-minute intervals (default)
    df1 = process_consumption_file(
        'SystemDemand_02_11_2025-06_11_2025_he-IL.xlsx',
        output_csv='processed_1min.csv',
        resample_freq='1T'
    )
    print("First few rows (1-minute resample):")
    print(df1.head())

    # Example 2: Downsample to 30-minute intervals
    df30 = process_consumption_file(
        'SystemDemand_02_11_2025-06_11_2025_he-IL.xlsx',
        output_csv='processed_30min.csv',
        resample_freq='30T',
        agg_method='mean'
    )
    print("First few rows (30-minute resample):")
    print(df30.head())

    # Example 3: Keep original timestamps (no resampling), also save to CSV
    df_orig = process_consumption_file(
        'SystemDemand_02_11_2025-06_11_2025_he-IL.xlsx',
        resample_freq=None,
        output_csv='processed_original.csv'
    )
    print("First few rows (original timestamps):")
    print(df_orig.head())
