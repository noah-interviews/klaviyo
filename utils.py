import pandas as pd
import numpy as np

def generate_15min_intervals(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate a DataFrame with 15-minute intervals between two dates, excluding weekends.

    Parameters:
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        A pd.DataFrame with a single column 'DateTime' containing datetime values
        every 15 minutes on weekdays within the specified date range.
    """
    # Generate date range with 15-minute frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq='15T')

    # Filter out weekends
    date_range = date_range[date_range.weekday < 5]  # Weekdays only

    # Create DataFrame
    df = pd.DataFrame({'interval_et': date_range})

    return df


def collect_and_prep_prophet_data(csv_file_path: str,
                          time_col: str = 'interval_et', 
                          series_col: str = 'chats',
                          impute_series_col: bool = True,
                          log_transform_series_col: bool = True
) -> pd.DataFrame:
    """
    Collects and prepares time series data from a CSV file for Prophet modeling.

    Parameters:
        csv_file_path (str): The file path to the CSV file containing the time series data.
        time_col (str): The name of the column containing the timestamp information.
            Default is 'interval_et'.
        series_col (str): The name of the column containing the time series data.
            Default is 'chats'.
        impute_series_col (bool): Whther or not to impute missing `series_col` values with 0
        log_transform_series_col (bool): Whether or not to log transform the `series_col`

    Returns:
        A pd.DataFrame containing the prepared time series data with 15-minute intervals,
        missing values imputed as zero, and datetime column converted to datetime format.
    """
    # Read in data
    df = pd.read_csv(csv_file_path)
    
    # Convert column to datetimes
    df[time_col] = pd.to_datetime(df[time_col])

    # Create all 15-minute timestamps across the data
    date_base_df = generate_15min_intervals(
        str(df[time_col].dt.date.min()),
        str(df[time_col].dt.date.max())
    )

    if log_transform_series_col:
        impute_val = 0.001
    else:
        impute_val = 0.0
    
    # Join actual data and impute missing values as zero
    df = date_base_df.merge(df, on=time_col, how='left')
    if impute_series_col:
        df[series_col] = df[series_col].fillna(impute_val)

    # Rename columns for use in Prophet
    df = df.rename(columns={time_col: 'ds', series_col: 'y'})

    # log(y)
    if log_transform_series_col:
        df['y'] = np.log(df['y'])

    return df