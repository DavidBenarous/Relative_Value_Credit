import pandas as pd

def load_data(ticker: str) -> pd.DataFrame:
    """
    Loads time series data for a given ticker from a CSV file.

    Args:
        ticker: The ticker symbol of the instrument.

    Returns:
        A pandas DataFrame with the time series data.
    """
    filepath = f"src/data/{ticker}.csv"
    df = pd.read_csv(filepath, index_col="DATE", parse_dates=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the time series data by handling missing values.

    Args:
        df: The input DataFrame.

    Returns:
        The cleaned DataFrame.
    """
    # Forward-fill missing values
    return df.ffill()

def synchronize_data(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronizes two time series DataFrames by taking the intersection of their indices.

    Args:
        df1: The first DataFrame.
        df2: The second DataFrame.

    Returns:
        A tuple of two DataFrames with a common index.
    """
    common_index = df1.index.intersection(df2.index)
    return df1.loc[common_index], df2.loc[common_index]
