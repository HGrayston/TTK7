import pandas as pd

def load_csv_to_dataframe(csv_path):
    """
    Loads a CSV file where data is in a single row (across columns),
    reshapes it so that each value becomes a row, and adds an integer index.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with 'Index' and 'Data' columns.
    """
    df = pd.read_csv(csv_path, header=None)
    # Transpose so that each value becomes a row
    df = df.T
    df.columns = ['Data']
    df['Index'] = range(len(df))
    df = df[['Index', 'Data']]
    return df
