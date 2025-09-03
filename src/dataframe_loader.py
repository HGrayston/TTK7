import pandas as pd

def load_csv_to_dataframe(csv_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    return pd.read_csv(csv_path)