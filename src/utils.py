
import matplotlib.pyplot as plt
import pandas as pd

def plot_single_row(df):
    """
    Plots the values of a single-row DataFrame as a bar chart.
    """
    if df.shape[0] != 1:
        raise ValueError("DataFrame must have exactly one row.")
    row = df.iloc[0]
    plt.figure(figsize=(8, 4))
    plt.bar(row.index, row.values)
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Single Row DataFrame Plot')
    plt.tight_layout()
    plt.show()




def load_csv_to_dataframe(csv_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    return pd.read_csv(csv_path)