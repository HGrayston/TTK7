
import matplotlib.pyplot as plt

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