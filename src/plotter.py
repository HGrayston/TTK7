
import matplotlib.pyplot as plt
from pandas import DataFrame

def plot_single_row(df:DataFrame):
    """
    Plots the values of a single-row DataFrame as a bar chart.
    """
    plt.figure(figsize=(14, 4))
    plt.plot(df["Index"], df["Data"])
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Single Row DataFrame Plot')
    plt.tight_layout()
    plt.show()