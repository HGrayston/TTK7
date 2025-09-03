from pathlib import Path

from pandas import DataFrame

from src.dataframe_loader import load_csv_to_dataframe
from src.plotter import plot_single_row

if __name__ == "__main__":
    dataframen: DataFrame = load_csv_to_dataframe(
        Path(__file__).parent / "signals" / "Signal3_2018.csv"
    )
    plot_single_row(dataframen)
