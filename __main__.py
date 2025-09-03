from pathlib import Path

from pandas import DataFrame

from src.dataframe_loader import load_csv_to_dataframe
from src.plotter import plot_single_row
from src.plot_transforms import (
    plot_fft,
    plot_stft,
    plot_wvt,
    plot_wt,
    plot_ht,
)

if __name__ == "__main__":
    dataframen: DataFrame = load_csv_to_dataframe(
        Path(__file__).parent / "signals" / "Signal3_2018.csv"
    )
    # plot_single_row(dataframen)

    plot_fft(dataframen)
    plot_stft(dataframen)
    plot_ht(dataframen)
    plot_wt(dataframen)
    plot_wvt(dataframen)


