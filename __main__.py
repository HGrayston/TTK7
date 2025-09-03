from pathlib import Path

from pandas import DataFrame

from src.dataframe_loader import load_csv_to_dataframe
from src.plotter import plot_single_row
import numpy as np
from src.plot_transforms import (
    plot_fft,
    plot_stft,
    plot_wvt,
    plot_wt,
    plot_ht,
)

# Load the final signal
dataframen: DataFrame = load_csv_to_dataframe(
    Path(__file__).parent / "signals" / "Signal3_2018.csv"
)

# Analyse stationarity (simple visual inspection)
plot_single_row(dataframen)

# FFT analysis
print("FFT Analysis:")
plot_fft(dataframen)
# Reflect: FFT is best for stationary signals; if components change over time, FFT may not capture them well.

# STFT analysis (window size selection)
print("STFT Analysis:")
# Window size: If frequency components change rapidly, use a smaller window; for slow changes, use a larger window.
plot_stft(dataframen)

# HT, WT, WVT analysis
print("Hilbert Transform Analysis:")
plot_ht(dataframen)
print("Wavelet Transform Analysis:")
plot_wt(dataframen)
print("Wavelet-Von Transform Analysis:")
plot_wvt(dataframen)
# Reflect: STFT/WT/WVT are better for non-stationary signals.

# Add an offset and repeat analysis
offset_df = dataframen + 5
print("Analysis with Offset:")
plot_fft(offset_df)
plot_stft(offset_df)
plot_ht(offset_df)
plot_wt(offset_df)
plot_wvt(offset_df)

# Add white noise and repeat analysis
noise_df = dataframen + np.random.normal(0, 1, dataframen.shape)
print("Analysis with White Noise:")
plot_fft(noise_df)
plot_stft(noise_df)
plot_ht(noise_df)
plot_wt(noise_df)
plot_wvt(noise_df)

# Add linearly time-varying frequency component and repeat analysis
t = np.arange(len(dataframen))
freq_component = np.sin(0.01 * t * t)
freq_df = dataframen.copy()
freq_df.iloc[:, 0] += freq_component
print("Analysis with Time-Varying Frequency Component:")
plot_fft(freq_df)
plot_stft(freq_df)
plot_ht(freq_df)
plot_wt(freq_df)
plot_wvt(freq_df)

# Add offset and white noise and repeat analysis
offset_noise_df = dataframen + 5 + np.random.normal(0, 1, dataframen.shape)
print("Analysis with Offset and White Noise:")
plot_fft(offset_noise_df)
plot_stft(offset_noise_df)
plot_ht(offset_noise_df)
plot_wt(offset_noise_df)
plot_wvt(offset_noise_df)


