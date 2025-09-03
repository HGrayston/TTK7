import numpy as np
from scipy.fft import fft
from scipy.signal import hilbert, stft
import pandas as pd


def apply_fft(df):
    """
    Apply FFT to each data column (index 1 and onwards).
    Returns a dict of frequency and transformed data for each column.
    """
    x = df.iloc[:, 0].values
    results = {}
    for col in df.columns[1:]:
        y = df[col].values
        freq = np.fft.fftfreq(len(x), d=(x[1] - x[0]))
        results[col] = {"freq": freq, "fft": fft(y)}
    return results


def apply_stft(df, nperseg=256):
    """
    Apply STFT to each data column.
    Returns a dict of (f, t, Zxx) for each column.
    """
    x = df.iloc[:, 0].values
    results = {}
    for col in df.columns[1:]:
        y = df[col].values
        fs = 1 / (x[1] - x[0])
        f, t, Zxx = stft(y, fs=fs, nperseg=nperseg)
        results[col] = {"f": f, "t": t, "Zxx": Zxx}
    return results


def apply_wvt(df):
    """
    Apply Wigner-Ville Transform to each data column.
    Returns a dict of time-frequency representations for each column.
    """

    def wigner_ville(signal):
        N = len(signal)
        tfr = np.zeros((N, N), dtype=complex)
        for t in range(N):
            for tau in range(-min(t, N - t - 1), min(t, N - t - 1) + 1):
                tfr[t, tau + N // 2] = signal[t + tau] * np.conj(signal[t - tau])
        tfr = np.fft.fftshift(np.fft.fft(tfr, axis=1), axes=1)
        return tfr

    results = {}
    for col in df.columns[1:]:
        y = df[col].values
        tfr = wigner_ville(y)
        results[col] = {"tfr": tfr}
    return results


def apply_wt(df):
    """
    Apply Wigner Transform to each data column.
    Returns a dict of Wigner distributions for each column.
    """

    def wigner(signal):
        N = len(signal)
        w = np.zeros((N, N), dtype=complex)
        for t in range(N):
            for tau in range(-min(t, N - t - 1), min(t, N - t - 1) + 1):
                w[t, tau + N // 2] = signal[t + tau] * np.conj(signal[t - tau])
        w = np.fft.fftshift(np.fft.fft(w, axis=1), axes=1)
        return w

    results = {}
    for col in df.columns[1:]:
        y = df[col].values
        w = wigner(y)
        results[col] = {"wigner": w}
    return results


def apply_ht(df):
    """
    Apply Hilbert Transform to each data column.
    Returns a dict of analytic signal for each column.
    """
    results = {}
    for col in df.columns[1:]:
        y = df[col].values
        analytic_signal = hilbert(y)
        results[col] = {"analytic_signal": analytic_signal}
    return results


