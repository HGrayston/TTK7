import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .transforms import apply_fft, apply_stft, apply_wvt, apply_wt, apply_ht

def plot_fft(df):
    results = apply_fft(df)
    for col, data in results.items():
        plt.figure()
        plt.title(f"FFT of {col}")
        plt.plot(data["freq"], np.abs(data["fft"]))
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.show()

def plot_stft(df):
    results = apply_stft(df)
    for col, data in results.items():
        plt.figure()
        plt.title(f"STFT of {col}")
        plt.pcolormesh(data["t"], data["f"], np.abs(data["Zxx"]), shading='gouraud')
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label="Magnitude")
        plt.show()

def plot_wvt(df):
    results = apply_wvt(df)
    for col, data in results.items():
        plt.figure()
        plt.title(f"Wigner-Ville Transform of {col}")
        plt.imshow(np.abs(data["tfr"]), aspect='auto', origin='lower')
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label="Magnitude")
        plt.show()

def plot_wt(df):
    results = apply_wt(df)
    for col, data in results.items():
        plt.figure()
        plt.title(f"Wigner Transform of {col}")
        plt.imshow(np.abs(data["wigner"]), aspect='auto', origin='lower')
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label="Magnitude")
        plt.show()

def plot_ht(df):
    results = apply_ht(df)
    for col, data in results.items():
        plt.figure()
        plt.title(f"Hilbert Transform (Analytic Signal) of {col}")
        plt.plot(np.real(data["analytic_signal"]), label="Real part")
        plt.plot(np.imag(data["analytic_signal"]), label="Imaginary part")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()
