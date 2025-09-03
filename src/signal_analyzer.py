import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, stft


class SignalAnalyzer:
    @staticmethod
    def hilbert_transform(signal):
        """Hilbert transform (HT)."""
        signal = np.asarray(signal)
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)

        return {
            "type": "HT",
            "analytic_signal": analytic_signal,
            "amplitude_envelope": amplitude_envelope,
            "instantaneous_phase": instantaneous_phase,
            "instantaneous_frequency": instantaneous_frequency,
        }

    @staticmethod
    def fft_transform(signal):
        """Fast Fourier Transform (FFT)."""
        signal = np.asarray(signal)
        n = len(signal)
        freq = np.fft.fftfreq(n)
        spectrum = np.fft.fft(signal)

        return {"type": "FFT", "frequency": freq, "spectrum": spectrum}

    @staticmethod
    def stft_transform(signal, fs=1.0, nperseg=256):
        """Short-Time Fourier Transform (STFT)."""
        signal = np.asarray(signal)
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
        return {"type": "STFT", "f": f, "t": t, "Zxx": Zxx}

    @staticmethod
    def wigner_transform(signal):
        """Wigner-Ville Transform (WVT) implemented manually."""
        signal = np.asarray(signal)
        n = len(signal)
        tfr = np.zeros((n, n), dtype=complex)

        for t in range(n):
            for tau in range(-min(t, n - t - 1), min(t, n - t - 1)):
                tfr[t, tau] = signal[t + tau] * np.conj(signal[t - tau])

        # FFT langs forsinkelsesaksen (tau)
        tfr = np.fft.fftshift(np.fft.fft(tfr, axis=1), axes=1)
        freqs = np.fft.fftshift(np.fft.fftfreq(n))

        return {"type": "WVT", "tfr": tfr, "t": np.arange(n), "f": freqs}

    # ---------------- Generalized plotting ----------------
    @staticmethod
    def plot_results(signal, results):
        """General plotting depending on transform type."""
        plt.figure(figsize=(16, 6))

        if results["type"] == "HT":
            plt.subplot(3, 1, 1)
            plt.plot(signal, label="Original Signal")
            plt.plot(results["amplitude_envelope"], label="Hilbert Envelope", alpha=0.7)
            plt.legend()
            plt.title("Hilbert Transform - Signal & Envelope")

            plt.subplot(3, 1, 2)
            plt.plot(results["instantaneous_phase"])
            plt.title("Instantaneous Phase")

            plt.subplot(3, 1, 3)
            plt.plot(results["instantaneous_frequency"])
            plt.title("Instantaneous Frequency")

        elif results["type"] == "FFT":
            plt.plot(results["frequency"], np.abs(results["spectrum"]))
            plt.title("FFT Spectrum")
            plt.xlabel("Frequency (cycles/sample)")
            plt.ylabel("Amplitude")

        elif results["type"] == "STFT":
            plt.pcolormesh(results["t"], results["f"], np.abs(results["Zxx"]), shading="gouraud")
            plt.title("STFT Spectrogram")
            plt.xlabel("Time")
            plt.ylabel("Frequency")

        elif results["type"] == "WVT":
            plt.pcolormesh(results["t"], results["f"], np.abs(results["tfr"].T), shading="auto")
            plt.title("Wigner-Ville Transform")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar(label="Magnitude")

        else:
            raise ValueError(f"Unknown transform type: {results['type']}")

        plt.tight_layout()
        plt.show()
