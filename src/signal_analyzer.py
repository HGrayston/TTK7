import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, stft
from ssqueezepy import cwt
from tftb.processing import WignerVilleDistribution
import math
from PyEMD import EMD


class SignalAnalyzer:
    # ---------------- Transform Methods ----------------
    @staticmethod
    def hilbert_transform(signal, fs=None, n_bins=256):
        """Hilbert transform with fs + n_bins info attached."""
        signal = np.asarray(signal)
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)  # cycles/sample

        if fs is not None:
            instantaneous_frequency = instantaneous_frequency * fs

        return {
            "type": "HT",
            "analytic_signal": analytic_signal,
            "amplitude_envelope": amplitude_envelope,
            "instantaneous_phase": instantaneous_phase,
            "instantaneous_frequency": instantaneous_frequency,
            "fs": fs,
            "n_bins": n_bins,
        }

    @staticmethod
    def fft_transform(signal, fs=None):
        signal = np.asarray(signal)
        n = len(signal)
        spectrum = np.fft.fft(signal)
        freq = np.fft.fftfreq(n, d=(1.0/fs) if fs else 1.0)
        spectrum = np.fft.fftshift(spectrum)
        freq = np.fft.fftshift(freq)
        return {"type": "FFT", "frequency": freq, "spectrum": spectrum, "fs": fs}

    @staticmethod
    def stft_transform(signal, fs=None, nperseg=256):
        signal = np.asarray(signal)
        fs_eff = 1.0 if fs is None else fs
        f, t, Zxx = stft(signal, fs=fs_eff, nperseg=nperseg)
        return {"type": "STFT", "f": f, "t": t, "Zxx": Zxx, "fs": fs_eff}

    @staticmethod
    def wvt_transform(signal, timestamps=None):
        signal = np.asarray(signal)
        wvd = WignerVilleDistribution(signal, timestamps=timestamps)
        tfr, t, f = wvd.run()
        return {"type": "WVT", "tfr": tfr, "t": t, "f": f}

    @staticmethod
    def wt_transform(signal, wavelet="morlet", fs=None):
        signal = np.asarray(signal)
        Wx, scales = cwt(signal, wavelet=wavelet)
        # Convert scales to frequencies
        if fs is not None:
            # For Morlet, frequency = fs * center_frequency / scale
            # ssqueezepy uses center_frequency=1 by default for Morlet
            center_frequency = 1.0
            freqs = fs * center_frequency / scales
        else:
            freqs = 1.0 / scales
        return {"type": "WT", "coefficients": Wx, "scales": scales, "freqs": freqs, "fs": fs}

    # ---------------- Hilbert Spectrum via PyEMD ----------------
    @staticmethod
    def hilbert_spectrum(signal, fs=None, n_bins=256):
        """Hilbert Spectrum via EMD + Hilbert (Hilbert-Huang Transform)."""
        signal = np.asarray(signal)
        emd = EMD()
        imfs = emd.emd(signal)
        if imfs.ndim == 1:
            imfs = imfs[np.newaxis, :]

        T = signal.shape[0]
        time = np.arange(T) if fs is None else np.arange(T)/fs

        all_if = []
        all_amp = []
        for imf in imfs:
            analytic = hilbert(imf)
            amp = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))
            ifreq = np.diff(phase) / (2.0*np.pi)
            if fs is not None:
                ifreq = ifreq * fs
            all_if.append(ifreq)
            all_amp.append(amp[1:])

        ifreqs = np.concatenate(all_if)
        amps = np.concatenate(all_amp)
        T_if = len(all_if[0])

        fmin, fmax = np.nanpercentile(ifreqs, [1, 99])
        f_bins = np.linspace(fmin, fmax, n_bins)
        Z = np.zeros((n_bins, T_if))

        for k, ifk in enumerate(all_if):
            amp = all_amp[k]
            idx = ((ifk - fmin) / (fmax - fmin) * (n_bins - 1)).round().astype(int)
            idx = np.clip(idx, 0, n_bins - 1)
            Z[idx, np.arange(T_if)] += amp

        return {"type": "HS", "time": time[:T_if], "freqs": f_bins, "Z": Z, "fs": fs}

    # ---------------- Plotting ----------------
    @staticmethod
    def plot(signal, results=None, width=10, height=6, ax=None):
        created_new_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height))
            created_new_fig = True

        if results["type"] == "HT":
            ax.plot(signal, label="Original Signal", alpha=0.7)
            ax.plot(results["amplitude_envelope"], label="Hilbert Envelope", alpha=0.7)
            ax.legend()
            ax.set_title("Hilbert Transform (Envelope)")

        elif results["type"] == "HT_IF":
            ax.plot(results["instantaneous_frequency"])
            ax.set_title("Hilbert Instantaneous Frequency")
            ax.set_xlabel("Time [samples]")
            ax.set_ylabel("Frequency [{}]".format("Hz" if results.get("fs") else "cycles/sample"))

        elif results["type"] == "HS":
            ax.pcolormesh(results["time"], results["freqs"], results["Z"], shading="auto", cmap="jet")
            ax.set_title("Hilbert Spectrum (HHT)")
            ax.set_xlabel("Time [{}]".format("s" if results.get("fs") else "samples"))
            ax.set_ylabel("Frequency [{}]".format("Hz" if results.get("fs") else "cycles/sample"))

        elif results["type"] == "FFT":
            ax.plot(results["frequency"], np.abs(results["spectrum"]))
            ax.set_title("FFT Spectrum")
            ax.set_xlabel("Frequency [{}]".format("Hz" if results.get("fs") else "cycles/sample"))
            ax.set_ylabel("Amplitude")

        elif results["type"] == "STFT":
            ax.pcolormesh(results["t"], results["f"], np.abs(results["Zxx"]), shading="gouraud")
            ax.set_title("STFT Spectrogram")
            ax.set_xlabel("Time [{}]".format("s" if results.get("fs") else "samples"))
            ax.set_ylabel("Frequency [{}]".format("Hz" if results.get("fs") else "cycles/sample"))

        elif results["type"] == "WVT":
            ax.pcolormesh(results["t"], results["f"], np.abs(results["tfr"].T), shading="auto")
            ax.set_title("Wigner-Ville Transform")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")

        elif results["type"] == "WT":
            ax.imshow(
                np.abs(results["coefficients"]),
                extent=[0, len(signal), results["freqs"].min(), results["freqs"].max()],
                cmap="jet", aspect="auto", origin="lower"
            )
            ax.set_title("Wavelet Transform (CWT)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency [{}]".format("Hz" if results.get("fs") else "cycles/sample"))

        elif results["type"] == "Original":
            ax.plot(signal)
            ax.set_title("Original Signal")

        else:
            raise ValueError("Unknown transform type for plotting.")

        if created_new_fig:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_all(signal, results_list, width=16, height=12, spectrum=False):
        expanded = []
        for res in results_list:
            expanded.append(res)
            if isinstance(res, dict) and res.get("type") == "HT":
                if spectrum:
                    expanded.append(
                        SignalAnalyzer.hilbert_spectrum(signal, fs=res.get("fs"), n_bins=res.get("n_bins", 256))
                    )
                else:
                    expanded.append(
                        {
                            "type": "HT_IF",
                            "instantaneous_frequency": res["instantaneous_frequency"],
                            "fs": res.get("fs"),
                        }
                    )
        n = len(expanded)
        rows = math.ceil(n/2)
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(width, height))
        axes = np.atleast_1d(axes).ravel()
        for ax, results in zip(axes, expanded):
            SignalAnalyzer.plot(signal, results, ax=ax)
        for j in range(n, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()