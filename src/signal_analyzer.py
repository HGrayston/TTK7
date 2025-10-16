import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, stft
from ssqueezepy import cwt
from tftb.processing import WignerVilleDistribution
import math
from PyEMD import EMD
import pywt
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

        # FFT (kun positive frekvenser)
        spectrum = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(n, d=(1.0/fs) if fs else 1.0)

        return {
            "type": "FFT",
            "frequency": freq,
            "spectrum": spectrum,
            "fs": fs
        }

    @staticmethod
    def stft_transform(signal, fs=None, nperseg=256):
        signal = np.asarray(signal)
        fs_eff = 1.0 if fs is None else fs
        f, t, Zxx = stft(signal, fs=fs_eff, nperseg=nperseg)
        return {"type": "STFT", "f": f, "t": t, "Zxx": Zxx, "fs": fs_eff}

    @staticmethod
    def wvt_transform(signal, fs=100, timestamps=None):
        signal = np.asarray(signal)
        wvd = WignerVilleDistribution(signal, timestamps=timestamps)
        tfr, t, f = wvd.run()
        tfr = np.abs(tfr) / np.max(np.abs(tfr))

        if fs is not None:
            t = t / fs     # sekunder i stedet for samples
            f = f * fs     # Hz i stedet for cycles/sample

        return {"type": "WVT", "tfr": tfr, "t": t, "f": f, "fs": fs}

    @staticmethod
    def wt_transform(signal, wavelet="cmor1.5-1.0", fs=None, n_freqs=128):
            signal = np.asarray(signal)
            if fs is None:
                raise ValueError("fs må settes for å bruke pywt.cwt")

            # Sett frekvensaksen (0–Nyquist)
            freqs = np.linspace(1, fs/2, n_freqs)
            fc = pywt.central_frequency(wavelet)
            scales = fc * fs / freqs

            # Kjør CWT
            Wx, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

            return {
                "type": "WT",
                "coefficients": Wx,
                "freqs": freqs,
                "fs": fs
            }

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
    #hht
    @staticmethod
    def plot_hht_per_imf(imfs, fs=1.0, signal=None, n_freq_bins=200):
        """
        Plot original signal (full bredde) + IMF(t) + Hilbert–Huang spectrum (Plotly)
        """
        n_imfs = len(imfs)
        t = np.arange(imfs.shape[1]) / fs

        # --- Hilbert transform for amplitude/frekvens ---
        analytic = hilbert(imfs, axis=1)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase, axis=1) * fs / (2*np.pi)
        inst_freq = np.abs(inst_freq)
        t_if = t[1:]

        # --- Beregn HHT-spekter for hver IMF ---
        max_freq = np.percentile(inst_freq, 99)
        freq_bins = np.linspace(0, max_freq, n_freq_bins)

        hht_maps = []
        for i in range(n_imfs):
            hht = np.zeros((n_freq_bins, len(t_if)))
            for j in range(len(t_if)):
                f = inst_freq[i, j]
                a = amp[i, j+1]
                if 0 <= f < max_freq:
                    idx = np.searchsorted(freq_bins, f)
                    hht[idx, j] += a
            hht = gaussian_filter(hht, sigma=1.0)
            hht_maps.append(hht)

        # --- Lag Plotly-layout: toppsignal full bredde + 2 kolonner for IMFs ---
        total_rows = n_imfs + (1 if signal is not None else 0)
        fig = make_subplots(
            rows=total_rows,
            cols=2,
            shared_xaxes=False,
            horizontal_spacing=0.08,
            vertical_spacing=0.04,
            column_widths=[0.45, 0.55],
            specs=[
                [{"colspan": 2}, None] if (signal is not None and r == 0)
                else [{}, {}]
                for r in range(total_rows)
            ],
            subplot_titles=[
                "Original signal" if signal is not None and i == 0 else
                (f"IMF {i if signal is not None else i+1}" if j == 0 else
                f"Hilbert–Huang Spectrum IMF {i if signal is not None else i+1}")
                for i in range(total_rows)
                for j in range(2)
                if not (signal is not None and i == 0 and j == 1)
            ]
        )

        row_offset = 1 if signal is not None else 0

        # --- (1) Original signal over hele toppen ---
        if signal is not None:
            fig.add_trace(
                go.Scatter(x=t, y=signal, mode='lines',
                        line=dict(color='black', width=1.5),
                        name='Original signal'),
                row=1, col=1
            )
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)

        # --- (2) IMF + HHT per IMF ---
        for i in range(n_imfs):
            r = i + row_offset + 1

            # Venstre: IMF
            fig.add_trace(
                go.Scatter(x=t, y=imfs[i], mode='lines',
                        line=dict(color='royalblue'),
                        name=f'IMF {i+1}'),
                row=r, col=1
            )

            # Høyre: Hilbert–Huang spektrum
            fig.add_trace(
                go.Heatmap(
                    z=hht_maps[i],
                    x=t_if,
                    y=freq_bins,
                    colorscale='turbo',
                    showscale=(i == 0)  # vis kun én fargeskala
                ),
                row=r, col=2
            )

            fig.update_yaxes(title_text="Amplitude", row=r, col=1)
            fig.update_yaxes(title_text="Freq [Hz]", row=r, col=2)

        # --- Layout ---
        fig.update_xaxes(title_text="Time [s]", row=total_rows, col=1)
        fig.update_xaxes(title_text="Time [s]", row=total_rows, col=2)
        fig.update_layout(
            height=260 * total_rows,
            width=1600,
            showlegend=False,
            title="Hilbert–Huang Transform per IMF",
            template="plotly_white",
            margin=dict(t=80, l=50, r=50, b=50),
        )

        fig.show()



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
            ax.pcolormesh(results["t"], results["f"], np.abs(results["tfr"]), shading="auto")
            ax.set_title("Wigner-Ville Transform")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")

        elif results["type"] == "WT":
            Wx = np.abs(results["coefficients"])
            freqs = results["freqs"]
            fs = results["fs"]

            T = np.arange(Wx.shape[1]) / fs
            pc = ax.pcolormesh(T, freqs, Wx, shading="auto", cmap="jet")
            ax.set_title("Wavelet Transform (CWT)")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")

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