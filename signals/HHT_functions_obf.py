#!/usr/bin/python
# Copyright (c) 2021, Olav B. Fosso, NTNU
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#
def ampl_filter_splines(x, treshold):
    """Code for filtering the signal based on amplitude
    Upper and lower splines are used in the filter

    Args:
        x: The numpy array of values
        treshold: Criteria for filtering

    Returns:
        Vector of filtered values

    """

    import numpy as np

    from signals.HHT_functions_obf import get_envelops_obf

    # Find modified upper and lower splines
    upper, lower = get_envelops_obf(x)
    upper_spline = upper - (upper + lower) / 2.0
    lower_spline = lower + (upper + lower) / 2.0

    k = len(x)
    i = 0
    while i < k - 1:
        a1 = upper_spline[i]
        a2 = lower_spline[i]
        if (np.abs(a1) <= treshold) and (np.abs(a1 - a2) <= 2 * treshold):
            x[i] = 0
        i = i + 1
    y = x
    return y


def ampl_filter(x, treshold):
    """Code for filtering the signal based on amplitude
    All values below the limit is removed

    Args:
        x: The numpy array of values
        treshold: Criteria for filtering

    Returns:
        Vector of filtered values

    """

    import numpy as np

    k = len(x)
    i = 0
    x[0] = 0
    while i < k - 2:
        a1 = x[i]
        a2 = x[i + 1]
        if (np.abs(a2 - a1)) <= treshold:
            x[i + 1] = 0
        i = i + 1
    y = x
    return y


def butter_obf(x, order, wn):
    """Code for low pass filtering of the input signal

    Args:
        x: The numpy array of values
        order: Order of the filter
        wn: Digital: half-periods/sample

    Returns:
        Vector of filtered values

    """
    import matplotlib.pyplot as plt
    from scipy import signal

    b, a = signal.butter(order, wn, "low")
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, x, zi=zi * x[0])
    # Apply filter twice
    z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])
    # Filter the filter
    y = signal.filtfilt(b, a, x)
    plt.plot(x, "r")
    plt.plot(y, "g")
    plt.grid()
    plt.show()
    return y


#
#
def inst_freq_nder(x, t, nfilt=1):
    """Code for calculating the instantaneous freqeunies

    Args:
        x: The numpy array of values
        t: Vector of time instants
        nfilt: Averaging over 'nfilt' values

    Returns:
        Vector of instanteous frequencies

    """
    import numpy as np
    from scipy.signal import hilbert

    from signals.HHT_functions_obf import array_ma

    def ifnew(hx1, hx2c, dt):
        hx12 = hx1 * hx2c
        hxi = np.imag(hx12)
        hxr = np.real(hx12)
        if1 = np.arctan(-hxi / hxr) / (2 * np.pi * dt)
        return if1

    xlen = len(x)

    x1 = array_ma(x, nfilt)

    instfreq = np.zeros(xlen)
    hx = hilbert(x1)
    hxc = np.conj(hx)

    dt = t[2] - t[1]
    i = 0
    ifagg = 0
    while i < xlen - 1:
        if2 = ifnew(hx[i], hxc[i + 1], dt)
        ifagg = ifagg + if2
        ifav = ifagg / (i + 1)
        instfreq[i] = if2
        #        print(ifav, if2)
        i = i + 1
    instfreq[xlen - 1] = instfreq[xlen - 2]
    return instfreq


#
def IMF_csv_writer(data, filename):
    """Code for writing a file in CSV-format

    Args:
        data: The numpy array to be saved
        filename: The csv-file to be used

    Returns:
        Nothing

    """
    import csv

    with open(filename, "w", newline="") as csvfile:
        imfswriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        imfswriter.writerow(data)
    return


#
def HHT_extreme_values(t, modes, title_text="Extreme values for function"):
    "Calculates extreme values of a a non-linear function and make plots"
    import matplotlib.pyplot as plt

    x = modes
    # Initialize
    #
    peak = [0]
    lower = [0]
    # Search extreme values
    i = 1
    value1 = x[i - 1]
    value2 = x[i]
    # Search first extrema
    if value2 > value1:
        peakv = "True"
    else:
        peakv = "False"

    while i < len(x):
        value1 = x[i - 1]
        value2 = x[i]

        # Upper envelope
        if peakv == "True" and value2 >= value1:
            peak.append(0)
            lower.append(0)
            i += 1
        elif peakv == "True" and value2 < value1:
            peak.append(value1)
            lower.append(0)
            peakv = "False"
            i += 1
        # Lower envelope
        elif peakv == "False" and value2 <= value1:
            lower.append(0)
            peak.append(0)
            i += 1
        elif peakv == "False" and value2 > value1:
            lower.append(value1)
            peak.append(0)
            peakv = "True"
            i += 1
    # Prepare plots
    plt.plot([1, 2, 2])
    plt.subplot(2, 1, 1)
    plt.ylabel("Signal", fontsize=16)
    plt.title(title_text, fontsize=20)
    plt.grid()
    plt.plot(x)
    plt.plot(peak, "g")
    plt.plot(lower, "r")
    plt.subplot(2, 1, 2)
    plt.ylabel("Extreme values", fontsize=16)
    plt.grid()
    plt.plot(peak, "g")
    plt.plot(lower, "r")
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    return


def HHT_upper_envelope(t, modes, title_text="Extreme values for function"):
    "Calculates extreme values of a a non-linear function and make plots"
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import splev, splrep

    x = modes
    # Initialize
    #
    peak = [0]
    lower = [0]
    ext_peak = [0]
    ext_lower = [0]
    # Search extreme values
    i = 1
    value1 = x[i - 1]
    value2 = x[i]
    # Search first extrema
    if value2 > value1:
        peakv = "True"
    else:
        peakv = "False"

    while i < len(x):
        value1 = x[i - 1]
        value2 = x[i]

        # Upper envelope
        if peakv == "True" and value2 >= value1:
            peak.append(0)
            lower.append(0)
            i += 1
        elif peakv == "True" and value2 < value1:
            peak.append(value1)
            ext_peak.append(i)
            lower.append(0)
            peakv = "False"
            i += 1
        # Lower envelope
        elif peakv == "False" and value2 <= value1:
            lower.append(0)
            peak.append(0)
            i += 1
        elif peakv == "False" and value2 > value1:
            lower.append(value1)
            ext_lower.append(i)
            peak.append(0)
            peakv = "True"
            i += 1
    ext_peak.append(len(x) - 1)
    ext_lower.append(len(x) - 1)
    # print('x ', len(x), x[-1])
    temp1 = np.copy(x)
    # print('x ', len(x), x[-1])
    # Modify the start and end point extremes to be equal to the first and last extreme point
    idf = ext_peak[1]
    idl = ext_peak[-2]
    temp1[0] = temp1[idf]
    temp1[-1] = temp1[idl]
    # Find upper sline
    tck = splrep(t[ext_peak], temp1[ext_peak], k=3)
    spl_upper = splev(t, tck)

    idf = ext_lower[1]
    idl = ext_lower[-2]
    temp1[0] = temp1[idf]
    temp1[-1] = temp1[idl]
    # Find lower spline
    tck = splrep(t[ext_lower], temp1[ext_lower], k=3)
    spl_lower = splev(t, tck)

    # print('x ', len(x), x[-1], x[-1])
    x[-1] = 0
    x[0] = 0
    # print('x ', len(x), x[-1])

    xs = np.linspace(0, 3, 3000)

    # Prepare plots
    plt.plot([1, 2, 2])
    plt.subplot(2, 1, 1)
    plt.ylabel("Signal", fontsize=16)
    plt.title("Values before spline functions", fontsize=20)
    plt.plot(x)
    plt.plot(peak)
    plt.plot(lower)
    plt.subplot(2, 1, 2)
    plt.plot(peak, "r")

    plt.plot(spl_upper, "r", lw=1)
    plt.ylabel("Extreme values & Splines", fontsize=16)
    plt.plot(lower, "g")

    plt.plot(spl_lower, "g", lw=1)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    x2 = (spl_upper + spl_lower) / 2.0
    return spl_upper - x2


#
def IMF_plots(x, imfs, title_text="IMF-plots", transp=False):
    """Code for plotting the original function and the IMFs

    Args:
        x: The original time series
        imfs: The array of calculated IMFs
        title_text: The text used on the plots

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt

    # Plot function
    if transp:
        p = imfs.shape[1]
    else:
        p = imfs.shape[0]
    fig = plt.figure(figsize=(10, 6))
    plt.plot([1, 2, p + 1])
    plt.grid()

    plt.subplot(p + 1, 1, 1)
    plt.ylabel("Signal", fontsize=16)
    #    title_text = 'Empirical Mode Decomposition - New plotfunction'
    plt.title(title_text, fontsize=20)
    # Plot Signal
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(x, "b", lw=2)
    i = 0
    # Plot IMFs and Residue
    while i < p:
        plt.subplot(p + 1, 1, i + 2)
        if i == p - 1:
            plt.ylabel("Res.", fontsize=16)
        else:
            plt.ylabel("IMFs", fontsize=16)
        plt.grid()
        plt.tick_params(width=2, labelsize=16)
        plt.plot(imfs[i], "b", lw=2)
        if transp:
            plt.plot(imfs[:, i], "b", lw=2)
        else:
            plt.plot(imfs[i], "b", lw=2)
        i = i + 1
    #   mng = plt.get_current_fig_manager()
    #   mng.window.state('zoomed')
    plt.show()
    return
    # Average frequency and amplitude of first IMF
    # imfnew = imfs[0]
    # plt.plot(imfnew)
    # plt.show()


def IMF_plots_with_time(t, x, imfs, title_text="IMF-plots", transp=False):
    """Code for plotting the original function and the IMFs

    Args:
        t: Vecor of time instants
        x: The original time series
        imfs: The array of calculated IMFs
        transp: Time series given in columns (True)
        title_text: The text used on the plots

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt

    # Plot function
    if transp:
        p = imfs.shape[1]
    else:
        p = imfs.shape[0]
    fig = plt.figure(figsize=(10, 6))
    plt.plot([1, 2, p + 1])
    plt.grid()

    plt.subplot(p + 1, 1, 1)
    plt.ylabel("Signal", fontsize=16)
    #    title_text = 'Empirical Mode Decomposition - New plotfunction'
    plt.title(title_text, fontsize=20)
    # Plot Signal
    plt.grid()
    plt.tick_params(width=2, labelsize=16, labelbottom="off")
    plt.plot(t, x, "b", lw=2)
    i = 0
    # Plot IMFs and Residue
    while i < p:
        plt.subplot(p + 1, 1, i + 2)
        if i == p - 1:
            plt.ylabel("Res.", fontsize=16)
            plt.tick_params(width=2, labelsize=16)
        else:
            plt.ylabel("IMFs", fontsize=16)
            plt.tick_params(width=2, labelsize=16, labelbottom="off")
        plt.grid()
        if transp:
            plt.plot(t, imfs[:, i], "b", lw=2)
        else:
            plt.plot(t, imfs[i], "b", lw=2)
        i = i + 1
    #    mng = plt.get_current_fig_manager()
    #    mng.window.state('zoomed')
    plt.show()
    return


def IMF_userdef_plots_with_time(
    nplot, t, x, imfs, title_text="IMF-plots", transp=False
):
    """Code for plotting the original function and the IMFs

    Args:
        nplot: User-defined number og IMFs to be plotted
        t: Vecor of time instants
        x: The original time series
        imfs: The array of calculated IMFs
        title_text: The text used on the plots

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Plot function
    if transp:
        p1 = imfs.shape[1]
    else:
        p1 = imfs.shape[0]
    fig = plt.figure(figsize=(10, 6))
    p = min(p1, nplot)
    modres = np.zeros(len(x))
    iplot = nplot
    while iplot < p1:  # Add the skipped IMFs to the residual
        if transp:
            modres = modres + imfs[:, iplot]
        else:
            modres = modres + imfs[iplot]
        iplot = iplot + 1

    plt.plot([1, 2, p + 1])
    plt.grid()

    plt.subplot(p + 1, 1, 1)
    plt.ylabel("Signal", fontsize=16)
    #    title_text = 'Empirical Mode Decomposition - New plotfunction'
    plt.title(title_text, fontsize=20)
    # Plot Signal
    plt.grid()
    plt.tick_params(width=2, labelsize=16, labelbottom="off")
    plt.plot(t, x, "b", lw=2)
    i = 0
    # Plot IMFs and Residue
    while i < p:
        plt.subplot(p + 1, 1, i + 2)
        if i == p - 1:
            plt.ylabel("Res.", fontsize=16)
            plt.tick_params(width=2, labelsize=16)
            plt.grid()
            if transp:
                plt.plot(t, imfs[:, i] + modres, "b", lw=2)
            else:
                plt.plot(t, imfs[i] + modres, "b", lw=2)

        else:
            plt.ylabel("IMFs", fontsize=16)
            plt.tick_params(width=2, labelsize=16, labelbottom="off")
            plt.grid()
            if transp:
                plt.plot(t, imfs[:, i], "b", lw=2)
            else:
                plt.plot(t, imfs[i], "b", lw=2)

        i = i + 1
    #    mng = plt.get_current_fig_manager()
    #    mng.window.state('zoomed')
    plt.show()
    return


def IMF_plots_with_time_reduced(t, x, imfs, title_text="IMF-plots"):
    """Code for plotting the original function and the IMFs reduced period

    Args:
        t: Vector of time instants
        x: The original time series
        imfs: The array of calculated IMFs
        title_text: The text used on the plots

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt

    # Plot function
    p = imfs.shape[0]
    plt.plot([1, 2, p + 1])
    plt.grid()

    plt.subplot(p + 1, 1, 1)
    plt.ylabel("Signal", fontsize=16)
    #    title_text = 'Empirical Mode Decomposition - New plotfunction'
    plt.title(title_text, fontsize=20)
    # Plot Signal
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t, x, "b", lw=2)
    i = 0
    # Plot IMFs and Residue
    while i < p:
        plt.subplot(p + 1, 1, i + 2)
        if i == p - 1:
            plt.ylabel("Res.", fontsize=16)
        else:
            plt.ylabel("IMFs", fontsize=16)
        plt.grid()
        plt.tick_params(width=2, labelsize=16)
        plt.plot(t, imfs[i], "b", lw=2)
        i = i + 1
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    return


# Average frequency and amplitude of first IMF
# imfnew = imfs[0]
# plt.plot(imfnew)
# plt.show()


def Sub_plots(
    t,
    nplot,
    plot1,
    plot2=[],
    plot3=[],
    plot4=[],
    plot5=[],
    plot6=[],
    title_text="Function plots",
):
    """Code for plotting a number of sub-components

    Args:
        t: Vector of time instants
        nplot: Number of plots to be plotted (min =1, max = 6)
        plots (1-6): Numpy arrays
        title_text: The text used on the plots

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt

    # Plot function
    plt.plot([1, 2, nplot])
    plt.grid()

    plt.subplot(nplot, 1, 1)
    plt.ylabel("Signal 1", fontsize=16)
    #    title_text = 'Plot'
    plt.title(title_text, fontsize=20)
    # Plot Signal
    plt.grid()
    #    plt.tick_params(width=2, labelsize=16)
    plt.tick_params(width=2, labelsize=16, labelbottom="on")
    plt.plot(t, plot1, "b", lw=2)
    if nplot > 1:
        plt.subplot(nplot, 1, 2)
        plt.ylabel("Signal 2", fontsize=16)
        plt.grid()
        #        plt.tick_params(width=2, labelsize=16)
        plt.tick_params(width=2, labelsize=16, labelbottom="on")
        plt.plot(t, plot2, "b", lw=2)
    if nplot > 2:
        plt.subplot(nplot, 1, 3)
        plt.ylabel("Signal 3", fontsize=16)
        plt.grid()
        #        plt.tick_params(width=2, labelsize=16)
        plt.tick_params(width=2, labelsize=16, labelbottom="off")
        plt.plot(t, plot3, "b", lw=2)
    if nplot > 3:
        plt.subplot(nplot, 1, 4)
        plt.ylabel("Signal 4", fontsize=16)
        plt.grid()
        plt.tick_params(width=2, labelsize=16)
        #        plt.tick_params(width=2, labelsize=16, labelbottom='off')
        plt.plot(t, plot4, "b", lw=2)
    if nplot > 4:
        plt.subplot(nplot, 1, 5)
        plt.ylabel("Signal 5", fontsize=16)
        plt.grid()
        plt.tick_params(width=2, labelsize=16)
        plt.plot(t, plot5, "b", lw=2)
    if nplot > 5:
        plt.subplot(nplot, 1, 6)
        plt.ylabel("Signal 6", fontsize=16)
        plt.grid()
        plt.tick_params(width=2, labelsize=16)
        plt.plot(t, plot6, "b", lw=2)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    return


def HHT_hilbert_one_signal(t, x1, samprate, nsamp):
    """Code for calculating the Hilbert transform for one IMF function

    Args:
        t: Vector of time instants
        x1: Time series 1
        samprate: sampling rate pr second
        nsamp: Number of samples

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import hilbert

    # signal1 = np.r_(x1)
    # signal2 = np.r_(x2)
    signal1 = x1

    # plt.plot(signal)
    # plt.show()
    hs1 = hilbert(signal1)

    ampl_env1 = np.abs(hs1)

    plt.plot(np.real(hs1), np.imag(hs1), "r")
    #    mng = plt.get_current_fig_manager()
    #    mng.window.state('zoomed')
    plt.show()
    omega_s1 = np.unwrap(np.angle(hs1))
    f_inst_s1 = np.diff(omega_s1) / (2 * np.pi / samprate)

    plt.tick_params(width=2, labelsize=16)
    plt.title("Instantaneous frequency IMF1(blue)", fontsize=20)
    plt.grid()

    plt.plot(t[0 : nsamp - 1], f_inst_s1[0 : nsamp - 1], "b", lw=2)

    #    mng = plt.get_current_fig_manager()
    #    mng.window.state('zoomed')
    plt.show()
    plt.title("Amplitudes of IMF1(blue)", fontsize=20)
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t[0 : nsamp - 1], ampl_env1[0 : nsamp - 1], "b", lw=2)

    #    mng = plt.get_current_fig_manager()
    #    mng.window.state('zoomed')
    plt.show()
    #
    plt.title("Amplitudes  (Red) Instantaneous Freq (Blue) ", fontsize=20)
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t[0 : nsamp - 1], ampl_env1[0 : nsamp - 1], "r", lw=2)
    plt.plot(t[0 : nsamp - 1], f_inst_s1[0 : nsamp - 1], "b", lw=2)
    #    mng = plt.get_current_fig_manager()
    #    mng.window.state('zoomed')
    plt.show()
    return


def HHT_hilbert_two_sign(t, x1, x2, samprate, nsamp):
    """Code for calculating the Hilbert transform for two IMF functions

    Args:
        t: Vector of time instants
        x1: Time series 1
        x2: Time series 2
        samprate: sampling rate pr second
        nsamp: Number of samples

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import hilbert

    # signal1 = np.r_(x1)
    # signal2 = np.r_(x2)
    signal1 = x1
    signal2 = x2

    # plt.plot(signal)
    # plt.show()
    hs1 = hilbert(signal1)
    hs2 = hilbert(signal2)
    ampl_env1 = np.abs(hs1)
    ampl_env2 = np.abs(hs2)

    # plt.plot(np.real(hs1), np.imag(hs1), 'b')
    # plt.plot(np.real(hs2), np.imag(hs2), 'g')
    # plt.plot(np.real(hs3), np.imag(hs3), 'r')
    # plt.grid()
    plt.plot(np.real(hs1), np.imag(hs1), "r")
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    omega_s1 = np.unwrap(np.angle(hs1))
    omega_s2 = np.unwrap(np.angle(hs2))
    f_inst_s1 = np.diff(omega_s1) / (2 * np.pi / samprate)
    f_inst_s2 = np.diff(omega_s2) / (2 * np.pi / samprate)
    plt.tick_params(width=2, labelsize=16)
    plt.title("Instantaneous frequency IMF1(blue). IMF2(green)", fontsize=20)
    plt.grid()

    plt.plot(t[0 : nsamp - 1], f_inst_s1[0 : nsamp - 1], "b", lw=2)
    # temp1 = moving_average2(f_inst_s1[10:nsamp-10], 200)
    # plt.plot(t[10:nsamp - 10], temp1, "m")
    plt.plot(t[0 : nsamp - 1], f_inst_s2[0 : nsamp - 1], "g", lw=2)
    # temp1 = moving_average2(f_inst_s2[10:nsamp - 10], 200)

    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    plt.title("Amplitudes of IMF1(blue). IMF2(green)", fontsize=20)
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t[0 : nsamp - 1], ampl_env1[0 : nsamp - 1], "b", lw=2)
    plt.plot(t[0 : nsamp - 1], ampl_env2[0 : nsamp - 1], "g", lw=2)
    #    plt.plot(ampl_env1, "y")
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    #
    plt.title("Amplitudes  (Red) Instantaneous Freq (Blue) ", fontsize=20)
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t[0 : nsamp - 1], ampl_env1[0 : nsamp - 1], "r", lw=2)
    plt.plot(t[0 : nsamp - 1], f_inst_s1[0 : nsamp - 1], "b", lw=2)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    return


def HHT_hilbert(t, imfs, x1, samprate, nsamp):
    """Code for calculating the Hilbert transform for:
        3 first IMFs and one extra signal
        Calculates the amplitude and instantaneous freq and make the plots

    Args:
        t: Vector of time instants
        imfs: Calculated IMFs
        x1: TThe additonal time series
        samprate: sampling rate pr second
        nsamp: Number of samples

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import hilbert

    signal1 = np.r_[imfs[0]]
    signal2 = np.r_[imfs[1]]
    signal3 = np.r_[imfs[2]]
    # plt.plot(signal)
    # plt.show()
    hs1 = hilbert(signal1)
    hx1 = hilbert(x1)
    hs2 = hilbert(signal2)
    hs3 = hilbert(signal3)
    ampl_env1 = np.abs(hs1)
    ampl_env2 = np.abs(hs2)
    ampl_env3 = np.abs(hs3)

    # plt.plot(np.real(hs1), np.imag(hs1), 'b')
    # plt.plot(np.real(hs2), np.imag(hs2), 'g')
    # plt.plot(np.real(hs3), np.imag(hs3), 'r')
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(np.real(hx1[200:400]), np.imag(hx1[200:400]), "r")
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    omega_s1 = np.unwrap(np.angle(hs1))
    omega_x1 = np.unwrap(np.angle(hx1))
    omega_s2 = np.unwrap(np.angle(hs2))
    omega_s3 = np.unwrap(np.angle(hs3))
    f_inst_s1 = np.diff(omega_s1) / (2 * np.pi / samprate)
    f_inst_x1 = np.diff(omega_x1) / (2 * np.pi / samprate)
    f_inst_s2 = np.diff(omega_s2) / (2 * np.pi / samprate)
    f_inst_s3 = np.diff(omega_s3) / (2 * np.pi / samprate)
    plt.title(
        "Instantaneous frequency IMF1(blue). IMF2(green) and IMF3(red)", fontsize=20
    )
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t[100 : nsamp - 100], f_inst_s1[100 : nsamp - 100], "b", lw=2)
    # temp1 = moving_average2(f_inst_s1[10:nsamp-10], 200)
    # plt.plot(t[10:nsamp - 10], temp1, "m")
    plt.plot(t[100 : nsamp - 100], f_inst_s2[100 : nsamp - 100], "g", lw=2)
    # temp1 = moving_average2(f_inst_s2[10:nsamp - 10], 200)
    # plt.plot(t[10:nsamp - 10], temp1, "m")
    plt.plot(t[100 : nsamp - 100], f_inst_s3[100 : nsamp - 100], "r", lw=2)
    # temp1 = moving_average2(f_inst_s3[10:nsamp - 10], 200)
    # plt.plot(t[10:nsamp - 10], temp1, "m")
    plt.plot(t[100 : nsamp - 100], f_inst_x1[100 : nsamp - 100], "y", lw=2)
    # temp1 = moving_average2(f_inst_x1[10:nsamp - 10], 200)
    # plt.plot(t[10:nsamp - 10], temp1, "m")
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    plt.title("Amplitudes of IMF1(blue). IMF2(green) and IMF3(red)", fontsize=20)
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t[100 : nsamp - 100], ampl_env1[100 : nsamp - 100], "b", lw=2)
    plt.plot(t[100 : nsamp - 100], ampl_env2[100 : nsamp - 100], "g", lw=2)
    plt.plot(t[100 : nsamp - 100], ampl_env3[100 : nsamp - 100], "r", lw=2)
    #    plt.plot(ampl_env1, "y")
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()
    # X = np.fft.fftshift(np.fft.fft(signal))
    # X = np.fft.fft(signal1)
    # plt.plot(np.abs(X) ** 2)
    # plt.plot(X)
    # plt.plot(np.linspace(-1, 1.0, 1000), np.abs(X) ** 2)
    # plt.show()

    # f, t1, Sxx = spectrogram(signal1, fs=1.0, mode='magnitude',scaling='spectrum')
    # plt.pcolormesh(t1, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    return


def moving_average(a, n=5):
    """Code for calculating the moving average of numbers in a list

    Args:
        a: List to be processed
        n: Number of elements to be used in averaging (Default =5)

    Returns:
        a1: Vector of averages values

    """
    import numpy as np

    i1 = len(a)
    a1 = []
    p1 = 0
    p2 = p1 + n
    while (p2 <= i1) and (p1 < p2):
        s1 = np.sum(a[p1:p2])
        if p1 < p2:
            a1.append(s1 / (p2 - p1))
        else:
            a1.append(s1)
        p1 = p1 + 1
        p2 = min(p1 + n, i1)
    return a1


def moving_average2(a, n=5):
    """Code for calculating the moving average for numbers in a list

    Args:
        a: List to be processed
        n: Number of elements to be used in averaging (Default =5)

    Returns:
        a1: Vector of averages values

    """
    import numpy as np

    i1 = len(a)
    a1 = []
    pc = 0
    p1 = 0
    p2 = p1 + n
    while pc < i1:
        s1 = np.sum(a[p1:p2])
        if p1 < p2:
            a1.append(s1 / (p2 - p1))
        else:
            a1.append(s1)
        if p1 + n >= i1:
            p2 = min(p1 + n, i1)
            p1 = p2 - n
        else:
            p1 = p1 + 1
            p2 = min(p1 + n, i1)
        pc = pc + 1
    return a1


def Find_zero_crossings(time_serie):
    """Code for calculating the number of zero-crossings of a time series

    Args:
        time_serie: Vector to be processed

    Returns:
        num_zero: Number of identified zero-crossings

    """
    import numpy as np

    num_zero = (np.diff(np.sign(time_serie)) != 0).sum()
    #    print('Num zero', num_zero)
    return num_zero


def get_envelops_obf(x, t=None):
    """Find the upper and lower envelopes of the array `x`.
    :Example:
    >>> import numpy as np
    >>> x = np.random.rand(100,)
    >>> upper, lower = get_envelops(x)
    """
    import numpy as np
    from scipy import interpolate
    from scipy.signal import argrelmax, argrelmin

    if t is None:
        t = np.arange(x.shape[0])
    maxima = argrelmax(x)[0]
    minima = argrelmin(x)[0]

    # consider the start and end to be extrema

    ext_maxima = np.zeros((maxima.shape[0] + 2,), dtype=int)
    ext_maxima[1:-1] = maxima
    ext_maxima[0] = 0
    ext_maxima[-1] = t.shape[0] - 1

    ext_minima = np.zeros((minima.shape[0] + 2,), dtype=int)
    ext_minima[1:-1] = minima
    ext_minima[0] = 0
    ext_minima[-1] = t.shape[0] - 1

    temp1 = np.copy(x)
    idf = ext_maxima[1]
    idl = ext_maxima[-2]
    temp1[0] = x[idf]
    temp1[-1] = x[idl]

    tck = interpolate.splrep(t[ext_maxima], temp1[ext_maxima], k=3)
    upper = interpolate.splev(t, tck)

    idf = ext_minima[1]
    idl = ext_minima[-2]
    temp1[0] = x[idf]
    temp1[-1] = x[idl]
    tck = interpolate.splrep(t[ext_minima], temp1[ext_minima], k=3)
    lower = interpolate.splev(t, tck)
    return upper, lower


def heterodyne(x1, t, hfreq):
    """Calculates the heterodyned signal and make a Standard EMD
    (Needs further refinement)
    Args:
        x1: Signal to be heterodyned
        t: Vector of time instants
        hfreq: Frequency to be used in the heterodyning
                Larger but close to the highest freq-component

    Returns:
        x2: The heterodyned signal

    """
    #    import pyhht
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import hilbert
    #    from JD_utils import inst_freq, extr, get_envelops

    signal1 = np.r_[x1]
    hx1 = hilbert(signal1)
    hxr = np.real(hx1)
    hxi = np.imag(hx1)
    # Sample    x = modes * mask + hmodes * hmask
    mask = 1.0 * np.cos(2 * np.pi * hfreq * t)
    hmask = 1.0 * np.sin(2 * np.pi * hfreq * t)

    x2 = hxr * mask + hxi * hmask
    #   temp1 = array_ma(x2, 4)
    plt.figure(figsize=(9, 4))
    plt.title("Heterodyned IMF-signal", fontsize=20)
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t, x2, "r")
    #    plt.plot(temp1, 'g')
    plt.show()
    # print(extr(x2))
    # print (extr(xx))
    # print(x2[0:100])
    # print(xx[0:100])
    # decomposer = pyhht.EMD(x2, alpha=0.05, fixe=0, maxiter=5000, nbsym=2)
    #    decomposer = pyhht.EMD(temp1, alpha=0.1, fixe=0, n_imfs=0, threshold_1=0.1, threshold_2=0.2, maxiter=3000, nbsym=2)
    #   imfs = decomposer.decompose()
    #   print(decomposer.io())

    # Plot IMF function
    #   title_EMD_pos = 'EMD - Heterodyned signal'
    #   IMF_plots_with_time(t,x2, imfs, title_text=title_EMD_pos)
    return x2


def array_ma(a, n=5):
    """Code for calculating the moving average of numbers in a numpy array

    Args:
        a: Array to be processed
        n: Number of elements to be used in averaging (Default =5)

    Returns:
        a1: Vector of averages values

    """
    import numpy as np

    i1 = len(a)
    a1 = np.zeros(i1)
    pc = 0
    p1 = 0
    p2 = p1 + n
    while pc < i1:
        s1 = np.sum(a[p1:p2])
        if p1 < p2:
            a1[p1] = s1 / (p2 - p1)
        else:
            a1[p1] = s1
        if p1 + n >= i1:
            p2 = min(p1 + n, i1)
            p1 = p2 - n
        else:
            p1 = p1 + 1
            p2 = min(p1 + n, i1)
        pc = pc + 1
    return a1


def InstFreq_Ampl(t, x1, samprate):
    """Calculates Instantaneous Frequencies and Amplitude
        based on Hilbert transform for one time series (Derivative approach)
        Make the plots

    Args:
        x1: Time series to be processed
        samprate: Sampling (number of samples pr second)

    Returns:
        Nothing

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import hilbert

    signal1 = np.r_[x1]

    hs1 = hilbert(signal1)
    ampl_env1 = np.abs(hs1)
    omega_s1 = np.unwrap(np.angle(hs1))
    f_inst_s1 = np.diff(omega_s1) / (2 * np.pi / samprate)
    tm = len(f_inst_s1) - 1

    # Plot function
    plt.plot([1, 2, 2])
    plt.grid()

    plt.subplot(2, 1, 1)
    plt.ylabel("Amplitudes", fontsize=16)
    title_text = "Instantaneous Frequencies and Amplitudes"
    plt.title(title_text, fontsize=20)
    # Plot Signal
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t[0:tm], ampl_env1[0:tm], "b", lw=2)
    plt.subplot(2, 1, 2)
    plt.ylabel("Inst Freq", fontsize=16)
    plt.grid()
    plt.tick_params(width=2, labelsize=16)
    plt.plot(t[0:tm], f_inst_s1[0:tm], "b", lw=2)
    # temp1 = moving_average2(f_inst_s2[10:nsamp - 10], 200)
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.show()

    return


# END
