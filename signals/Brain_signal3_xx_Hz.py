#!/usr/bin/python

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from HHT_functions_obf import Sub_plots, get_envelops_obf

timep = 4  # Number of seconds
nsamp = 400  # Number of samples

# Save data to the location "cpath" and filecase name "case'id'.csv"

mypath = "output/braindata210118/Signal3/"
filecase = "signal3_imf_"

# Time sequence for contineous signals
t = np.linspace(0, timep, nsamp)

samprate = nsamp / timep  # Sample rate


# Three modes are assumed - Each mode can have several frequencies
mode1 = np.zeros(nsamp)
mode2 = np.zeros(nsamp)
mode3 = np.zeros(nsamp)
mask = np.zeros(nsamp)

with open(Path(__file__).parent / "Signal3_2018.csv", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        km = len(row)
        k = 0
        print("Number of points: ", km)
        while k < km:
            mode1[k] = float(row[k]) * 1000
            k = k + 1

# Final signal
modes = mode1 + mode2 + mode3

# Mask signal applied (0/1)
mask_sig = 1


# Display texts for plots

if mask_sig == 1:
    # Mask Text
    title_signal_plots = "Signal of Brain data"
    title_extr_val = "Extreme values for for spline calculation"
    title_EMD_pos = "EMD - Applying positive mask signal"
    title_EMD_neg = "EMD - Applying negative mask signal"
    title_EMD_Av = "EMD - Averaged IMFs after applying mask"
    title_EMD_Fin = "EMD - Final IMFs"
else:
    # No Mask text
    title_signal_plots = "Signal of Brain data"
    title_extr_val = "Extreme values for for spline calculation"
    title_EMD_pos = "Standard EMD -Signal2, Braindata no mask signal"
    title_EMD_neg = "EMD - Applying negative mask signal"
    title_EMD_Av = "EMD - Averaged IMFs "

Sub_plots(t, 2, mode1, mask, title_text=title_signal_plots)

# Extract envelopes

upper, lower = get_envelops_obf(modes)
upper_spline = upper - (upper + lower) / 2.0

# Time intervals
t1 = np.linspace(0, timep, nsamp)
t2 = np.linspace(0, 2.5, 250)
t3 = np.linspace(0, 1.0, 100)
t4 = np.linspace(0, 0.5, 50)
t5 = np.linspace(0, 1.5, 150)


if mask_sig == 1:
    # Define masking functions

    # HF Noise filter

    mfilt = 0.5 * np.sin(2 * np.pi * 28 * t1)
    mfilt1 = 0.0 * np.sin(2 * np.pi * 30 * t1)
    # Masking components for the highest frequency intermittent component

    mmode1 = 0.5 * np.sin(2 * np.pi * 20 * t5)
    # mmode2 = upper_spline[150:250]*np.sin(2 * np.pi * 15 * t3)
    mmode2 = 1.5 * np.sin(2 * np.pi * 20 * t3)
    mmode3 = 0.7 * np.sin(2 * np.pi * 20 * t5)
    # mmode = np.concatenate((mmode1, mmode2*2.5, mmode3))
    mmode = 1.0 * np.sin(2 * np.pi * 10 * t1)

    # Masking components for the second highest frequency intermittent component

    mmode5 = 0.5 * np.sin(2 * np.pi * 8 * t4)
    # mmode6 = upper_spline[150:250]*np.sin(2 * np.pi * 8 * t3)
    mmode6 = 2.0 * np.sin(2 * np.pi * 8 * t3)
    mmode7 = 0.5 * np.sin(2 * np.pi * 8 * t2)
    mmode8 = np.concatenate((mmode5, mmode6 * 2.5, mmode7))

    # Masking components for third highest
    mmode10 = 0.7 * np.sin(2 * np.pi * 9 * t1)

    # Final Masking Signal composition based on componets defined in input

    mask1 = mfilt
    mask2 = mmode
    mask3 = mmode10
    mask4 = 0.7 * np.sin(2 * np.pi * 9 * t1)

    new_final_signal = modes + mask1 + mask2 + mask3 + mask4

    # Save new_final_signal to CSV
    output_path = Path(__file__).parent
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / f"{filecase}new_final_signal.csv"
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(new_final_signal)

else:
    # No mask signal
    mfilt = 0.0 * np.sin(2 * np.pi * 28 * t1)
    mask1 = mfilt
    mask2 = mfilt
    mask3 = mfilt
    mask4 = mfilt

# End

plt.title("Signal to be decomposed - red / mask - green ", fontsize=20)
plt.plot(t, modes, "r")
plt.plot(t, new_final_signal, "g")
plt.grid()
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.show()
