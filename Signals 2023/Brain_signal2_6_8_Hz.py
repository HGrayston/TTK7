#!/usr/bin/python

import numpy as np
import csv
from HHT_functions_obf import Sub_plots,  get_envelops_obf

timep = 4 #Numer of seconds
nsamp = 400 #Number of samples

# Save data to the location "cpath" and filecase name "case'id'.csv"

mypath = "output/braindata210118/Signal2/"
filecase = "signal2_imf_"

# Time sequence for contineous signals
t = np.linspace(0, timep, nsamp)

samprate = nsamp/timep # Sample rate


# Three modes are assumed - Each mode can have several frequencies
mode1 = np.zeros(nsamp)
mode2 = np.zeros(nsamp)
mode3 = np.zeros(nsamp)
mask = np.zeros(nsamp)

with open('Signal2_2018.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        km = len(row)
        k = 0
        print('Number of points: ', km)
        while k < km:
            mode1[k] = float(row[k])*100000
            k = k + 1

# Final signal
modes = mode1 + mode2 + mode3

# Mask signal applied (0/1)
mask_sig = 1


# Display texts for plots

if mask_sig == 1:

# Mask Text
    title_signal_plots = 'Signal of Brain data'
    title_extr_val = 'Extreme values for for spline calculation'
    title_EMD_pos = 'EMD - Applying positive mask signal'
    title_EMD_neg = 'EMD - Applying negative mask signal'
    title_EMD_Av = 'EMD - Averaged IMFs after applying mask'
    title_EMD_Fin = 'EMD - Final IMFs'
else:
#No Mask text
    title_signal_plots = 'Signal components of Signal1 in Brain data'
    title_extr_val = 'Extreme values for for spline calculation'
    title_EMD_pos = 'Standard EMD -Signal2, Braindata no mask signal'
    title_EMD_neg = 'EMD - Applying negative mask signal'
    title_EMD_Av = 'EMD - Averaged IMFs '
    title_EMD_Fin = 'EMD - Final IMFs'

Sub_plots(t,2,mode1, mask, title_text=title_signal_plots)

# Extract envelopes

upper, lower = get_envelops_obf(modes)
upper_spline = upper - (upper+lower)/2.0

# Time intervals
t1 = np.linspace(0, timep, nsamp)
t2 = np.linspace(0, 2.5, 250)
t3 = np.linspace(0, 1.0, 100)
t4 = np.linspace(0, 0.5, 50)
t5 = np.linspace(0, 1.5, 150)


if mask_sig == 1:

# Define masking functions

#HF Noise filter

    mfilt = 0.7*np.sin(2 * np.pi * 25 * t1)
    mfilt1 = 0.7*np.sin(2 * np.pi * 18 * t1)
#Masking components for the highest frequency intermittent component

    mmode1 = 0.5*np.sin(2 * np.pi * 11 * t5)
#mmode2 = upper_spline[150:250]*np.sin(2 * np.pi * 15 * t3)
    mmode2 = 1.5*np.sin(2 * np.pi * 11 * t3)
    mmode3 = 0.7*np.sin(2 * np.pi * 11 * t5)
    mmode = np.concatenate((mmode1, mmode2*2.5, mmode3))


#Masking components for the second highest frequency intermittent component

    mmode5 = 0.7*np.sin(2 * np.pi * 8 * t4)
#mmode6 = upper_spline[150:250]*np.sin(2 * np.pi * 8 * t3)
    mmode6 = 2.0*np.sin(2 * np.pi * 8 * t3)
    mmode7 = 0.7*np.sin(2 * np.pi * 8 * t2)
    mmode8 = np.concatenate((mmode5, mmode6*1.5, mmode7))

# Masking components for third highest
    mmode10 = 0.7*np.sin(2 * np.pi * 8 * t1)

#
#Final Masking Signal composition based on componets defined in input

    mask1 = mfilt
    mask2 =  mmode + mfilt
    mask3 = mfilt1
    mask4 = mmode8 + mfilt1

else:
# No mask signal
    mfilt = 0.0*np.sin(2 * np.pi * 25 * t1)
    
#Final Masking Signal composition based on componets defined in input

    mask1 = mfilt
    mask2 = mfilt
    mask3 = mfilt
    mask4 = mfilt



# End
