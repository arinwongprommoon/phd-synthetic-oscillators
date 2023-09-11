#!/usr/bin/env python3
# %%
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# %% [markdown]
# setting stuff up

# %% setting stuff up
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from postprocessor.core.processes.findpeaks import findpeaks

from src.crosscorr import crosscorr
from src.synthetic import fitzhugh_nagumo, gillespie_noise, sinusoid
from src.utils import multiarray_random_shift, tile_signals


def acfs_gillespie_noise(
    signal_function,
    num_timeseries=100,
    timeaxis=np.linspace(0, 500, 500),
    noise_timescale=20,
    noise_amp=100,
    gill_time_final=7500,
    gill_num_intervals=5000,
):
    # TODO: docs

    # Array for signal function
    signal_array = signal_function(num_timeseries=num_timeseries, timeaxis=timeaxis)

    # Array of Gillespie noise
    # filename generator
    deathrate = 1 / noise_timescale
    birthrate = noise_amp / noise_timescale
    num_timeseries_str = f"{num_timeseries:.0f}"
    deathrate_str = f"{deathrate:.3f}".replace(".", "p")
    birthrate_str = f"{birthrate:.3f}".replace(".", "p")
    gill_noise_filename = (
        "../data/interim/gillespienoise/fitzhughnagumo/gillespienoise_n"
        + num_timeseries_str
        + "_k"
        + birthrate_str
        + "_d"
        + deathrate_str
        + ".csv"
    )
    # Load from file if it exists, or generate new
    try:
        gill_noise_array = np.genfromtxt(gill_noise_filename, delimiter=",")
        gill_noise_array = gill_noise_array[:num_timeseries, :]
    except:
        print(f"{gill_noise_filename} does not exist, running simulations...")
        gill_noise_array = gillespie_noise(
            num_timeseries=num_timeseries,
            num_timepoints=len(timeaxis),
            noise_timescale=noise_timescale,
            noise_amp=noise_amp,
            time_final=gill_time_final,
            grid_num_intervals=gill_num_intervals,
        )
        np.savetxt(gill_noise_filename, gill_noise_array, delimiter=",")

    # Add signal and noise
    combined_array = signal_array + gill_noise_array
    # Construct dataframes for correlation processes
    combined_df1 = pd.DataFrame(combined_array)

    # Autocorrelation
    autocorr_result = crosscorr.as_function(
        combined_df1, stationary=False, normalised=True, only_pos=True
    )

    return autocorr_result


def sinusoid_outofphase_array(num_timeseries, timeaxis, amp, freq):
    """Array of sinusoids, random phases"""
    sinusoid_outofphase_array = np.empty((num_timeseries, len(timeaxis)), dtype=float)
    for row_index in range(num_timeseries):
        phase = np.random.random() * 2 * np.pi
        sinusoid_outofphase_array[row_index] = sinusoid(
            timeaxis=timeaxis, amp=amp, freq=freq, phase=phase
        )
    return sinusoid_outofphase_array


def fitzhugh_nagumo_outofphase_array(
    num_timeseries, timeaxis, ext_stimulus=0.4, tau=12.5, a=0.7, b=0.82
):
    fitzhugh_nagumo_single, _ = fitzhugh_nagumo(
        timeaxis=timeaxis, ext_stimulus=ext_stimulus, tau=tau, a=a, b=b
    )
    fitzhugh_nagumo_single -= np.mean(fitzhugh_nagumo_single)
    fitzhugh_nagumo_array = tile_signals([fitzhugh_nagumo_single], [num_timeseries])
    fitzhugh_nagumo_array = multiarray_random_shift([fitzhugh_nagumo_array])[0]
    return fitzhugh_nagumo_array


def model_func(t, K, C):
    return (1 - C) * np.exp(-K * t) + C


def fit_exp_nonlinear(t, y, p0):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, p0, maxfev=1000)
    K, C = opt_parms
    return K, C


def fit_mean(
    array,
    initial_K,
    initial_C=0,
):
    # get mean time series
    mean_df = array.mean().to_frame().T
    timeaxis = mean_df.columns.to_numpy()
    mean_acf = mean_df.to_numpy()[0]

    # initial guess is the decay function in acf plot
    initial_guess = [initial_K, initial_C]

    # fit mean
    est_coeffs = fit_exp_nonlinear(
        timeaxis,
        mean_acf,
        p0=initial_guess,
    )

    return est_coeffs


def fit_peak_trough(
    array,
    initial_K,
    initial_C=0,
):
    """
    array: 2d numpy array
    """
    # find peaks & troughs
    mean_df = array.mean().to_frame().T
    peaks_df = findpeaks.as_function(mean_df)
    troughs_df = findpeaks.as_function(-mean_df)
    # datatype conversions
    timeaxis = mean_df.columns.to_numpy()
    mean_acf = mean_df.to_numpy()[0]
    peaks_mask = peaks_df.to_numpy()[0] != 0
    troughs_mask = troughs_df.to_numpy()[0] != 0
    # add (0,1) to datapoints
    peaks_mask[0] = True
    troughs_mask[0] = True

    # initial guess is the decay function in acf plot
    initial_guess = [initial_K, initial_C]

    # fit peaks
    upper_coeffs = fit_exp_nonlinear(
        timeaxis[peaks_mask],
        mean_acf[peaks_mask],
        p0=initial_guess,
    )
    # fit troughs
    lower_coeffs = fit_exp_nonlinear(
        timeaxis[troughs_mask],
        mean_acf[troughs_mask],
        p0=initial_guess,
    )

    return upper_coeffs, lower_coeffs


# %% [markdown]
# get stats

# %%
gill_time_final = 7500
gill_num_intervals = 5000

# %%
noise_timescale = 20

# sinusoid
autocorr_result = acfs_gillespie_noise(
    signal_function=lambda num_timeseries, timeaxis: sinusoid_outofphase_array(
        num_timeseries=2, timeaxis=timeaxis, amp=1, freq=0.03
    ),
    noise_timescale=noise_timescale,
    num_timeseries=2
)

# FHM
#autocorr_result = acfs_gillespie_noise(signal_function=fitzhugh_nagumo_outofphase_array)

auc = std_auc(autocorr_result)

initial_K = (gill_time_final / (gill_num_intervals - 1)) * (1 / noise_timescale)
upper_coeffs, lower_coeffs = fit_peak_trough(autocorr_result, initial_K=initial_K)
est_coeffs = fit_mean(autocorr_result, initial_K=initial_K)

print(auc)
print(upper_coeffs)
print(lower_coeffs)
print(est_coeffs)
# %% [markdown]
# vary stuff

# %%
# this is VERY ugly, but it's at the end of the day and just want a plot out
#noise_timescale_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
#noise_amp_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
noise_timescale_list = [20] * 11
noise_amp_list = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
auc_list = []
upper_coeffs_list = []
lower_coeffs_list = []
est_coeffs_list = []
ss_res_list = []
amp_list = []

for noise_timescale, noise_amp in zip(noise_timescale_list, noise_amp_list):
    # generate signals & compute acf
    #autocorr_result = acfs_gillespie_noise(
    #    signal_function=lambda num_timeseries, timeaxis: sinusoid_outofphase_array(
    #        num_timeseries=200, timeaxis=timeaxis, amp=1, freq=0.03
    #    ),
    #    num_timeseries=200,
    #    noise_timescale=noise_timescale,
    #    noise_amp=noise_amp,
    #)
    autocorr_result = acfs_gillespie_noise(
        signal_function=lambda num_timeseries, timeaxis: fitzhugh_nagumo_outofphase_array(
        num_timeseries=200, timeaxis=timeaxis
        ),
        num_timeseries=200,
        noise_timescale=noise_timescale,
        noise_amp=noise_amp,
    )

    # fit exponential
    initial_K = (gill_time_final / (gill_num_intervals - 1)) * (1 / noise_timescale)
    upper_coeffs, lower_coeffs = fit_peak_trough(autocorr_result, initial_K=initial_K)
    est_coeffs = fit_mean(autocorr_result, initial_K=initial_K)
    upper_coeffs_list.append(upper_coeffs)
    lower_coeffs_list.append(lower_coeffs)
    est_coeffs_list.append(est_coeffs)


# %% [markdown]
# plots

# %%
fig_K, ax_K = plt.subplots()
deathrate_list = 1 / np.array(noise_timescale_list)
ax_K.scatter(deathrate_list, lower_coeffs_array[:, 0], label="Fit to troughs")
ax_K.scatter(deathrate_list, est_coeffs_array[:, 0], label="Fit to mean")
ax_K.scatter(deathrate_list, upper_coeffs_array[:, 0], label="Fit to peaks")
ax_K.set_xlabel("Death rate ($d_0$)")
ax_K.set_ylabel(
    "estimated decay rate ($D$)"
)
ax_K.legend()

# %%
fig_C, ax_C = plt.subplots()
ax_C.scatter(noise_amp_list, lower_coeffs_array[:, 1], label="Fit to troughs")
ax_C.scatter(noise_amp_list, est_coeffs_array[:, 1], label="Fit to mean")
ax_C.scatter(noise_amp_list, upper_coeffs_array[:, 1], label="Fit to peaks")
ax_C.set_xlabel("Noise amplitude ($k_0/d_0$)")
ax_C.set_ylabel(
    "estimated y-displacement ($C$)"
)
ax_C.legend()

# %%
birthrate_vs_ydispl_df = pd.DataFrame({
    'noise_amp': noise_amp_list,
    'C_lower': lower_coeffs_array[:,1],
    'C_central': est_coeffs_array[:,1],
    'C_upper': upper_coeffs_array[:,1],
})

# %%
birthrate_vs_ydispl_df

# %%
birthrate_vs_ydispl_df.to_csv("../data/interim/birthrate_vs_ydispl.csv", index=False)

# %%
deathrate_vs_decay_df = pd.DataFrame({
    'deathrate': deathrate_list,
    'D_lower': lower_coeffs_array[:,0],
    'D_central': est_coeffs_array[:,0],
    'D_upper': upper_coeffs_array[:,0],
})

# %%
deathrate_vs_decay_df

# %%
deathrate_vs_decay_df.to_csv("../data/interim/deathrate_vs_decay.csv", index=False)
