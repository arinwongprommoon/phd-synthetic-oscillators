#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy as sp
from postprocessor.core.processes.findpeaks import findpeaks

from src.crosscorr import crosscorr
from src.synthetic import gillespie_noise, sinusoid


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
        "../data/interim/gillespienoise_n"
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


gill_time_final = 7500
gill_num_intervals = 5000
noise_timescale = 20
noise_amp = 250

autocorr_result = acfs_gillespie_noise(
    signal_function=lambda num_timeseries, timeaxis: sinusoid_outofphase_array(
        num_timeseries=1, timeaxis=timeaxis, amp=1, freq=0.03
    ),
    num_timeseries=200,
    noise_timescale=noise_timescale,
    noise_amp=noise_amp,
)

initial_K = (gill_time_final / (gill_num_intervals - 1)) * (1 / noise_timescale)
upper_coeffs, lower_coeffs = fit_peak_trough(autocorr_result, initial_K=initial_K)

breakpoint()
