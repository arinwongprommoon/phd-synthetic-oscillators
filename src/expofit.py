#!/usr/bin/env python3

import numpy as np
import scipy as sp
from postprocessor.core.processes.findpeaks import findpeaks


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
