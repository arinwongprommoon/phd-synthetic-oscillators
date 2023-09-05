#!/usr/bin/env python3
# %% Import requirements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from postprocessor.core.processes.findpeaks import findpeaks

# from postprocessor.core.multisignal.crosscorr import crosscorr
from src.crosscorr import crosscorr
from src.synthetic import gillespie_noise, sinusoid
from src.utils import simple_median_plot


# %% [markdown]
# GENERATE SIGNALS & COMPUTATIONS

# %%
# Parameters
num_timeseries = 100
timeaxis = np.linspace(0, 500, 500)
noise_timescale = 20
noise_amp = 100

# Array of sinusoids, random phases
sinusoid_outofphase_array = np.empty((num_timeseries, len(timeaxis)), dtype=float)
for row_index in range(num_timeseries):
    phase = np.random.random() * 2 * np.pi
    sinusoid_outofphase_array[row_index] = sinusoid(
        timeaxis=timeaxis, amp=1, freq=0.03, phase=phase
    )

# Array of Gillespie noise
gill_time_final = 7500
gill_num_intervals = 5000
# gill_noise_array = gillespie_noise(
#    num_timeseries=num_timeseries,
#    num_timepoints=len(timeaxis),
#    noise_timescale=noise_timescale,
#    noise_amp=noise_amp,
#    time_final=gill_time_final,
#    grid_num_intervals=gill_num_intervals,
# )

# Filename generator
deathrate = 1 / noise_timescale
birthrate = noise_amp / noise_timescale
deathrate_str = str(deathrate).replace(".", "p")
birthrate_str = str(birthrate).replace(".", "p")
gill_noise_filename = "gillespienoise_k" + birthrate_str + "_d" + deathrate_str + ".csv"
# LOAD noise array
gill_noise_array = np.genfromtxt(gill_noise_filename, delimiter=",")
# gill_noise_array = gill_noise_array[:num_timeseries,:]


# Assign signal arrays
signal_array1 = sinusoid_outofphase_array
# Assign noise arrays
noise_array1 = gill_noise_array
# Add signal and noise
signal_array1 = signal_array1 + noise_array1

# Construct dataframes for correlation processes
signal_df1 = pd.DataFrame(signal_array1)

# Autocorrelation
autocorr_result = crosscorr.as_function(
    signal_df1, stationary=False, normalised=True, only_pos=True
)
# Mean across replicates
mean_across_replicates = np.nanmean(signal_array1, axis=0).reshape(
    (1, signal_array1.shape[1])
)
mean_across_replicates = mean_across_replicates.T

# %% [markdown]
# PLOTTING

# %% [markdown]
# input data

# %%
sns.heatmap(signal_df1)

# %% [markdown]
# gillespie noise

# %%
gill_array = signal_array1
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    mean_across_replicates,
    linewidth=3,
    label=f"mean across {gill_array.shape[0]} replicates",
)
ax.plot(gill_array[0], linewidth=1, label="replicate 1")
ax.plot(gill_array[1], linewidth=1, label="replicate 2")
ax.set_xlabel("Time point")
ax.set_ylabel("Signal")
fig.legend()

# %% [markdown]
# acf of signals with gillespie noise, with options

# %%
fit_exp_decay = True
scale_lag_by_pd = True
freq = 0.03

fig, ax = plt.subplots()

# scale lag axis by sinusoid period
autocorr_result_scaled = autocorr_result.copy()
if scale_lag_by_pd:
    freq = freq
    xlabel = "Lag (periods)"
    plt.vlines(x=[1, 2, 3, 4], ymin=-1, ymax=1, ls="--")
else:
    freq = 1
    xlabel = "Lag (time points)"
autocorr_result_scaled.columns *= freq

# fit exp decay
if fit_exp_decay:
    decayrate = (gill_time_final / (gill_num_intervals - 1)) * (1 / noise_timescale)
    t = autocorr_result.columns.to_numpy()
    decay_function = np.exp(-(decayrate) * t)
    ax.plot(t * freq, decay_function, color="r")

# draw acf
simple_median_plot(
    autocorr_result_scaled,
    xlabel=xlabel,
    ylabel="Autocorrelation function",
    ax=ax,
)
# and axes
plt.axhline(0, color="k")
plt.axvline(0, color="k")

# %% [markdown]
# at longer lags, fewer data points are used to compute acf, and thus the std dev across replicate acfs at those points are greater

# %%
lag = np.linspace(0, 499, 500)
num_datapoints = signal_array1.shape[1] * np.linspace(500, 1, 500)
acf_variation = np.std(autocorr_result)

fig, ax = plt.subplots()
ax.plot(num_datapoints, acf_variation)
ax.set_xlabel("Number of data points used at lag value")
ax.set_ylabel("Standard deviation of\nautocorrelation function values at lag value")

# %% [markdown]
# variation between acfs, expressed as area the curve of std dev change over lag time

# %%
std_array = autocorr_result.std(axis=0)
plt.plot(std_array)
plt.xlabel("Lag (time point)")
plt.ylabel("Standard deviation")
auc = np.trapz(std_array)
print(auc)

# %% [markdown]
# envelope function

# %%
# find peaks & troughs
mean_acf_df = autocorr_result.mean().to_frame().T
peaks_df = findpeaks.as_function(mean_acf_df)
troughs_df = findpeaks.as_function(-mean_acf_df)

# datatype conversions
timeaxis = mean_acf_df.columns.to_numpy()
mean_acf = mean_acf_df.to_numpy()[0]
peaks_mask = peaks_df.to_numpy()[0] != 0
troughs_mask = troughs_df.to_numpy()[0] != 0

# add (0,1) to datapoints
peaks_mask[0] = True
troughs_mask[0] = True


# %% [markdown]
# option: non-linear fit

# %%
# https://stackoverflow.com/questions/3938042/fitting-exponential-decay-with-no-initial-guessing
def model_func(t, A, K, C):
    return A * np.exp(-K * t) + C


def fit_exp_nonlinear(t, y, p0):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, p0, maxfev=1000)
    A, K, C = opt_parms
    return A, K, C


# initial guess is the decay function in acf plot
central_decay_rate = (gill_time_final / (gill_num_intervals - 1)) * (
    1 / noise_timescale
)
initial_A = 1
initial_K = central_decay_rate
initial_C = 0
initial_guess = [initial_A, initial_K, initial_C]
guess_func = model_func(timeaxis, initial_A, initial_K, initial_C)

# fit peaks
upper_A, upper_K, upper_C = fit_exp_nonlinear(
    timeaxis[peaks_mask],
    mean_acf[peaks_mask],
    p0=initial_guess,
)
upper_func = model_func(timeaxis, upper_A, upper_K, upper_C)

# fit troughs
lower_A, lower_K, lower_C = fit_exp_nonlinear(
    timeaxis[troughs_mask],
    mean_acf[troughs_mask],
    p0=initial_guess,
)
lower_func = model_func(timeaxis, lower_A, lower_K, lower_C)


# %% [markdown]
# option: linear fit, using C estimated from non-linear fit

# %%
# https://stackoverflow.com/questions/3938042/fitting-exponential-decay-with-no-initial-guessing
def model_func(t, A, K, C):
    return A * np.exp(-K * t) + C


def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K


# initial guess is the decay function in acf plot
central_decay_rate = (gill_time_final / (gill_num_intervals - 1)) * (
    1 / noise_timescale
)
initial_A = 1
initial_K = central_decay_rate
initial_C = 0
initial_guess = [initial_A, initial_K, initial_C]
guess_func = model_func(timeaxis, initial_A, initial_K, initial_C)

# fit peaks
upper_A, upper_K = fit_exp_linear(
    timeaxis[peaks_mask],
    mean_acf[peaks_mask],
    C=upper_C,
)
upper_func = model_func(timeaxis, upper_A, upper_K, upper_C)

# fit troughs
lower_A, lower_K = fit_exp_linear(
    timeaxis[troughs_mask],
    mean_acf[troughs_mask],
    C=lower_C,
)
lower_func = model_func(timeaxis, lower_A, lower_K, lower_C)


# %% [markdown]
# option: fewer params: $y(t) = (1 - C)e^{-kt} + C$

# %%
def model_func(t, K, C):
    return (1 - C) * np.exp(-K * t) + C


def fit_exp_nonlinear(t, y, p0):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, p0, maxfev=1000)
    K, C = opt_parms
    return K, C


# initial guess is the decay function in acf plot
central_decay_rate = (gill_time_final / (gill_num_intervals - 1)) * (
    1 / noise_timescale
)
initial_K = central_decay_rate
initial_C = 0
initial_guess = [initial_K, initial_C]
guess_func = model_func(timeaxis, initial_K, initial_C)

# fit peaks
upper_K, upper_C = fit_exp_nonlinear(
    timeaxis[peaks_mask],
    mean_acf[peaks_mask],
    p0=initial_guess,
)
upper_func = model_func(timeaxis, upper_K, upper_C)

# fit troughs
lower_K, lower_C = fit_exp_nonlinear(
    timeaxis[troughs_mask],
    mean_acf[troughs_mask],
    p0=initial_guess,
)
lower_func = model_func(timeaxis, lower_K, lower_C)

initial_A = 1 - initial_C
upper_A = 1 - upper_C
lower_A = 1 - lower_C

# %% [markdown]
# option: fit to mean acf rather than peaks/troughs

# %%
# initial guess is the decay function in acf plot
central_decay_rate = (gill_time_final / (gill_num_intervals - 1)) * (
    1 / noise_timescale
)
initial_K = central_decay_rate
initial_C = 0
initial_guess = [initial_K, initial_C]
guess_func = model_func(timeaxis, initial_K, initial_C)

# fit mean
est_K, est_C = fit_exp_nonlinear(
    timeaxis,
    mean_acf,
    p0=initial_guess,
)
est_func = model_func(timeaxis, est_K, est_C)

initial_A = 1 - initial_C
est_A = 1 - est_C

# %% [markdown]
# plotting

# %%
plt.plot(timeaxis, mean_acf)
plt.scatter(timeaxis[peaks_mask], mean_acf[peaks_mask])
plt.scatter(timeaxis[troughs_mask], mean_acf[troughs_mask])
plt.plot(timeaxis, guess_func, label="theoretical decay function")
plt.plot(timeaxis, est_func, label="fit to mean ACF")
plt.plot(timeaxis, upper_func, label="fit to peaks")
plt.plot(timeaxis, lower_func, label="fit to troughs")
plt.legend()
plt.xlabel("Lag (time points)")
plt.ylabel("Autocorrelation function")

print(
    f"theoretical decay function: {initial_A:.4f} * exp(- {initial_K:.4f}) + {initial_C:.4f}"
)
print(f"upper envelope: {upper_A:.4f} * exp(- {upper_K:.4f}) + {upper_C:.4f}")
print(f"lower envelope: {lower_A:.4f} * exp(- {lower_K:.4f}) + {lower_C:.4f}")

residuals = mean_acf - est_func
ss_res = np.sum(residuals**2)
print(f"residual sum of squares: {ss_res:.4f}")

# %%
from scipy.signal import periodogram


def fft(timeseries):
    freqs, power = periodogram(
        timeseries,
        fs=1,
        nfft=len(timeseries),
        return_onesided=True,
        scaling="spectrum",
    )
    return freqs, power


osc = mean_acf - est_func
freqs, power = fft(osc)
amp = np.sqrt(2 * max(power))

fig_0, ax_0 = plt.subplots()
ax_0.plot(osc)
ax_0.hlines(y=[-amp, amp], xmin=0, xmax=len(osc), color="r", linestyle="--")
ax_0.set_title("Oscillations in acf")
ax_0.set_xlabel("Lag")
ax_0.set_ylabel("Deviation of mean acf from fitted exponential")

print(f"Amplitude of acf oscillations is {amp:.4f}")

fig_1, ax_1 = plt.subplots()
ax_1.plot(freqs, power)
ax_1.set_title("Fourier spectrum")
ax_1.set_xlabel("Frequency")
ax_1.set_ylabel("Power")
