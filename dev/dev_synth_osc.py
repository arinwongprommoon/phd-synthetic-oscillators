#!/usr/bin/env python3
# %% Import requirements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from postprocessor.core.processes.fft import fft
from postprocessor.core.processes.findpeaks import findpeaks
from postprocessor.routines.mean_plot import mean_plot

# from postprocessor.core.multisignal.crosscorr import crosscorr
from crosscorr import crosscorr
from synthetic import (
    fitzhugh_nagumo,
    fitzhugh_nagumo_stochastic,
    gillespie_noise,
    gillespie_noise_raw,
    harmonic_stochastic,
    sinusoid,
)
from utils import multiarray_random_shift, simple_median_plot, tile_signals


# %% Optional: define logistic envelope function
def logistic_envelope(timeaxis, k_min, k_max, tau):
    """Logistic function, to function as envelope function for oscillations"""
    return k_min + k_max * (1 - np.exp(-timeaxis / tau))


# %% [markdown]
# GENERATE SIGNALS

# %% [markdown]
# Step 1: Define parameters

# %% [markdown]
# Step 2: Generate arrays of signals

# %% parameters
num_timeseries = 100
timeaxis = np.linspace(0, 500, 500)

# %% [markdown]
# Choice group A: fill with same signal
# (thus using numpy.tile instead of for loop to make it fast)

# %% [markdown]
# Choice 1: Array of FHNs

# %%
fitzhugh_nagumo_single, _ = fitzhugh_nagumo(
    timeaxis=timeaxis, ext_stimulus=0.4, tau=12.5, a=0.7, b=0.82
)
fitzhugh_nagumo_single -= np.mean(fitzhugh_nagumo_single)
fitzhugh_nagumo_array = tile_signals([fitzhugh_nagumo_single], [num_timeseries])

# %% [markdown]
# Choice 2: Array of sinusoids

# %%
sinusoid_single = sinusoid(timeaxis=timeaxis, amp=1, freq=0.0235, phase=0)
sinusoid_array = tile_signals([sinusoid_single], [num_timeseries])

# %% [markdown]
# Choice 3: Mixed array of sinusoids

# %%
sinusoid_long = sinusoid(timeaxis=timeaxis, amp=1, freq=0.03, phase=0)
sinusoid_short = sinusoid(timeaxis=timeaxis, amp=1, freq=0.04, phase=0)
sinusoid_mixed_array = tile_signals([sinusoid_short, sinusoid_long], [20, 20])

# %% [markdown]
# Shift phases -- grouping pairs/triplets/tuples of signals that come from
# the same sources

# %%
fitzhugh_nagumo_array, sinusoid_array = multiarray_random_shift(
    [fitzhugh_nagumo_array, sinusoid_array]
)

# %%
sinusoid_array = multiarray_random_shift([sinusoid_array])[0]

# %%
sinusoid_mixed_array = multiarray_random_shift([sinusoid_mixed_array])[0]

# %% [markdown]
# Choice group B: each row is different

# %% [markdown]
# Choice 4: Array of sinusoids, random phases

# %%
sinusoid_outofphase_array = np.empty((num_timeseries, len(timeaxis)), dtype=float)
for row_index in range(num_timeseries):
    phase = np.random.random() * 2 * np.pi
    sinusoid_outofphase_array[row_index] = sinusoid(
        timeaxis=timeaxis, amp=1, freq=0.03, phase=phase
    )

# %% [markdown]
# Choice 5: Mixed array of sinusoids, random phases

# %%
def generate_sinusoid_outofphase_array(num_timeseries, timeaxis, freq):
    sinusoid_outofphase_array = np.empty((num_timeseries, len(timeaxis)), dtype=float)
    for row_index in range(num_timeseries):
        phase = np.random.random() * 2 * np.pi
        sinusoid_outofphase_array[row_index] = sinusoid(
            timeaxis=timeaxis, amp=1, freq=freq, phase=phase
        )
    return sinusoid_outofphase_array


sinusoid_outofphase_long = generate_sinusoid_outofphase_array(200, timeaxis, 0.03)
sinusoid_outofphase_short = generate_sinusoid_outofphase_array(200, timeaxis, 0.04)
sinusoid_mixed_array = np.concatenate(
    (sinusoid_outofphase_short, sinusoid_outofphase_long)
)

# %% [markdown]
# Choice 6: Array of sinusoids with envelope function, random phases
# (was functionalised, could make it function again if needed)

# %%
k_min = 1
k_max = 10
tau = 100
envelope = logistic_envelope(timeaxis=timeaxis, k_min=k_min, k_max=k_max, tau=tau)
nonstat_sinusoid_array = np.empty((num_timeseries, len(timeaxis)), dtype=float)
for row_index in range(num_timeseries):
    phase = np.random.random() * 2 * np.pi
    nonstat_sinusoid_array[row_index] = envelope * sinusoid(
        timeaxis=timeaxis, amp=1, freq=0.03, phase=phase
    )

# %% [markdown]
# Step 3: Generate arrays of noise

# %% [markdown]
# Choice 1: white/Gaussian noise

# %%
white_noise_std = 3
white_noise_array1 = np.random.normal(
    loc=0, scale=white_noise_std, size=(num_timeseries, len(timeaxis))
)
white_noise_array2 = np.random.normal(
    loc=0, scale=white_noise_std, size=(num_timeseries, len(timeaxis))
)

# %% [markdown]
# Choice 2: Gillespie noise

# %%
noise_timescale = 20
noise_amp = 500

gill_time_final = 7500
gill_num_intervals = 5000
gill_noise_array = gillespie_noise(
    num_timeseries=num_timeseries,
    num_timepoints=len(timeaxis),
    noise_timescale=noise_timescale,
    noise_amp=noise_amp,
    time_final=gill_time_final,
    grid_num_intervals=gill_num_intervals,
)

# %% [markdown]
# Step 4: Assign signal and noisy arrays, then construct final dataframes

# %% [markdown]
# Step 4.1: Assign signal arrays

# %%
signal_array1 = sinusoid_outofphase_array
# signal_array2 = fitzhugh_nagumo_array

# %% [markdown]
# Step 4.2: Assign noise arrays

# %%
noise_array1 = gill_noise_array
# noise_array2 = white_noise_array2

# %%
# Filename generator for next two cells
# noise_timescale = 200
# noise_amp = 1000

deathrate = 1 / noise_timescale
birthrate = noise_amp / noise_timescale
deathrate_str = str(deathrate).replace(".", "p")
birthrate_str = str(birthrate).replace(".", "p")
filename = "gillespienoise_k" + birthrate_str + "_d" + deathrate_str + ".csv"
print(filename)

# %%
# Alternative: LOAD noise array(s)
gill_noise_array = np.genfromtxt(filename, delimiter=",")

# %%
# Optional: SAVE noise array(s)
# Useful for Gillespie noise because it takes time to generate,
# especially for long final times.
deathrate = 1 / noise_timescale
birthrate = noise_amp / noise_timescale
deathrate_str = str(deathrate).replace(".", "p")
birthrate_str = str(birthrate).replace(".", "p")
filename = "gillespienoise_k" + birthrate_str + "_d" + deathrate_str + ".csv"

np.savetxt(filename, gill_noise_array, delimiter=",")

# %% [markdown]
# Step 4.3: Add signal and noise

# %%
signal_array1 = signal_array1 + noise_array1
# signal_array2 = signal_array2 + noise_array2

# %% [markdown]
# Alternative: harmonic oscillation with stochastic parameters

# %%
# Generate Gillespie noise (raw)
noise_timescale = 20
noise_amp = 500

gill_time_final = 7500
gill_num_intervals = 5000
gill_noise_time, gill_noise_list = gillespie_noise_raw(
    num_timeseries=num_timeseries,
    noise_timescale=noise_timescale,
    noise_amp=noise_amp,
    time_final=gill_time_final,
)

# Model parameter
ang_freq = 0.1
# Noise parameter
std = 0.03

# Scale Gillespie time axis to fit time axis
for gill_time_element in gill_noise_time:
    gill_time_element -= gill_time_element[0]
    gill_time_element *= timeaxis[-1] / gill_time_element[-1]

# Scale noise array to create angular frequency array
# ang_freq_2darray = (gill_noise_array * std) + ang_freq
ang_freq_2darray = [
    (gill_noise_element * std) + ang_freq for gill_noise_element in gill_noise_array
]

# Generate sinusoids via harmonic DEs, with stochastic angular frequency
# defined by gill_noise_array
# TODO: Make this faster, this is ridiculously slow
gill_array = np.empty((num_timeseries, len(timeaxis)), dtype=float)
for row_index in range(num_timeseries):
    # Random phase shift
    # Determine initial conditions...
    # (add code here)

    # Generate sinusoid
    gill_array[row_index], _ = harmonic_stochastic(
        timeaxis=gill_noise_time[row_index],
        ang_freq_array=ang_freq_2darray[row_index],
    )
    print(f"Generating time series {row_index+1} of {num_timeseries}")

signal_array1 = gill_array

# %% [markdown]
# Alternative: FitzHugh-Nagumo model with stochastic parameters

# %%
# Model parameters
ext_stimulus = 0.4
tau = 12.5
a = 0.7
b = 0.8
# Noise parameter
std = 0.03

# Scale noise array to create ext_stimulus array
# TODO: Use a different gill_noise_array per each parameter -- i.e.
# generate a 4x size gill_noise_array and slice that into four
ext_stimulus_2darray = (gill_noise_array * std) + ext_stimulus
tau_2darray = (gill_noise_array * std) + tau
a_2darray = (gill_noise_array * std) + a
b_2darray = (gill_noise_array * std) + b
# Generate time series via FHN DEs, with stochastic parameters
# defined by gill_noise_array
# TODO: Make this faster, this is ridiculously slow
gill_array = np.empty((num_timeseries, len(timeaxis)), dtype=float)
for row_index in range(num_timeseries):
    # Random phase shift
    # Determine initial conditions...
    # (add code here)

    # Generate sinusoid
    gill_array[row_index], _ = fitzhugh_nagumo_stochastic(
        timeaxis=timeaxis,
        ext_stimulus_array=ext_stimulus_2darray[row_index],
        tau_array=tau_2darray[row_index],
        a_array=a_2darray[row_index],
        b_array=b_2darray[row_index],
    )
    print(f"Generating time series {row_index+1} of {num_timeseries}")

# %% [markdown]
# Step 4.4: Construct dataframes for correlation processes

# %%
signal_df1 = pd.DataFrame(signal_array1)
# signal_df2 = pd.DataFrame(signal_array2)

# %% [markdown]
# Step 5: Autocorrelation & cross-correlation

# %% [markdown]
# Autocorrelation

# %%
autocorr_result = crosscorr.as_function(
    signal_df1, stationary=False, normalised=True, only_pos=True
)

# %% [markdown]
# Cross-correlation

# %%
crosscorr_result = crosscorr.as_function(signal_df1, signal_df2)

# %% [markdown]
# Mean across replicates

# %%
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

# %%
sns.heatmap(signal_df2)

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
auc = np.trapz(std_array)
print(auc)

# %% [markdown]
# robustness: fft computed and power at oscillation frequency examined

# %%
fft_freqs, fft_power = fft.as_function(autocorr_result, sampling_period=1)

fig_fft, ax_fft = plt.subplots()
mean_plot(
    fft_power,
    unit_scaling=fft_freqs.iloc[0, 1],
    label="",
    xlabel="Frequency ($\mathrm{min}^{-1}$)",
    ylabel="Power",
    plot_title=f"Mean Fourier spectrum across all time series\n(n = {len(fft_power)})",
    ax=ax_fft,
)
ax_fft.axvline(
    freq,
    color="r",
    linestyle="--",
)

index_freq = np.argwhere(fft_freqs.iloc[0].to_numpy() == 0.03)[0][0]
powers_at_freq = fft_power.iloc[:, index_freq]
print(f"mean power at freq = {np.mean(powers_at_freq)}")
print(f"std dev of power at freq = {np.std(powers_at_freq)}")

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
    -mean_acf[troughs_mask],
    p0=initial_guess,
)
lower_func = -model_func(timeaxis, lower_A, lower_K, lower_C)

# %%
plt.plot(timeaxis, mean_acf)
plt.scatter(timeaxis[peaks_mask], mean_acf[peaks_mask])
plt.scatter(timeaxis[troughs_mask], mean_acf[troughs_mask])
plt.plot(timeaxis, upper_func)
plt.plot(timeaxis, lower_func)

print(f"upper envelope: {upper_A:.4f} * exp(- {upper_K:.4f}) + {upper_C:.4f}")
print(f"lower envelope: {lower_A:.4f} * exp(- {lower_K:.4f}) + {lower_C:.4f}")

# %% [markdown]
# cross-correlation

# %%
simple_median_plot(
    crosscorr_result, ylabel="Cross correlation", xlabel="Lag (time points)"
)
plt.axhline(0, color="k")
plt.axvline(0, color="k")
