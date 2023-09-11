#!/usr/bin/env python3

import numpy as np
import pandas as pd
from src.synthetic import gillespie_noise

from src.crosscorr import crosscorr


def generate_filepath_gillespie_noise(
    num_timeseries,
    noise_timescale,
    noise_amp,
    dir="../data/interim/gillespienoise/",
):
    """filename generator"""
    deathrate = 1 / noise_timescale
    birthrate = noise_amp / noise_timescale
    num_timeseries_str = f"{num_timeseries:.0f}"
    deathrate_str = f"{deathrate:.3f}".replace(".", "p")
    birthrate_str = f"{birthrate:.3f}".replace(".", "p")
    gill_noise_filepath = (
        dir
        + "gillespienoise_n"
        + num_timeseries_str
        + "_k"
        + birthrate_str
        + "_d"
        + deathrate_str
        + ".csv"
    )
    return gill_noise_filepath


def load_gillespie_noise(gill_noise_filepath, num_timeseries):
    # bodge. ideally, it should detect the number of time series from the filename
    gill_noise_array = np.genfromtxt(gill_noise_filepath, delimiter=",")
    gill_noise_array = gill_noise_array[:num_timeseries, :]
    return gill_noise_array


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
    gill_noise_filepath = generate_filepath_gillespie_noise(
        num_timeseries=num_timeseries,
        noise_timescale=noise_timescale,
        noise_amp=noise_amp,
    )
    # Load from file if it exists, or generate new
    try:
        gill_noise_array = load_gillespie_noise(
            gill_noise_filepath, num_timeseries=num_timeseries
        )
    except:
        print(f"{gill_noise_filepath} does not exist, running simulations...")
        gill_noise_array = gillespie_noise(
            num_timeseries=num_timeseries,
            num_timepoints=len(timeaxis),
            noise_timescale=noise_timescale,
            noise_amp=noise_amp,
            time_final=gill_time_final,
            grid_num_intervals=gill_num_intervals,
        )
        np.savetxt(gill_noise_filepath, gill_noise_array, delimiter=",")

    # Add signal and noise
    combined_array = signal_array + gill_noise_array
    # Construct dataframes for correlation processes
    combined_df1 = pd.DataFrame(combined_array)

    # Autocorrelation
    autocorr_result = crosscorr.as_function(
        combined_df1, stationary=False, normalised=True, only_pos=True
    )

    return autocorr_result
