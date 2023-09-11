#!/usr/bin/env python3

from collections import namedtuple

import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.array import fitzhugh_nagumo_outofphase_array, sinusoid_outofphase_array
from src.expofit import fit_mean, fit_peak_trough
from src.gillespienoise import acfs_gillespie_noise

model_options = {
    # list of noise timescale values to go through
    "noise_timescale_list": [20] * 11,
    # list of noise amplitude values to go through.
    # should be equal length as above
    "noise_amp_list": [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "gill_time_final": 7500,
    "gill_num_intervals": 5000,
    "num_timeseries": 200,
    # function to use for signal,
    # "sinusoid_outofphase_array" or "fitzhugh_nagumo_outofphase_array"
    "signal_function": fitzhugh_nagumo_outofphase_array,
}

plot_choices = {
    # Death rate vs decay rate
    "decay": True,
    # Noise amplitude vs y-displacement
    "ydispl": True,
}

logging.basicConfig(
    level=logging.INFO,
    filename="../logs/gillespie_noise_stat.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# define list of params to go through

NoiseParams = namedtuple("NoiseParams", "noise_timescale noise_amp")

noise_timescale_list = model_options["noise_timescale_list"]
noise_amp_list = model_options["noise_amp_list"]

noise_params_list = [
    NoiseParams(*el) for el in zip(noise_timescale_list, noise_amp_list)
]

deathrate_list = 1 / np.array(noise_timescale_list)

# generate acfs
logger.info("Begin generating ACFs")

acfs_dict = {}
for noise_params in noise_params_list:
    autocorr_result = acfs_gillespie_noise(
        signal_function=model_options["signal_function"],
        num_timeseries=model_options["num_timeseries"],
        noise_timescale=noise_params.noise_timescale,
        noise_amp=noise_params.noise_amp,
        gill_time_final=model_options["gill_time_final"],
        gill_num_intervals=model_options["gill_num_intervals"],
    )
    acfs_dict[noise_params] = autocorr_result

logger.info("Done generating ACFs")

# fit exponentials

logger.info("Begin fitting exponentials")

upper_coeffs_list = []
lower_coeffs_list = []
est_coeffs_list = []

for noise_params in noise_params_list:
    noise_timescale = noise_params.noise_timescale
    autocorr_result = acfs_dict[noise_params]

    initial_K = (
        model_options["gill_time_final"] / (model_options["gill_num_intervals"] - 1)
    ) * (1 / noise_timescale)
    upper_coeffs, lower_coeffs = fit_peak_trough(autocorr_result, initial_K=initial_K)
    est_coeffs = fit_mean(autocorr_result, initial_K=initial_K)
    upper_coeffs_list.append(upper_coeffs)
    lower_coeffs_list.append(lower_coeffs)
    est_coeffs_list.append(est_coeffs)

lower_coeffs_array = np.array(lower_coeffs_list)
upper_coeffs_array = np.array(upper_coeffs_list)
est_coeffs_array = np.array(est_coeffs_list)

logger.info("Done fitting exponentials")

# plots

if plot_choices["decay"]:
    fig_K, ax_K = plt.subplots()
    ax_K.scatter(deathrate_list, lower_coeffs_array[:, 0], label="Fit to troughs")
    ax_K.scatter(deathrate_list, est_coeffs_array[:, 0], label="Fit to mean")
    ax_K.scatter(deathrate_list, upper_coeffs_array[:, 0], label="Fit to peaks")
    ax_K.set_xlabel("Death rate ($d_0$)")
    ax_K.set_ylabel("estimated decay rate ($D$)")
    ax_K.legend()

if plot_choices["ydispl"]:
    fig_C, ax_C = plt.subplots()
    ax_C.scatter(noise_amp_list, lower_coeffs_array[:, 1], label="Fit to troughs")
    ax_C.scatter(noise_amp_list, est_coeffs_array[:, 1], label="Fit to mean")
    ax_C.scatter(noise_amp_list, upper_coeffs_array[:, 1], label="Fit to peaks")
    ax_C.set_xlabel("Noise amplitude ($k_0/d_0$)")
    ax_C.set_ylabel("estimated y-displacement ($C$)")
    ax_C.legend()

# Save figures
pdf_filename = "../reports/gillespie_noise_stat.pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")

logger.info("Done saving plots")

# save stats

birthrate_vs_ydispl_df = pd.DataFrame(
    {
        "noise_amp": noise_amp_list,
        "C_lower": lower_coeffs_array[:, 1],
        "C_central": est_coeffs_array[:, 1],
        "C_upper": upper_coeffs_array[:, 1],
    }
)

birthrate_vs_ydispl_df.to_csv("../data/interim/birthrate_vs_ydispl.csv", index=False)

deathrate_vs_decay_df = pd.DataFrame(
    {
        "deathrate": deathrate_list,
        "D_lower": lower_coeffs_array[:, 0],
        "D_central": est_coeffs_array[:, 0],
        "D_upper": upper_coeffs_array[:, 0],
    }
)

deathrate_vs_decay_df.to_csv("../data/interim/deathrate_vs_decay.csv", index=False)

logger.info("Done saving stats.")
