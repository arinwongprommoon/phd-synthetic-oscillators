#!/usr/bin/env python3

import numpy as np

from src.synthetic import fitzhugh_nagumo, sinusoid
from src.utils import multiarray_random_shift, tile_signals


def sinusoid_outofphase_array(num_timeseries, timeaxis, amp=1, freq=0.03):
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
