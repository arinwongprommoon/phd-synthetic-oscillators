#!/usr/bin/env python3
import numpy as np


def acorr(y):
    norm = y - np.mean(y)
    var = np.var(y)
    acorr = np.correlate(norm, norm, "full")[len(norm) - 1 :]
    acorr = acorr / var / len(norm)
    return acorr


def xcorr(x, y):
    norm_x = x - np.mean(x)
    std_x = np.std(x)
    norm_y = y - np.mean(y)
    std_y = np.std(y)
    # assumes x and y are the same length
    xcorr = np.correlate(norm_x, norm_y, "full")
    xcorr = xcorr / (std_x * std_y) / len(norm_x)
    return xcorr
