#!/usr/bin/env python3

import numpy as np
from postprocessor.routines.median_plot import median_plot


def tile_signals(list_of_signals, list_of_repeats):
    """Takes a list of 1d arrays, repeats each as needed, and outputs 2d array

    Parameters
    ----------
    list_of_signals : list of 1d numpy.arrays
        list of signals
    list_of_repeats : list of int
        list of integers, indicates how many times each signal is repeated.
        First integer controls how many times the first signal is repeated,
        second integer controls how many times the second signal is repeated,
        etc.

    Examples
    --------
    signal1 = np.array([1, 2, 3])
    signal2 = np.array([4, 5, 6])
    my_array = tile_signals([signal1, signal2], [10, 20])
    """
    return np.concatenate(
        tuple(
            [
                np.tile(signal, (repeat, 1))
                for signal, repeat in zip(list_of_signals, list_of_repeats)
            ]
        )
    )


def multiarray_random_shift(list_of_arrays):
    """Shifts randomly each row in each array in a list of arrays

    Shifts randomly each row in each array in a list of arrays. All arrays in
    the list are subject to the same shifts, defined randomly. All arrays in the
    list must have the same dimensions.

    Parameters
    ----------
    list_of_arrays : list of 2d numpy.array
        list of arrays

    Raises
    ------
    ValueError
        if not all input arrays have the same shape

    Examples
    --------
    array1 = np.array([[1,2], [3,4]])
    array2 = np.array([[5,6], [7,8]])
    array_new1, array_new2 = multiarray_random_shift([array1, array2])
    """
    # Check if all arrays have the same shape, otherwise this function
    # wouldn't make sense
    if len(list_of_arrays) != 1:
        arr_iter = iter(list_of_arrays)
        if not all(arr.shape == next(arr_iter).shape for arr in arr_iter):
            raise ValueError("Not all input arrays have the same shape.")

    num_rows = list_of_arrays[0].shape[0]
    num_columns = list_of_arrays[0].shape[1]
    shift_list = np.random.randint(num_columns, size=num_rows)

    # TODO: Increase performance by sorting by shift interval, see:
    # df_shift from postprocessor.core.multisignal.align
    list_of_arrays_copy = np.copy(list_of_arrays)
    for array in list_of_arrays_copy:
        for row_index, row in enumerate(array):
            array[row_index] = np.roll(row, shift_list[row_index])

    return list_of_arrays_copy


def simple_median_plot(
    trace_df,
    ylabel,
    median_color="b",
    error_color="lightblue",
    xlabel="Time point",
    ax=None,
):
    """Wrapper for median plot to strip away hard-coding/irrelevant stuff"""
    return median_plot(
        trace_df=trace_df,
        trace_name="signal",
        label="signal",
        median_color=median_color,
        error_color=error_color,
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
    )
