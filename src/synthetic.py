#!/usr/bin/env python3
# TODO: Create function/class builders
"""
Functions to create synthetic signals
"""
from abc import ABC, abstractmethod

import numpy as np
import stochpy

from scipy import integrate, interpolate


def white_noise(num_samples, mean=0.0, std_dev=1.0):
    """Generate white noise

    Generate white (Gaussian) noise

    Parameters
    ----------
    num_samples : int
        number of samples/length of time series
    mean : float
        mean of Gaussian distribution to sample values from
    std_dev : float
        standard deviation of Gaussian distribution to sample values from

    Examples
    --------
    FIXME: Add docs.

    """
    return np.random.normal(mean, std_dev, size=num_samples)


def sinusoid(timeaxis, amp, freq, phase):
    """Generate sine wave

    Parameters
    ----------
    timeaxis : array-like
        time axis
    amp : float
        amplitude
    freq : float
        ordinary frequency (oscillations per unit time)
    phase : float
        phase, specifies in radian where in its cycle the oscillation is at t =
        0

    Examples
    --------
    import numpy as np
    mysignal = sinusoid(timeaxis=np.linspace(-np.pi, np.pi), amp=2, freq=0.5, phase=0.5*np.pi)

    """
    return amp * np.sin(2 * np.pi * freq * timeaxis + phase)


def harmonic(
    timeaxis=np.linspace(0, 1000, 1000),
    displ_init=0,
    velocity_init=1,
    ang_freq=1,
):
    """Generate signal defined by simple harmonic oscillator

    Generate signal defined by simple harmonic oscillator.

    Parameters
    ----------
    timeaxis : 1d numpy.array
        time axis
    displ_init : float
        initial displacement (y)
    velocity_init : float
        initial velocity (v)
    ang_freq : float
        angular frequency of oscillator

    Returns
    -------
    displ : 1d numpy.array
        time series of displacement (y) over time
    velocity : 1d numpy.array
        time series of velocity (v) over time

    Examples
    --------
    FIXME: Add docs.

    """
    model = Harmonic(
        timeaxis=timeaxis,
        init_cond={"y": displ_init, "v": velocity_init},
        ode_parameters={"ang_freq": ang_freq},
    )
    var_out = model.solver()
    displ = var_out.T[0]
    velocity = var_out.T[1]
    return displ, velocity


def harmonic_stochastic(
    timeaxis=np.linspace(0, 1000, 1000),
    displ_init=0,
    velocity_init=1,
    ang_freq_array=0.05 * np.random.rand(1000) + 0.1,
):
    """Generate stochastic signal derived from simple harmonic oscillator

    Generate stochastic signal derived from simple harmonic oscillator.

    Parameters
    ----------
    timeaxis : 1d numpy.array
        time axis
    displ_init : float
        initial displacement (y)
    velocity_init : float
        initial velocity (v)
    ang_freq_array : 1d numpy.array
        angular frequency of oscillator at each time point,
        should be the same length as timeaxis

    Returns
    -------
    displ : 1d numpy.array
        time series of displacement (y) over time
    velocity : 1d numpy.array
        time series of velocity (v) over time

    Examples
    --------
    FIXME: Add docs.

    """
    model = HarmonicStochastic(
        timeaxis=timeaxis,
        init_cond={"y": displ_init, "v": velocity_init},
        ode_parameters={"ang_freq": ang_freq_array},
    )
    var_out = model.solver()
    displ = var_out.T[0]
    velocity = var_out.T[1]
    return displ, velocity


def fitzhugh_nagumo(
    timeaxis=np.linspace(0, 1000, 1000),
    voltage_init=0,
    recovery_init=1,
    ext_stimulus=0.5,
    tau=12.5,
    a=0.7,
    b=0.8,
):
    """Generate signal defined by FitzHugh-Nagumo model

    Generate signal defined by FitzHigh-Nagumo model. Parameters defined as in

    Parameters
    ----------
    timeaxis : 1d numpy.array
        time axis
    voltage_init : float
        initial condition for voltage (v)
    recovery_init : float
        initial condition for linear recovery variable (w)
    ext_stimulus : float
        external stimulus parameter (RI_ext)
    tau : float
        tau parameter
    a : float
        a parameter
    b : float
        b parameter

    Returns
    -------
    voltage : 1d numpy.array
        time series of voltage (v) over time
    recovery : 1d numpy.array
        time series of linear recovery (w) variable over time

    Examples
    --------
    FIXME: Add docs.
    ext_stimulus=0.4, a=0.7 or 0.75, b=0.82 : shapes similar to htb2 localisation
    Insert parameters that make it look sinusoid here.

    """
    model = FitzHughNagumoModel(
        timeaxis=timeaxis,
        init_cond={"v": voltage_init, "w": recovery_init},
        ode_parameters={"ext_stimulus": ext_stimulus, "tau": tau, "a": a, "b": b},
    )
    var_out = model.solver()
    voltage = var_out.T[0]
    recovery = var_out.T[1]
    return voltage, recovery


def fitzhugh_nagumo_stochastic(
    timeaxis=np.linspace(0, 1000, 1000),
    voltage_init=0,
    recovery_init=1,
    ext_stimulus_array=0.05 * np.random.rand(1000) + 0.5,
    tau_array=0.05 * np.random.rand(1000) + 12.5,
    a_array=0.05 * np.random.rand(1000) + 0.7,
    b_array=0.05 * np.random.rand(1000) + 0.8,
):
    """
    FIXME: Add docs.
    """
    model = FitzHughNagumoModelStochastic(
        timeaxis=timeaxis,
        init_cond={"v": voltage_init, "w": recovery_init},
        ode_parameters={
            "ext_stimulus": ext_stimulus_array,
            "tau": tau_array,
            "a": a_array,
            "b": b_array,
        },
    )
    var_out = model.solver()
    voltage = var_out.T[0]
    recovery = var_out.T[1]
    return voltage, recovery


def gillespie_noise(
    num_timeseries,
    num_timepoints,
    noise_amp=100,
    noise_timescale=20,
    time_final=1500,
    grid_num_intervals=1000,
):
    """Generate Gillespie noise signal based on birth-death process

    Parameters
    ----------
    num_timeseries : int
        number of time series
    noise_amp : float
        noise amplitude
    noise_timescale : float
        noise timescale
    time_final : int
        final time of simulation
    grid_num_intervals : int
        number of time points for the matrix the simulation is interpolated onto
    num_timepoints : int, optional
        number of time points from the end of the interpolated grid to extract
        Gillespie noise from.  By default, set to half of grid_num_intervals

    Examples
    --------
    FIXME: Add docs.

    """
    # Transform parameters
    birthrate = noise_amp / noise_timescale
    deathrate = 1 / noise_timescale
    # Simulate birth-death process
    model = BirthDeathProcess(
        birthrate=birthrate, deathrate=deathrate, time_final=time_final
    )
    model.run(trajectories=num_timeseries)
    # Put on trajectories on grid
    model_grid = model.put_grid(num_intervals=grid_num_intervals)

    # By default, assumes that stationary state is reached halfway down the
    # time series.  This isn't always guaranteed and depends on the parameters --
    # best to check before using.
    if num_timepoints is None:
        num_timepoints = int(grid_num_intervals / 2)
    gill_noise_array = model_grid[:, -int(num_timepoints) :]

    # steady-state mean = birthrate/deathrate
    # steady-state variance = birthrate/deathrate
    # Normalise to 0 mean
    gill_noise_array -= birthrate / deathrate
    # Normalise noise amplitude -- not to 1 std dev because I want noise_amp to
    # mean something
    gill_noise_array /= np.sqrt(1 / deathrate)
    return gill_noise_array


def gillespie_noise_raw(
    num_timeseries,
    noise_amp=100,
    noise_timescale=20,
    time_final=1500,
):
    """Generate Gillespie noise signal based on birth-death process, preserving uneven timepoints

    Parameters
    ----------
    num_timeseries : int
        number of time series
    noise_amp : float
        noise amplitude
    noise_timescale : float
        noise timescale
    time_final : int
        final time of simulation
    Examples
    --------
    FIXME: Add docs.

    """
    # Transform parameters
    birthrate = noise_amp / noise_timescale
    deathrate = 1 / noise_timescale
    # Simulate birth-death process
    model = BirthDeathProcess(
        birthrate=birthrate, deathrate=deathrate, time_final=time_final
    )

    time_array = []
    species_array = []
    for repeat in range(num_timeseries):
        model.run_single()
        t, y = model.put_uneven()
        # By default, assumes that stationary state is reached halfway down the
        # time series.  This isn't always guaranteed and depends on the parameters --
        # best to check before using.
        time_array.append(t[int(len(t) / 2) :])
        y_half = y[int(len(y) / 2) :]
        y_half = y_half.astype(float)
        # steady-state mean = birthrate/deathrate
        # steady-state variance = birthrate/deathrate
        # Normalise to 0 mean
        y_half -= birthrate / deathrate
        # Normalise noise amplitude -- not to 1 std dev because I want noise_amp to
        # mean something
        y_half /= np.sqrt(1 / deathrate)
        species_array.append(y_half)

    # lists of arrays because number of timepoints is not even
    return time_array, species_array


class ODEModelBaseClass(ABC):
    """Base class for solving systems of 1st-order differential equations"""

    def __init__(self, timeaxis, init_cond, ode_parameters):
        """Initial variables

        Parameters
        ----------
        timeaxis : array
            Time axis, i.e. a sequence of time points for which to solve for
            variables.
        init_cond : dict
            Initial conditions; keys: variable names, values: initial conditions
        ode_parameters : dict
            Parameters for differential equations; keys: parameter names,
            values: value of parameter

        Examples
        --------
        FIXME: Add docs.

        """

        self.timeaxis = timeaxis
        self.init_cond = init_cond
        self.ode_parameters = ode_parameters

    @abstractmethod
    def odes(self, var_array, timeaxis, *args):
        # *args is intended for ODE parameters and accounts for different
        # systems having different numbers of parameters
        """Define ordinary differential equations"""
        pass

    def solver(self):
        return integrate.odeint(
            func=self.odes,
            y0=list(self.init_cond.values()),
            t=self.timeaxis,
            args=tuple(self.ode_parameters.values()),
        )


class Harmonic(ODEModelBaseClass):
    # using spring equation to describe simple harmonic motion
    # originally 2nd order DE: d2y/dt2 = (-(omega)**2)*y
    # re-written in the form of two 1st order ODEs
    # VARIABLES
    # y :: displacement
    # v :: velocity
    # PARAMETERS
    # ang_freq :: angular frequency, which is omega = sqrt(k/m) where
    #             k is the spring constant and m is mass
    def odes(self, var_array, timeaxis, *args):
        # Define variables
        y = var_array[0]
        v = var_array[1]
        # Define parameters
        ang_freq = args[0]
        # Define differential equations
        dy_dt = v
        dv_dt = (-((ang_freq) ** 2)) * y
        return [dy_dt, dv_dt]


class FitzHughNagumoModel(ODEModelBaseClass):
    # VARIABLES
    # v :: membrane voltage
    # w :: linear recovery variable
    # PARAMETERS
    # ext_stimulus ::
    # tau ::
    # a ::
    # b ::
    def odes(self, var_array, timeaxis, *args):
        # Define variables
        v = var_array[0]
        w = var_array[1]
        # Define parameters
        ext_stimulus = args[0]
        tau = args[1]
        a = args[2]
        b = args[3]
        # Define differential equations
        dv_dt = v - (v**3 / 3) - w + ext_stimulus
        dw_dt = (1 / tau) * (v + a - b * w)
        return [dv_dt, dw_dt]


# Alternative: Might write a base class for stochastic parameters instead
class HarmonicStochastic(Harmonic):
    def __init__(self, timeaxis, init_cond, ode_parameters):
        # Here, ode_parameters['ang_freq'] is an array.
        super().__init__(timeaxis, init_cond, ode_parameters)
        # Save ode_parameters because I'll change ode_parameters later
        # so that I don't have to re-implement solver method.
        self.ode_parameters_original = self.ode_parameters

        self.ang_freq_array = self.ode_parameters_original["ang_freq"]
        self.ang_freq_interp1d = interpolate.interp1d(
            self.timeaxis, self.ang_freq_array
        )

        # Redefine as function so it plays well with solver method
        self.ode_parameters["ang_freq"] = self.ang_freq

    # I can probably generalise this by writing a function constructor..?

    # Assumes timeaxis and ang_freq_array have the same number of elements
    # TODO: add dimension validation
    def ang_freq(self, t):
        # bodge: sometimes t overshoots
        if t > np.max(self.timeaxis):
            t = np.max(self.timeaxis)
        return self.ang_freq_interp1d(np.asarray(t))

    def odes(self, var_array, timeaxis, ang_freq):
        y = var_array[0]
        v = var_array[1]
        dypdt = v
        dvpdt = (-((ang_freq(timeaxis)) ** 2)) * y
        return [dypdt, dvpdt]


class FitzHughNagumoModelStochastic(FitzHughNagumoModel):
    def __init__(self, timeaxis, init_cond, ode_parameters):
        # Here, ode_parameters['ang_freq'] is an array.
        super().__init__(timeaxis, init_cond, ode_parameters)
        # Save ode_parameters because I'll change ode_parameters later
        # so that I don't have to re-implement solver method.
        self.ode_parameters_original = self.ode_parameters

        # FIXME: Make it more general so I can make any parameter stochastic
        # FIXME: If self.ode_parameters_original["XXX"] has one element,
        # make self.XXX_array an array of this one element, with length equal to
        # self.timeaxis

        # ext_stimulus
        self.ext_stimulus_array = self.ode_parameters_original["ext_stimulus"]
        self.ext_stimulus_interp1d = interpolate.interp1d(
            self.timeaxis, self.ext_stimulus_array
        )
        # Redefine as function so it plays well with solver method
        # IDEA: Properly write a dictionary of functions?
        self.ode_parameters["ext_stimulus"] = self.ext_stimulus

        # tau
        self.tau_array = self.ode_parameters_original["tau"]
        self.tau_interp1d = interpolate.interp1d(self.timeaxis, self.tau_array)
        self.ode_parameters["tau"] = self.tau

        # a
        self.a_array = self.ode_parameters_original["a"]
        self.a_interp1d = interpolate.interp1d(self.timeaxis, self.a_array)
        self.ode_parameters["a"] = self.a

        # b
        self.b_array = self.ode_parameters_original["b"]
        self.b_interp1d = interpolate.interp1d(self.timeaxis, self.b_array)
        self.ode_parameters["b"] = self.b

    # FIXME: I can probably generalise these methods by writing a decorator!
    # This decorator will likely be part of a base class for stochastic parametrisations

    # Assumes timeaxis and ext_stimulus_array have the same number of elements
    def ext_stimulus(self, t):
        # bodge: sometimes t overshoots
        if t > np.max(self.timeaxis):
            t = np.max(self.timeaxis)
        return self.ext_stimulus_interp1d(np.asarray(t))

    def tau(self, t):
        if t > np.max(self.timeaxis):
            t = np.max(self.timeaxis)
        return self.tau_interp1d(np.asarray(t))

    def a(self, t):
        if t > np.max(self.timeaxis):
            t = np.max(self.timeaxis)
        return self.a_interp1d(np.asarray(t))

    def b(self, t):
        if t > np.max(self.timeaxis):
            t = np.max(self.timeaxis)
        return self.b_interp1d(np.asarray(t))

    def odes(self, var_array, t, *args):
        # Define variables
        v = var_array[0]
        w = var_array[1]
        # Define parameters
        ext_stimulus = args[0]
        tau = args[1]
        a = args[2]
        b = args[3]
        # Define differential equations
        dv_dt = v - (v**3 / 3) - w + ext_stimulus(t)
        dw_dt = (1 / tau(t)) * (v + a(t) - b(t) * w)
        return [dv_dt, dw_dt]


class BirthDeathProcess:
    def __init__(self, birthrate, deathrate, time_final):
        self.model_file = "birth_death.psc"
        self.params = {
            "birthrate": birthrate,
            "deathrate": deathrate,
            "time_final": time_final,
        }

        # load stochastic birth-death model
        self.model = stochpy.SSA(model_file=self.model_file, dir=".")
        self.model.ChangeParameter("k", self.params["birthrate"])
        self.model.ChangeParameter("d", self.params["deathrate"])

    def run(self, trajectories=500):
        """Run simulations"""
        self.model.DoStochSim(
            end=self.params["time_final"],
            mode="time",
            trajectories=trajectories,
            quiet=False,
        )

    def run_single(self):
        """Run simulations, single trajectory"""
        self.model.DoStochSim(
            end=self.params["time_final"],
            mode="time",
            trajectories=1,
            quiet=False,
        )

    def put_grid(self, num_intervals):
        """Put trajectories on a matrix with regularly spaced time points"""
        time_axis = np.linspace(0, self.params["time_final"], num_intervals)
        dt = np.mean(np.diff(time_axis))
        self.model.GetRegularGrid(n_samples=num_intervals)
        # each row is one trajectory
        data = np.array(self.model.data_stochsim_grid.species[0]).astype("float")
        return data

    def put_uneven(self):
        """Return trajectory, with time points, unevenly spaced"""
        t = self.model.data_stochsim.time
        y = self.model.data_stochsim.species.T[0]
        return t, y
