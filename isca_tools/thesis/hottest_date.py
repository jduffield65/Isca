import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple


def get_extrema_date_from_spline(spline: CubicSpline, type: str = 'max', thresh: Optional[float] = None,
                                 n_extrema: int = 2) -> np.ndarray:
    """
    Given a spline, this returns the dates (x variable) corresponding to the maxima or minima.

    Args:
        spline: spline to find extrema of
        type: which extrema to find ('max' or 'min')
        thresh: Only keep maxima (minima) with values above (below) this.
        n_extrema: Keep at most this many extrema, if more than this then will only keep highest (lowest).
    """
    extrema_date = spline.derivative().roots(extrapolate=False)
    if type == 'max':
        extrema_date = extrema_date[spline(extrema_date, 2) < 0]  # maxima have a negative second derivative
    elif type == 'min':
        extrema_date = extrema_date[spline(extrema_date, 2) > 0]  # minima have a positive second derivative
    else:
        raise ValueError('type is not valid, it should be max or min')
    extrema_values = spline(extrema_date)
    if thresh is not None:
        # Only keep maxima with value above threshold
        if type == 'max':
            keep = extrema_values > thresh
        elif type == 'min':
            keep = extrema_values < thresh
        extrema_date = extrema_date[keep]
        extrema_values = extrema_values[keep]
    if len(extrema_date) > n_extrema:
        if type == 'max':
            keep_ind = np.argsort(extrema_values)[-n_extrema:]
        elif type == 'min':
            keep_ind = np.argsort(extrema_values)[:n_extrema]
        extrema_date = extrema_date[keep_ind]
    return extrema_date


def get_var_extrema_date(time: np.ndarray, var: np.ndarray, spline_spacing: int, type: str = 'max',
                         thresh_extrema: Optional[float] = None,
                         max_extrema: int = 2) -> Tuple[np.ndarray, CubicSpline]:
    """
    Finds the dates of extrema of a variable, given some smoothing is performed first.
    Also returns the splines themselves.

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`)
        var: `float [n_time]`</br>
            Value of variable at each time. Again, assume periodic
        spline_spacing: Interval of time to use to fit spline to variable. Smaller equals more accurate fit.
        type: which extrema to find ('max' or 'min')
        thresh_extrema: Only keep maxima (minima) with values above (below) this.
        max_extrema: Keep at most this many extrema, if more than this then will only keep highest (lowest).

    Returns:
        `extrema_date`: `float [max_extrema]`</br>
            Dates of extrema of var
        `spline_var`: Spline fit to var to find the extrema.
    """
    # Make so last element of arrays equal first as periodic
    time = np.append(time, time[-1]+1)
    var = np.append(var, var[0])
    # Get spline
    spline_var = CubicSpline(time[::spline_spacing], var[::spline_spacing], bc_type='periodic')
    extrema_date = get_extrema_date_from_spline(spline_var, type, thresh_extrema, max_extrema)
    return extrema_date, spline_var
