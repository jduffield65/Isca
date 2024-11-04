import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple
import scipy.ndimage


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


def get_var_extrema_date(time: np.ndarray, var: np.ndarray, smooth_window: int = 1,
                         type: str = 'max', thresh_extrema: Optional[float] = None,
                         max_extrema: int = 2, smooth_method: str = 'convolve') -> Tuple[np.ndarray, CubicSpline]:
    """
    Finds the dates of extrema of a variable, given some smoothing is performed first.
    Also returns the splines themselves.

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`)
        var: `float [n_time]`</br>
            Value of variable at each time. Again, assume periodic
        smooth_window: Number of days to use to smooth `var` before finding extrema. Smaller equals more accurate fit.
            `1` is perfect fit.
        type: which extrema to find ('max' or 'min')
        thresh_extrema: Only keep maxima (minima) with values above (below) this.
        max_extrema: Keep at most this many extrema, if more than this then will only keep highest (lowest).
        smooth_method: `convolve` or `spline`
            If `convolve`, will return smooth via convolution with window of length `smooth_window`.
            If `spline`, will fit a spline using every `smooth_window` days.

    Returns:
        `extrema_date`: `float [max_extrema]`</br>
            Dates of extrema of var
        `spline_var`: Spline fit to var to find the extrema.
    """
    if smooth_method.lower() == 'spline':
        # Make so last element of arrays equal first as periodic
        time_smooth = np.append(time, time[-1]+1)[::smooth_window]
        var_smooth = np.append(var, var[0])[::smooth_window]
    elif smooth_method.lower() == 'convolve':
        var_smooth = scipy.ndimage.convolve(var, np.ones(smooth_window) / smooth_window, mode='wrap')
        time_smooth = np.append(time, time[-1] + 1)
        var_smooth = np.append(var_smooth, var_smooth[0])
    else:
        raise  ValueError('smooth_method must be either spline or convolve')
    # Spline var is the spline replicating var_smooth exactly i.e. spline_var(t) = var_smooth[t] if t in time_smooth
    spline_var = CubicSpline(time_smooth, var_smooth, bc_type='periodic')
    extrema_date = get_extrema_date_from_spline(spline_var, type, thresh_extrema, max_extrema)
    return extrema_date, spline_var

