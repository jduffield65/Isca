import numpy as np


def get_maxima_date_from_spline(spline, thresh=None, n_maxima=2):
    """
    Given a spline, this returns the dates (x variable) corresponding to the maxima.

    Args:
        spline: spline to find maxima of
        thresh: Only keep maxima with values above this.
        n_maxima: Keep at most this many maxima, if more than this then will only keep highest.
    """
    extrema_date = spline.derivative().roots(extrapolate=False)
    maxima_date = extrema_date[spline(extrema_date, 2) < 0]  # maxima have a negative second derivative
    maxima_values = spline(maxima_date)
    if thresh is not None:
        # Only keep maxima with value above threshold
        maxima_date = maxima_date[maxima_values > thresh]
        maxima_values = maxima_values[maxima_values > thresh]
    if len(maxima_date) > n_maxima:
        keep_ind = np.argsort(maxima_values)[-n_maxima:]
        maxima_date = maxima_date[keep_ind]
    return maxima_date
