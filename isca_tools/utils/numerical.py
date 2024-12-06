import numpy as np
import scipy.ndimage
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple
import warnings


# def apply_polyfit(x: np.ndarray, poly_coefs: np.ndarray) -> np.ndarray:
#     """
#     Given the polynomial coefficients found by `np.polyfit` for fitting a polynomial
#     of degree `len(polyfit)-1` of $x$ to $y$, this will return the approximation of $y$:
#
#     $y_{approx} = \sum_{n=0}^{n_{deg}} \lambda_n x^n$
#
#     where $\lambda_n=$`poly_coefs[-1-n]`.
#
#     Args:
#         x: `float [n_x]`</br>
#             $x$ coordinates used to approximate $y$.
#         poly_coefs: `float [n_deg+1]`</br>
#             Polynomial coefficients as output by `np.polyfit`, lowest power last.</br>
#
#     Returns:
#         y_approx: `float [n_x]`</br>
#             Polynomial approximation to $y$.
#     """
#     return np.sum([poly_coefs[-i - 1] * x ** i for i in range(len(poly_coefs))], axis=0)


def apply_polyfit_phase(x: np.ndarray, poly_coefs: np.ndarray) -> np.ndarray:
    """
    Given the polynomial coefficients found by `polyfit_phase` for fitting a polynomial
    of degree `len(polyfit)-2` of $x$ to $y$, this will return the approximation of $y$:

    $y_{approx} = \\frac{1}{2} \lambda_{phase}(x(t-T/4) - x(t+T/4)) + \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_n=$`poly_coefs[-1-n]`, $\lambda_{phase}=$`poly_coefs[0]` and
    $x$ is assumed periodic with period $T=$.

    Args:
        x: `float [n_x]`</br>
            $x$ coordinates used to approximate $y$.
        poly_coefs: `float [n_deg+2]`</br>
            Polynomial coefficients as output by `polyfit_phase`, lowest power last.</br>
            $\lambda_{phase}=$`poly_coefs[0]` and $\lambda_n=$`poly_coefs[-1-n]`.

    Returns:
        y_approx: `float [n_x]`</br>
            Polynomial approximation to $y$.
    """
    # In this case, poly_coefs are output of polyfit_with_phase so first coefficient is the phase coefficient
    y_approx = np.polyval(poly_coefs[1:], x)
    # y_approx = np.sum([poly_coefs[-i - 1] * x ** i for i in range(len(poly_coefs) - 1)], axis=0)
    # time_spacing = np.median(np.ediff1d(time))
    # x_spline_fit = CubicSpline(np.append(time, time[-1] + time_spacing), np.append(x, x[0]),
    #                            bc_type='periodic')
    # period_length = time[-1] - time[0] + time_spacing
    shift_n_elements = int(np.round(x.size / 4))
    if shift_n_elements != x.size / 4:
        warnings.warn('Cannot shift by whole number of elements - may be better using spline')
    x_shift = 0.5 * (np.roll(x, shift_n_elements) - np.roll(x, -shift_n_elements))
    return y_approx + poly_coefs[0] * x_shift


def polyfit_phase(x: np.ndarray, y: np.ndarray,
                  deg: int,
                  deg_phase_calc: int = 10) -> np.ndarray:
    """
    This fits a polynomial `y_approx(x) = p[0] * x**deg + ... + p[deg]` of degree `deg` to points (x, y) as `np.polyfit`
    but also includes additional phase shift term such that the total approximation for y is:

    $y_{approx} = \\frac{1}{2} \lambda_{phase}(x(t-T/4) - x(t+T/4)) + \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_n=$`poly_coefs[-1-n]` and $\lambda_{phase}=$`poly_coefs[0]`.
    $x$ is assumed periodic with period $T=$`time[-1]-time[0]+time_spacing`.

    The phase component, `y_phase=`$\lambda_{phase} x(t-T/4)$, is found first from the residual of $y-y_{best}$,
    where $y_{best}$ is the polynomial approximation of degree `deg_phase_calc`.

    $\sum_{n=0}^{n_{deg}} \lambda_n x^n$ is then found by doing the normal polynomial approximation of degree
    `deg` to the residual $y-y_{phase}$.

    Args:
        x: `float [n_x]`</br>
            $x$ coordinates used to approximate $y$.
        y: `float [n_x]`</br>
            $y$ coordinate correesponding to each $x$.
        deg: Degree of the fitting polynomial.
        deg_phase_calc: Degree of the fitting polynomial to use in the phase term calculation.
            Should be a large integer.

    Returns:
        poly_coefs: `float [n_deg+2]`
            Polynomial coefficients, phase first and then normal output of `np.polyfit` with lowest power last.
    """
    coefs = np.zeros(deg + 2)  # last coef is phase coef
    # time_spacing = np.median(np.ediff1d(time))
    # x_spline_fit = CubicSpline(np.append(time, time[-1] + time_spacing), np.append(x, x[0]),
    #                            bc_type='periodic')
    # y_best_polyfit = apply_polyfit(x, np.polyfit(x, y, deg_phase_calc))
    # period_length = time[-1] - time[0] + time_spacing
    # # Use linalg to find coefficient not polyfit as know 0th order coefficient is 0 i.e. want y=mx not y=mx+c
    # coefs[0] = np.linalg.lstsq(x_spline_fit(time - period_length / 4)[:, np.newaxis], y - y_best_polyfit,
    #                            rcond=-1)[0][0]
    shift_n_elements = int(np.round(x.size / 4))
    if shift_n_elements != x.size / 4:
        warnings.warn('Cannot shift by whole number of elements - may be better using spline')
    y_best_polyfit = np.polyval(np.polyfit(x, y, deg_phase_calc), x)
    x_shift = 0.5 * (np.roll(x, shift_n_elements) - np.roll(x, -shift_n_elements))[:, np.newaxis]
    coefs[0] = np.linalg.lstsq(x_shift, y - y_best_polyfit, rcond=-1)[0][0]
    y_no_phase = y - apply_polyfit_phase(x, coefs)  # residual after removing phase dependent term
    coefs[1:] = np.polyfit(x, y_no_phase, deg)
    return coefs


def resample_data(time: np.ndarray, x: np.ndarray, y: np.ndarray, x_return: Optional[np.ndarray] = None,
                  n_return: int = 360, bc_type: str = 'periodic',
                  extrapolate: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given that `x[i]` and `y[i]` both occur at time `time[i]`, this resamples data to return values of `y`
    corresponding to `x_return`.

    Args:
        time: `float [n_time]`</br>
            Times such that `x[i]` and `y[i]` correspond to time `time[i]`.
        x: `float [n_time]`</br>
            Value of variable $x$ at each time.
        y: `float [n_time]`</br>
            Value of variable $y$ at each time.
        x_return: `float [n_return]`</br>
            Values of $x$ for the resampled $y$ data to be returned. If not provided, will use
            `np.linspace(x.min(), x.max(), n_return)`.
        n_return: Number of resampled data if `x_return` is not provided.
        bc_type: Boundary condition type in `scipy.interpolate.CubicSpline`.
        extrapolate: Whether to extrapolate if any `x_return` outside`x` is provided.

    Returns:
        times_return: `float [n_return_out]`</br>
            Times corresponding to `x_return`. Not necessarily `n_return` values because can have multiple $y$
            values for each $x$.
        x_return_out: `float [n_return_out]`</br>
            $x$ values corresponding to `times_return`. Will only contain values in `x_return`, but may contain
            multiple of each.
        y_return: `float [n_return_out]`</br>
            $y$ values corresponding to `times_return` and `x_return_out`.
    """
    if 'periodic' in bc_type:
        x_spline = CubicSpline(np.append(time, [time[-1]+1]), np.append(x, x[0]),
                               bc_type=bc_type)
        y_spline = CubicSpline(np.append(time, [time[-1]+1]), np.append(y, y[0]),
                               bc_type=bc_type)
    else:
        x_spline = CubicSpline(time, x, bc_type=bc_type)
        y_spline = CubicSpline(time, y, bc_type=bc_type)
    if x_return is None:
        x_return = np.linspace(x.min(), x.max(), n_return)
    times_return = []
    for i in range(x_return.size):
        times_return+= [*x_spline.solve(x_return[i], extrapolate=extrapolate)]
    times_return = np.asarray(times_return)
    return times_return, x_spline(times_return), y_spline(times_return)


def spline_integral(x: np.ndarray, dy_dx: np.ndarray, y0: float = 0, x0: Optional[float] = None,
                    x_return: Optional[np.ndarray] = None,
                    periodic: bool = False) -> np.ndarray:
    """
    Uses spline integration to solve for $y$ given $\\frac{dy}{dx}$ such that $y=y_0$ at $x=x_0$.

    Args:
        x: `float [n_x]`</br>
            Values of $x$ where $\\frac{dy}{dx}$ given, and used to fit the spline.
        dy_dx: `float [n_x]`</br>
            Values of $\\frac{dy}{dx}$ corresponding to $x$, and used to fit the spline.
        y0: Boundary condition, $y(x_0)=y_0$.
        x0: Boundary condition, $y(x_0)=y_0$.</br>
            If not given, will assume `x0=x_return[0]`.
        x_return: `float [n_x_return]`</br>
            Values of $y$ are returned for these $x$ values. If not given, will set to `x`.
        periodic: Whether to use periodic boundary condition.</br>
            If periodic expect $\\frac{dy}{dx}$ at $x=$`x[-1]+x_spacing` is equal to `dy_dx[0]`.

    Returns:
        y_return: `float [n_x_return]`</br>
            Values of $y$ corresponding to `x_return.
    """
    if periodic:
        x_spacing = np.median(np.ediff1d(x))
        spline_use = CubicSpline(np.append(x, x[-1] + x_spacing), np.append(dy_dx, dy_dx[0]), bc_type='periodic')
    else:
        spline_use = CubicSpline(x, dy_dx)
    if x_return is None:
        x_return = x
    if x0 is None:
        x0 = x_return[0]
    y = np.full_like(x_return, y0, dtype=float)
    for i in range(x_return.size):
        y[i] += spline_use.integrate(x0, x_return[i], extrapolate='periodic' if periodic else None)
    return y


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
        smooth_window: Number of time steps to use to smooth `var` before finding extrema.
            Smaller equals more accurate fit. `1` is perfect fit.
        type: which extrema to find ('max' or 'min')
        thresh_extrema: Only keep maxima (minima) with values above (below) this.
        max_extrema: Keep at most this many extrema, if more than this then will only keep highest (lowest).
        smooth_method: `convolve` or `spline`</br>
            If `convolve`, will smooth via convolution with window of length `smooth_window`.
            If `spline`, will fit a spline using every `smooth_window` values of `time`.

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
