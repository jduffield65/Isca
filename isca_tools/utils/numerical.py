import numpy as np
import scipy.ndimage
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, Union, Callable
from scipy.optimize import root_scalar
import warnings
from ..utils.fourier import get_fourier_coef, fourier_series


def get_var_shift(x: np.ndarray, shift_time: Optional[float]=None, shift_phase: Optional[float]=None,
                time: Optional[np.ndarray] = None, time_start: Optional[float] = None,
                time_end: Optional[float] = None) -> np.ndarray:
    """
    Returns the periodic variable $x(t-t_{shift})$ where $t$=`time`, and $t_{shift}=$`shift_time`.
    If `shift_phase` is provided, will set `shift_time = shift_phase * period`.

    Args:
        x: `float [n_x]`</br>
            $x$ variable such that `x[i]` is the value of $x$ at time `time[i]`.
        shift_time: How much to shift $x$ by in units of `time`.
        shift_phase: Fraction of period to shift $x$ by.
        time: `float [n_x]`</br>
            Time such that `x[i]` is $x$ at time `time[i]`.</n>
            If `time` provided, will use spline to apply shift to $x$.</n>
            If `time` not provided, assume time is `np.arange(n_x)`, and will use `np.roll` to apply shift to $x$.
        time_start: Start time such that period is given by `time_end - time_start + 1`.
            If not provided, will set to min value in `time`.
        time_end: End time such that period is given by `time_end - time_start + 1`.
            If not provided, will set to max value in `time`.

    Returns:
        x_shift: `float [n_x]`</br>
            $x$ variable shifted in time such that `x_shift[i]` is value of $x$ at time `time[i] - shift_time`.
    """
    if time is not None:
        ind = np.argsort(time)
        if time_start is None:
            time_start = time[ind][0]
        if time_end is None:
            time_end = time[ind][-1]
        if time[ind][0] < time_start:
            raise ValueError(f'Min time={time[ind][0]} is less than time_start={time_start}')
        if time[ind][-1] > time_end:
            raise ValueError(f'Max time={time[ind][-1]} is greater than time_end={time_end}')
        x_spline_fit = CubicSpline(np.append(time[ind], time_end+time[ind][0]-time_start+1), np.append(x[ind], x[ind][0]),
                                   bc_type='periodic')
        period = time_end - time_start + 1
        if shift_phase is not None:
            shift_time = shift_phase * period
        x_shift = x_spline_fit(time - shift_time)
    else:
        if shift_phase is not None:
            shift_time = shift_phase * x.size
        if int(np.round(shift_time)) != shift_time:
            raise ValueError(f'shift_time={shift_time} is not a whole number - '
                             f'may be better using spline by providing time.')
        x_shift = np.roll(x, int(np.round(shift_time)))
    return x_shift


def polyval_phase(poly_coefs: np.ndarray, x: np.ndarray, time: Optional[np.ndarray] = None,
                  time_start: Optional[float] = None, time_end: Optional[float] = None,
                  coefs_fourier_amp: Optional[np.ndarray] = None,
                  coefs_fourier_phase: Optional[np.ndarray] = None,
                  pad_coefs_phase: bool = False) -> np.ndarray:
    """
    Given the polynomial coefficients found by `polyfit_phase` for fitting a polynomial
    of degree `len(polyfit)-2` of $x$ to $y$, this will return the approximation of $y$:

    $y_{approx} = \\frac{1}{2} \lambda_{phase}(x(t-T/4) - x(t+T/4)) + \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_n=$`poly_coefs[-1-n]`, $\lambda_{phase}=$`poly_coefs[0]` and
    $x$ is assumed periodic with period $T$.

    If `coefs_fourier_amp` is provided, then a fourier series will also be added to $y_{approx}$:

    $y_{approx} + \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi ft - \\Phi_n)$

    Args:
        poly_coefs: `float [n_deg+2]`</br>
            Polynomial coefficients as output by `polyfit_phase`, lowest power last.</br>
            $\lambda_{phase}=$`poly_coefs[0]` and $\lambda_n=$`poly_coefs[-1-n]`.
        x: `float [n_x]`</br>
            $x$ coordinates used to approximate $y$.
        time: `float [n_x]`</br>
            Time such that `x[i]` is $x$ at time `time[i]`.</n>
            If `time` provided, will use spline to apply shift to $x$.</n>
            If `time` not provided, assume time is `np.arange(n_x)`, and will use `np.roll` to apply shift to $x$.
        time_start: Start time such that period is given by `time_end - time_start + 1`.
            If not provided, will set to min value in `time`.
        time_end: End time such that period is given by `time_end - time_start + 1`.
            If not provided, will set to max value in `time`.
        coefs_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients $F_n$.
        coefs_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients $\\Phi_n$.
        pad_coefs_phase: If `True`, expect `coefs_fourier_phase` to be of length `n_harmonics+1` with first value
            equal to zero.

    Returns:
        y_approx: `float [n_x]`</br>
            Polynomial approximation to $y$, possibly including phase and Fourier terms.
    """
    # In this case, poly_coefs are output of polyfit_with_phase so first coefficient is the phase coefficient
    y_approx = np.polyval(poly_coefs[1:], x)
    x_shift = 0.5 * (get_var_shift(x, shift_phase=0.25, time=time, time_start=time_start, time_end=time_end) -
                     get_var_shift(x, shift_phase=-0.25, time=time, time_start=time_start, time_end=time_end))
    if coefs_fourier_amp is not None:
        time_use = np.arange(x.size) if time is None else time
        y_residual_fourier = fourier_series(time_use, coefs_fourier_amp, coefs_fourier_phase, pad_coefs_phase)
    else:
        y_residual_fourier = 0
    return y_approx + poly_coefs[0] * x_shift + y_residual_fourier


def polyfit_phase(x: np.ndarray, y: np.ndarray,
                  deg: int, time: Optional[np.ndarray] = None, time_start: Optional[float] = None,
                  time_end: Optional[float] = None,
                  deg_phase_calc: int = 10, resample: bool = False,
                  include_phase: bool = True, fourier_harmonics: Optional[Union[int, np.ndarray]] = None,
                  integ_method: str = 'spline',
                  pad_coefs_phase: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    This fits a polynomial `y_approx(x) = p[0] * x**deg + ... + p[deg]` of degree `deg` to points (x, y) as `np.polyfit`
    but also includes additional phase shift term such that the total approximation for y is:

    $y_{approx} = \\frac{1}{2} \lambda_{phase}(x(t-T/4) - x(t+T/4)) + \sum_{n=0}^{n_{deg}} \lambda_n x^n$

    where $\lambda_n=$`poly_coefs[-1-n]` and $\lambda_{phase}=$`poly_coefs[0]`.
    $x$ is assumed periodic with period $T=$`time[-1]-time[0]+time_spacing`.

    The phase component, $y_{phase}=\\frac{1}{2} \lambda_{phase}(x(t-T/4) - x(t+T/4))$, is found first from the
    residual of $y-y_{best}$, where $y_{best}$ is the polynomial approximation of degree `deg_phase_calc`.

    $\sum_{n=0}^{n_{deg}} \lambda_n x^n$ is then found by doing the normal polynomial approximation of degree
    `deg` to the residual $y-y_{phase}$.

    If `fourier_harmonics` is provided, then a fourier series will also be added to $y_{approx}$ containing all
    harmonics, $n$, in `fourier_harmonics`:

    $y_{approx} + \\sum_{n} F_n\\cos(2n\\pi ft - \\Phi_n)$

    The idea behind this is to account for part of $y$ not directly related to $x$.

    Args:
        x: `float [n_x]`</br>
            $x$ coordinates used to approximate $y$. `x[i]` is value at time `time[i]`.
        y: `float [n_x]`</br>
            $y$ coordinate correesponding to each $x$.
        deg: Degree of the fitting polynomial. If negative, will only do the phase fitting.
        time: `float [n_x]`</br>
            Time such that `x[i]` is $x$ and `y[i]` is $y$ at time `time[i]`.</n>
            If time provided, will use spline to apply phase shift to $x$.</n>
            If time not provided, assume time is `np.arange(n_x)`, and will use `np.roll` to apply phase shift to $x$.
        time_start: Start time such that period is given by `time_end - time_start + 1`.
            If not provided, will set to min value in `time`.
        time_end: End time such that period is given by `time_end - time_start + 1`.
            If not provided, will set to max value in `time`.
        deg_phase_calc: Degree of the fitting polynomial to use in the phase term calculation.
            Should be a large integer.
        resample: If `True`, will use `resample_data` to resample x and y before each calling of
            `np.polyfit`.
        include_phase: If `False`, will only call `np.polyfit`, but first return value will be 0 indicating
            no phase shift. Only makes sense to call this rather than `np.polyfit` if you want to use `resample`.
        fourier_harmonics: `int [n_harmonics_include]`</br>
            After applied polynomial of degree `deg_phase_calc` and fit the phase factor, a residual is obtained
            `y_residual = y - y_best - y_phase`. If `fourier_harmonics` is provided, a fourier
            series will be directly fit to this residual including all the harmonics in `fourier_harmonics`.
            If `fourier_harmonics` is an integer, all harmonics up to and including the value will be fit.</br>
            The final polynomial of degree `deg` will then be fit to `y - y_phase - y_fourier`.</br>
            Idea behind this is to account for part of $y$ not directly related to $x$.
        integ_method: How to perform the integration when calculating Fourier coefficients..</br>
            If `spline`, will fit a spline and then integrate the spline, otherwise will use `scipy.integrate.simpson`.
        pad_coefs_phase: If `True`, will set `coefs_fourier_phase` to length `fourier_harmonics.max()+1`, with
            the first value as zero. Otherwise will be size `fourier_harmonics.max()`.

    Returns:
        poly_coefs: `float [n_deg+2]`
            Polynomial coefficients, phase first and then normal output of `np.polyfit` with lowest power last.
        coefs_fourier_amp: `float [fourier_harmonics.max()+1]`</br>
            `coefs_fourier_amp[n]` is the amplitude fourier coefficient for the $n^{th}$ harmonic.</br>
            First value will be zero, because $0^{th}$ harmonic is just a constant and so will be found in
            polynomial fitting.</br>
            Only returned if `fourier_harmonics` is provided.
        coefs_fourier_phase: `float [fourier_harmonics.max()]`</br>
            `coefs_fourier_phase[n]` is the phase fourier coefficient for the $(n+1)^{th}$ harmonic.</br>
            Only returned if `fourier_harmonics` is provided.
            If `pad_coefs_phase`, will pad at start with a zero so of length `fourier_harmonics.max()+1`.
    """
    coefs = np.zeros(np.clip(deg, 0, 1000) + 2)  # first coef is phase coef
    if resample:
        x_fit, y_fit = resample_data(time, x, y)[1:]
    else:
        x_fit = x
        y_fit = y
    if not include_phase:
        coefs[1:] = np.polyfit(x_fit, y_fit, deg)       # don't do phase stuff so 1st value is 0
    else:
        y_best_polyfit = np.polyval(np.polyfit(x_fit, y_fit, deg_phase_calc), x)
        x_shift = 0.5 * (get_var_shift(x, shift_phase=0.25, time=time, time_start=time_start, time_end=time_end) -
                         get_var_shift(x, shift_phase=-0.25, time=time, time_start=time_start, time_end=time_end))
        if resample:
            x_shift_fit, y_residual_fit = resample_data(time, x_shift, y - y_best_polyfit)[1:]
        else:
            x_shift_fit = x_shift
            y_residual_fit = y - y_best_polyfit
        coefs[[0, -1]] = np.polyfit(x_shift_fit, y_residual_fit, 1)
        y_no_phase = y - polyval_phase(coefs, x, time, time_start, time_end)  # residual after removing phase dependent term

        if fourier_harmonics is not None:
            time_use = np.arange(x.size) if time is None else time
            if not all(np.ediff1d(time_use) == np.ediff1d(time_use)[0]):
                raise ValueError('Can only include Fourier with evenly spaced data')
            if isinstance(fourier_harmonics, int):
                # if int, use all harmonics up to value indicated but without 0th harmonic
                fourier_harmonics = np.arange(1, fourier_harmonics+1)
            coefs_fourier_amp = np.zeros(np.max(fourier_harmonics)+1)
            coefs_fourier_phase = np.zeros(np.max(fourier_harmonics))
            for n in fourier_harmonics:
                if n==0:
                    warnings.warn('Will not fit 0th harmonic as constant will be fit in polynomial')
                else:
                    coefs_fourier_amp[n], coefs_fourier_phase[n-1] = \
                        get_fourier_coef(time_use, y_no_phase - y_best_polyfit, n, integ_method)
            y_residual_fourier = fourier_series(time_use, coefs_fourier_amp, coefs_fourier_phase)
            y_no_phase = y_no_phase - y_residual_fourier

        if deg >= 0:
            if resample:
                x_fit, y_no_phase_fit = resample_data(time, x, y_no_phase)[1:]
            else:
                x_fit = x
                y_no_phase_fit = y_no_phase
            coefs[1:] += np.polyfit(x_fit, y_no_phase_fit, deg)

    if fourier_harmonics is None:
        return coefs
    else:
        if pad_coefs_phase:
            # Set first value to zero
            coefs_fourier_phase = np.hstack((np.zeros(1), coefs_fourier_phase))
        return coefs, coefs_fourier_amp, coefs_fourier_phase


def resample_data_distance(time: Optional[np.ndarray], x: np.ndarray, y: np.ndarray, n_return: Optional[int] = None,
                           bc_type: str = 'periodic', norm: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given that `x[i]` and `y[i]` both occur at time `time[i]`, this resamples data to return `n_return` values of
    $x$ and $y$ evenly spaced along the line connecting all (x, y) coordinates.

    Args:
        time: `float [n_time]`</br>
            Times such that `x[i]` and `y[i]` correspond to time `time[i]`.</br>
            If time not provided, assume time is `np.arange(n_x)`.
        x: `float [n_time]`</br>
            Value of variable $x$ at each time.
        y: `float [n_time]`</br>
            Value of variable $y$ at each time.
        n_return:  Number of resampled data, will set to `n_time` if not provided.
        bc_type: Boundary condition type in `scipy.interpolate.CubicSpline` for `x` and `y`.
        norm: If `True` will normalize `x` and `y` so both have a range of 1, before calculating distance along line.

    Returns:
        times_return: `float [n_return]`</br>
            Times of returned $x$ and $y$ such that they are evenly spaced along line connecting all input
            (x, y) coordinates.
        x_return_out: `float [n_return_out]`</br>
            $x$ values corresponding to `times_return`.
        y_return: `float [n_return_out]`</br>
            $y$ values corresponding to `times_return` and `x_return`.
    """
    if n_return is None:
        n_return = x.size
    if time is None:
        time = np.arange(x.size)
    time_spacing = np.median(np.ediff1d(time))
    # dist[i] is distance along line from (x[0], y[0]) at time=time[i]
    if norm:
        coords_dist_calc = np.vstack((x / (np.max(x) - np.min(x)), y / (np.max(y) - np.min(y))))
    else:
        coords_dist_calc = np.vstack((x, y))
    dist = np.append(0, np.cumsum(np.sqrt(np.sum(np.diff(coords_dist_calc, axis=1) ** 2, axis=0))))
    x_spline = CubicSpline(np.append(time, [time[-1] + time_spacing]), np.append(x, x[0]),
                                             bc_type=bc_type)
    y_spline = CubicSpline(np.append(time, [time[-1] + time_spacing]), np.append(y, y[0]),
                                             bc_type=bc_type)
    dist_spline = CubicSpline(time, dist)
    dist_return = np.linspace(dist[0], dist[-1], n_return)
    # Adjust first and last values by tiny amount, to ensure that within the range when trying to solve
    small = 0.0001 * (dist_return[1]-dist_return[0])
    dist_return[0] += small
    dist_return[-1] -= small
    time_resample = np.zeros(n_return)
    for i in range(n_return):
        time_resample[i] = dist_spline.solve(dist_return[i], extrapolate=False)[0]
    return time_resample, x_spline(time_resample), y_spline(time_resample)


def resample_data(time: Optional[np.ndarray], x: np.ndarray, y: np.ndarray, x_return: Optional[np.ndarray] = None,
                  n_return: Optional[int] = None, bc_type: str = 'periodic',
                  extrapolate: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given that `x[i]` and `y[i]` both occur at time `time[i]`, this resamples data to return values of `y`
    corresponding to `x_return`.

    Args:
        time: `float [n_time]`</br>
            Times such that `x[i]` and `y[i]` correspond to time `time[i]`. It assumes time has a spacing of 1, and
            starts with 0, so for a 360-day year, it would be `np.arange(360)`.
        x: `float [n_time]`</br>
            Value of variable $x$ at each time.
        y: `float [n_time]`</br>
            Value of variable $y$ at each time.
        x_return: `float [n_return]`</br>
            Values of $x$ for the resampled $y$ data to be returned. If not provided, will use
            `np.linspace(x.min(), x.max(), n_return)`.
        n_return: Number of resampled data if `x_return` is not provided. Will set to `n_time` neither this
            nor `x_return` provided.
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
    if n_return is None:
        n_return = x.size
    if time is None:
        time = np.arange(x.size)
    time_spacing = np.median(np.ediff1d(time))
    if 'periodic' in bc_type:
        x_spline = CubicSpline(np.append(time, [time[-1]+time_spacing]), np.append(x, x[0]),
                               bc_type=bc_type)
        y_spline = CubicSpline(np.append(time, [time[-1]+time_spacing]), np.append(y, y[0]),
                               bc_type=bc_type)
    else:
        x_spline = CubicSpline(time, x, bc_type=bc_type)
        y_spline = CubicSpline(time, y, bc_type=bc_type)
    if x_return is None:
        x_return = np.linspace(x.min(), x.max(), n_return)
    times_return = []
    for i in range(x_return.size):
        times_return += [*x_spline.solve(x_return[i], extrapolate=extrapolate)]
    times_return = np.asarray(times_return) % time[-1]      # make return times between 0 and time[-1]
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

def interp_nan(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set all nan values in `y` based on the two nearest values of `x` for which `y` is not nan.

    Args:
        x: `[n_points]` Independent variable e.g. pressure
        y: `[n_points]` Dependent variable e.g. temperature

    Returns:
        y: `[n_points]` Same as input but with all nans replaced through interpolation.
        not_valid_idx: `[n_not_valid]` Indices of nans in `y` that were replaced through interpolation.
    """
    # Set all nan values in x based on the two nearest indices that are not nan
    # using linear interpolation
    not_valid = np.isnan(y)
    if not_valid.sum() == 0:
        return y, np.zeros(0)
    else:
        valid_idx = np.where(~not_valid)[0]
        not_valid_idx = np.where(not_valid)[0]
        for i in not_valid_idx:
            if i < valid_idx[0]:
                # Case: before first valid point
                j1, j2 = valid_idx[0], valid_idx[1]
            elif i > valid_idx[-1]:
                # Case: after last valid point
                j1, j2 = valid_idx[-2], valid_idx[-1]
            else:
                # Case: between valid points
                # nearest valid indices around i
                j2 = valid_idx[valid_idx > i][0]
                j1 = valid_idx[valid_idx < i][-1]
            # Linear interpolation between (x[j1], y[j1]) and (x[j2], y[j2])
            slope = (y[j2] - y[j1]) / (x[j2] - x[j1])
            y[i] = y[j1] + slope * (x[i] - x[j1])
    return y, not_valid_idx


def hybrid_root_find(
    residual: Callable,
    guess: float,
    search_radius: float,
    n_bracket_samples: int = 32
) -> float:
    """Find a root of a 1-D function using a fast secant/brentq hybrid method.

    Attempts:
      1) Solve using two secant initialisations (fastest).
      2) If those fail or leave the search interval, construct a local bracket
         by scanning the interval at fixed resolution.
      3) Solve inside the constructed bracket using brentq (guaranteed).

    Args:
        residual: Function whose root is sought. Called as residual(x).
        guess: Initial guess around which to search for a solution.
        search_radius: Half-width of the interval in which the root is assumed
            to lie. The search interval is [guess - search_radius,
            guess + search_radius].
        n_bracket_samples: Number of sample points used to locate a sign
            change when constructing a fallback bracket (1-D array of this
            length is evaluated).

    Returns:
        The root of `residual` inside the specified interval.

    Raises:
        ValueError: If no sign change is located in the interval despite an
            assumed root, or if brentq fails to converge.
    """
    lower = guess - search_radius
    upper = guess + search_radius

    # ---- 1. Fast attempt: secant method --------
    try:
        sol = root_scalar(
            residual,
            x0=guess,
            x1=guess + 0.5 * search_radius,
            method="secant"
        )
        if sol.converged and (lower <= sol.root <= upper):
            return sol.root
    except Exception:
        pass

    # ---- 2. Second secant attempt --------
    try:
        sol = root_scalar(
            residual,
            x0=guess - 0.5 * search_radius,
            x1=guess + search_radius,
            method="secant"
        )
        if sol.converged and (lower <= sol.root <= upper):
            return sol.root
    except Exception:
        pass

    # ---- 3. Construct a local bracket by scanning --------
    grid = np.linspace(lower, upper, n_bracket_samples)
    fvals = np.array([residual(x) for x in grid])

    # Find intervals with sign change
    sign_idx = np.where(np.signbit(fvals[:-1]) != np.signbit(fvals[1:]))[0]
    if len(sign_idx) == 0:
        raise ValueError(
            "No sign change found in interval despite assumed root."
        )

    # Take the first sign-change interval
    i = sign_idx[0]
    a, b = grid[i], grid[i + 1]

    # ---- 4. Guaranteed convergence using brentq --------
    sol = root_scalar(residual, bracket=[a, b], method='brentq')
    if not sol.converged:
        raise ValueError("brentq did not converge within constructed bracket.")

    return sol.root


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Compute the (lower) weighted median of 1D data.

    This returns the first value `x[i]` (after sorting by `x`) for which the
    cumulative weight is at least half of the total weight. This is a common
    convention and always yields a valid weighted median (though the weighted
    median set can be an interval in special tie cases).

    Args:
        x: 1D array of values, shape `(n,)`.
        w: 1D array of non-negative weights, shape `(n,)`. Typically you want
            strictly positive weights on the points you include.

    Returns:
        Weighted median value (float). By construction this implementation
        returns a value that is an element of `x`.

    Raises:
        ValueError: If `x` and `w` have different shapes, are not 1D, or if the
            total weight is zero.
    """
    x = np.asarray(x)
    w = np.asarray(w)

    if x.ndim != 1 or w.ndim != 1:
        raise ValueError("x and w must be 1D arrays.")
    if x.shape != w.shape:
        raise ValueError("x and w must have the same shape.")
    wsum = np.sum(w)
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0.")

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    cw = np.cumsum(w)
    cutoff = 0.5 * wsum
    return float(x[np.searchsorted(cw, cutoff, side="left")])