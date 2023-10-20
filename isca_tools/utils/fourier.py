import numpy as np
from typing import List, Union, Tuple
from scipy.interpolate import CubicSpline
import scipy.integrate


def fourier_series(time: np.ndarray, period: float, coefs_amp: Union[List[float], np.ndarray],
                   coefs_phase: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    For $N$ harmonics, the fourier series with frequency $f$ is:

    $$F(t) \\approx \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi ft - \\Phi_n)$$

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`)
        period: Period of fourier series in days, $1/f$
        coefs_amp: `float [N+1]`</br>
            The amplitude coefficients, $F_n$
        coefs_phase: `float [N]`</br>
            The phase coefficients in radians, $\Phi_n$

    Returns:
        `float [n_time]`</br>
            Value of Fourier series solution at each time
    """
    n_harmonics = len(coefs_amp)
    ans = 0.5 * coefs_amp[0]
    for n in range(1, n_harmonics):
        ans += coefs_amp[n] * np.cos(2*n*np.pi*time/period - coefs_phase[n-1])
    return ans


def fourier_series_deriv(time: np.ndarray, period: float, coefs_amp: List[float],
                         coefs_phase: List[float], day_seconds: float = 86400) -> np.ndarray:
    """
    For $N$ harmonics, the derivative of a fourier series with frequency $f$ is:

    $$\\frac{dF}{dt} \\approx -\\sum_{n=1}^{N} 2n\\pi fF_n\\sin(2n\\pi ft - \\Phi_n)$$

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`)
        period: Period of fourier series in days, $1/f$
        coefs_amp: `float [N+1]`</br>
            The amplitude coefficients, $F_n$. $F_0$ needs to be provided even though it is not used.
        coefs_phase: `float [N]`</br>
            The phase coefficients in radians, $\Phi_n$
        day_seconds: Duration of a day in seconds

    Returns:
        `float [n_time]`</br>
            Value of derivative to Fourier series solution at each time. Units is units of $F$ divided by seconds.
    """
    n_harmonics = len(coefs_amp)
    ans = np.zeros_like(time, dtype=float)
    for n in range(1, n_harmonics):
        ans -= coefs_amp[n] * np.sin(2*n*np.pi*time/period - coefs_phase[n-1]) * (2*n*np.pi/period)
    return ans / day_seconds     # convert units from per day to per second


def get_fourier_coef(time: np.ndarray, var: np.ndarray, period: float, n: int,
                     integ_method: str = 'spline') -> [Union[float, Tuple[float, float]]]:
    """
    This calculates the analytic solution for the amplitude and phase coefficients for the `n`th harmonic
    [🔗](https://www.bragitoff.com/2021/05/fourier-series-coefficients-and-visualization-python-program/)

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`)
        var: `float [n_time]`</br>
            Variable to fit fourier series to.
        period: Period of fourier series in days, $1/f$
        n: Harmonic to find coefficients for, if 0, will just return amplitude coefficient.
            Otherwise, will return an amplitude and phase coefficient.
        integ_method: How to perform the integration.</br>
            If `spline`, will fit a spline and then integrate the spline, otherwise will use `scipy.integrate.simpson`.
    Returns:
        `amp_coef`: The amplitude fourier coefficient $F_n$.
        `phase_coef`: The phase fourier coefficient $\\Phi_n$. Will not return if $n=0$.
    """
    # Computes the analytical fourier coefficients for the n harmonic of a given function
    # With integrate method = spline works very well i.e. fit spline then use spline.integrate functionality
    # Otherwise, there are problems with the integration especially at the limits e.g. t=0 and t=T.
    if integ_method == 'spline':
        var = np.append(var, var[0])
        time = np.append(time, time[-1]+1)
    if n == 0:
        if integ_method == 'spline':
            spline = CubicSpline(time, var, bc_type='periodic')
            return 2/period * spline.integrate(0, period)
        else:
            return 2/period * scipy.integrate.simpson(var, time)
    else:
        # constants for acos(t) + bsin(t) form
        if integ_method == 'spline':
            spline = CubicSpline(time,var * np.cos(2*n*np.pi*time/period), bc_type='periodic')
            cos_coef = 2/period * spline.integrate(0, period)
            sin_curve = var * np.sin(2*n*np.pi*time/period)
            sin_curve[-1] = 0       # Need first and last value to be the same to be periodic spline
                                    # usually have last value equal 1e-10 so not equal
            spline = CubicSpline(time,sin_curve, bc_type='periodic')
            sin_coef = 2/period * spline.integrate(0, period)
        else:
            cos_coef = 2/period * scipy.integrate.simpson(var * np.cos(2*n*np.pi*time/period), time)
            sin_coef = 2/period * scipy.integrate.simpson(var * np.sin(2*n*np.pi*time/period), time)
        # constants for Acos(t-phi) form
        phase_coef = np.arctan(sin_coef/cos_coef)
        amp_coef = cos_coef / np.cos(phase_coef)
        return amp_coef, phase_coef


def get_fourier_fit(time: np.ndarray, var: np.ndarray, period: float, n_harmonics: int,
                    integ_method: str = 'spline') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtains the Fourier series solution for $F=$`var`, using $N=$`n_harmonics`:

    $$F(t) \\approx \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi ft - \\Phi_n)$$

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`)
        var: `float [n_time]`</br>
            Variable to fit fourier series to.
        period: Period of fourier series in days, $1/f$
        n_harmonics: Number of harmonics to use to fit fourier series, $N$.
        integ_method: How to perform the integration when obtaining Fourier coefficients.</br>
            If `spline`, will fit a spline and then integrate the spline, otherwise will use `scipy.integrate.simpson`.

    Returns:
        `fourier_solution`: `float [n_time]`</br>
            The Fourier series solution that was fit to `var`.
        `amp_coef`: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients $F_n$.
        `phase_coef`: `float [n_harmonics]`</br>
            The phase Fourier coefficients $\\Phi_n$.
    """
    # Returns the fourier fit of a function using a given number of harmonics
    amp_coefs = np.zeros(n_harmonics+1)
    phase_coefs = np.zeros(n_harmonics)
    amp_coefs[0] = get_fourier_coef(time, var, period, 0, integ_method)
    for i in range(1, n_harmonics+1):
        amp_coefs[i], phase_coefs[i-1] = get_fourier_coef(time, var, period, i, integ_method)
    return fourier_series(time, period, amp_coefs, phase_coefs), amp_coefs, phase_coefs

# These functions find the fourier fit numerically
# Initially did this way but analytical way makes more sense.
# def func_for_curve_fit(time, *fitting_params):
#     n_harmonics = int(np.ceil(len(fitting_params)/2))
#     coefs_amp = fitting_params[:n_harmonics]
#     coefs_phase = fitting_params[n_harmonics:]
#     return fourier_series(time, coefs_amp, coefs_phase)
#
# # Coefficients have analytic solution - should use these rather than fitting
# def get_fourier_fit_numerical(time, var, n_harmonics, phase_shift_guess=0):
#     mean_var = np.mean(var)
#     mean_amp = np.mean(np.abs(var-mean_var))
#     phase_guess = np.deg2rad(phase_shift_guess/n_year_days*360)
#     coefs_guess = [mean_var*2] + [mean_amp]*n_harmonics + [phase_guess]*n_harmonics
#     # Limit phase shift coefs between +/- 180 degrees
#     bounds = ([-np.inf]*(n_harmonics+1)+[-np.pi]*n_harmonics,
#               [np.inf]*(n_harmonics+1)+[np.pi]*n_harmonics)
#     fit_result = scipy.optimize.curve_fit(func_for_curve_fit, time, var, coefs_guess, bounds=bounds)
#     std_error = np.sqrt(np.diag(fit_result[1]))
#     return (fourier_series(time, fit_result[0][:n_harmonics+1], fit_result[0][n_harmonics+1:]),
#             fit_result[0][:n_harmonics+1], fit_result[0][n_harmonics+1:], std_error[:n_harmonics+1],
#             std_error[n_harmonics+1:])
