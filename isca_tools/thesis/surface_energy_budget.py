import numpy as np
from ..utils import fourier
from scipy import optimize
from scipy.interpolate import CubicSpline
import warnings
from typing import Optional


def gamma_linear_approx(time: np.ndarray, temp: np.ndarray,
                        lambda_const: np.ndarray, lambda_time_lag: Optional[np.ndarray] = None,
                        lambda_sq: float = 0) -> np.ndarray:
    n_lambda = len(lambda_const) - 1
    if lambda_time_lag is None:
        lambda_temp = np.sum(lambda_const[1:]) * temp
    else:
        # To apply phase shift, need spline so can compute temp anomaly at times outside range in `time`.
        temp_spline_fit = CubicSpline(np.append(time, time[-1]+1), np.append(temp, temp[0]),
                                      bc_type='periodic')
        lambda_temp = np.zeros_like(temp)
        for i in range(n_lambda):
            lambda_temp += lambda_const[1+i] * temp_spline_fit(time - lambda_time_lag[i])

    return lambda_const[0] + lambda_temp + lambda_sq * (temp-np.mean(temp))**2


def swdn_from_temp_fourier(time: np.ndarray, temp_fourier_amp: np.ndarray, temp_fourier_phase: np.ndarray,
                           heat_capacity: float, lambda_const: np.ndarray,
                           lambda_time_lag: Optional[np.ndarray] = None, lambda_sq: float = 0) -> np.ndarray:
    n_year_days = len(time)
    temp_fourier = fourier.fourier_series(time, n_year_days, temp_fourier_amp, temp_fourier_phase)
    gamma = gamma_linear_approx(time, temp_fourier, lambda_const,
                                lambda_time_lag, lambda_sq)
    dtemp_dt = fourier.fourier_series_deriv(time, n_year_days, temp_fourier_amp, temp_fourier_phase)
    return heat_capacity * dtemp_dt + gamma


def get_temp_fourier(time: np.ndarray, swdn: np.ndarray, heat_capacity: float,
                     lambda_const: np.ndarray, lambda_time_lag: Optional[np.ndarray] = None,
                     lambda_sq: float = 0, n_harmonics: int = 2,
                     include_sw_phase: bool = False, numerical: bool = False):
    """
    Seeks a fourier solution of the form $T(t) = \\frac{T_0}{2} + \\sum_{n=1}^{N} T_n\\cos(2n\\pi ft - \\phi_n)$
    to the linearized surface energy budget of the general form:

    $$
    C\\frac{\partial T}{\partial t} = F(t) - \lambda_0 - \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) -
    \lambda_{sq}\\left(T(t) - \overline{T}\\right)^2
    $$

    where:

    * $\overline{T} = T_0/2$ is the mean temperature.
    * $F(t) = \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi ft - \\varphi_n)$ is the Fourier representation
    of the downward shortwave radiation at the surface, $SW^{\downarrow}$.
    * $\lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) +
    \lambda_{sq}\\left(T(t) - \overline{T}\\right)^2$ is the approximation for
    $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$

    The solution is exact if $\lambda_{sq}=0$ and has the form:

    * $T_0 = (F_0-2\lambda_0)/\sum_{i=1}^{N_{\lambda}}\lambda_i$
    * $T_n = \\frac{F_n \cos(\\varphi_n)\sqrt{1+\\tan^2\phi_n}}{(2\pi nfC - \sum_i \lambda_i \sin \Phi_i)\\tan\phi_n +
    \sum_i \lambda_i \cos \Phi_i}$
    * $\\tan \phi_n = \\frac{2\pi nfC + \\tan \\varphi_n \sum_i \lambda_i \cos \Phi_i - \sum_i \lambda_i \sin \Phi_i}
    {-2\pi nfC \\tan \\varphi_n + \sum_i \lambda_i \cos \Phi_i - \\tan \\varphi_n \sum_i \lambda_i \sin \Phi_i}$
    * $\Phi_i = 2\pi nf \Delta_i$ for each $n$ (units of radians, whereas $\Lambda_i$ is in units of time - days).

    If $\lambda_{sq}\\neq 0$, an approximate numerical solution will be obtained, still of the Fourier form.

    Args:
        time:
        sw_fourier_amp:
        sw_fourier_phase:
        lambda_const:
        heat_capacity:
        n_harmonics:

    Returns:

    """
    n_year_days = len(time)
    n_lambda = len(lambda_const)-1
    if lambda_time_lag is None:
        lambda_time_lag = np.zeros(n_lambda)

    # Get fourier representation of SW radiation
    sw_fourier_amp, sw_fourier_phase = fourier.get_fourier_fit(time, swdn, n_year_days, n_harmonics)[1:]
    if not include_sw_phase:
        sw_fourier_phase = np.zeros(n_harmonics)
    sw_fourier = fourier.fourier_series(time, n_year_days, sw_fourier_amp, sw_fourier_phase)


    if numerical or lambda_sq != 0:
        if not numerical:
            warnings.warn('Analytic solution not possible with lambda_sq non zero')

        def fit_func(time_array, *args):
            fourier_amp_coef = np.asarray([args[i] for i in range(n_harmonics + 1)])
            fourier_phase_coef = np.asarray([args[i] for i in range(n_harmonics + 1, len(args))])
            return swdn_from_temp_fourier(time_array, fourier_amp_coef, fourier_phase_coef, heat_capacity, lambda_const,
                                          lambda_time_lag, lambda_sq)

        # force positive phase coefficient to match analytic solution
        bounds_lower = [-np.inf] * (n_harmonics + 1) + [0] * n_harmonics
        bounds_upper = [np.inf] * (n_harmonics + 1) + [2 * np.pi] * n_harmonics
        args_found = optimize.curve_fit(fit_func, time, sw_fourier, np.ones(2 * n_harmonics + 1),
                                        bounds=(bounds_lower, bounds_upper))[0]
        temp_fourier_amp = args_found[:n_harmonics + 1]
        temp_fourier_phase = args_found[n_harmonics + 1:]
    else:
        f = 1/(n_year_days*24*60**2)    # must have frequency in units of s^{-1} to deal with phase stuff in radians
        sw_tan = np.tan(sw_fourier_phase)

        temp_fourier_amp = np.zeros(n_harmonics+1)
        temp_fourier_phase = np.zeros(n_harmonics)
        temp_fourier_amp[0] = (sw_fourier_amp[0] - 2*lambda_const[0]) / np.sum(lambda_const[1:])

        for n in range(1, n_harmonics+1):
            # convert lambda time lag from days to units of radians
            lambda_phase_const = lambda_time_lag * 2 * n * np.pi / n_year_days
            lambda_cos = np.sum(lambda_const[1:] * np.cos(lambda_phase_const))
            lambda_sin = np.sum(lambda_const[1:] * np.sin(lambda_phase_const))

            temp_fourier_phase[n-1] = np.arctan((2*np.pi*n*f*heat_capacity + sw_tan[n-1] * lambda_cos - lambda_sin) / (
                    -2*np.pi*n*f*heat_capacity * sw_tan[n-1] + lambda_cos - sw_tan[n-1]*lambda_sin))
            temp_fourier_amp[n] = sw_fourier_amp[n] * np.cos(sw_fourier_phase[n-1]) / np.cos(
                temp_fourier_phase[n-1]) / (
                    (2*np.pi*n*f*heat_capacity - lambda_sin)*np.tan(temp_fourier_phase[n-1]) + lambda_cos)
    temp_fourier = fourier.fourier_series(time, n_year_days, temp_fourier_amp, temp_fourier_phase)
    return temp_fourier, temp_fourier_amp, temp_fourier_phase, sw_fourier_amp, sw_fourier_phase

