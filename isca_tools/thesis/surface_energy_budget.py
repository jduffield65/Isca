import numpy as np
from ..utils import fourier
from scipy import optimize
from scipy.interpolate import CubicSpline
import warnings
from typing import Optional, Tuple


def gamma_linear_approx(time: np.ndarray, temp: np.ndarray,
                        lambda_const: np.ndarray, lambda_time_lag: Optional[np.ndarray] = None,
                        lambda_sq: float = 0, temp_anom_squared: Optional[np.ndarray] = None) -> np.ndarray:
    """
    This approximates $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$ as:

    $$\Gamma^{\\uparrow} \\approx \lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) +
    \lambda_{sq}\\left(T(t) - \overline{T}\\right)^2$$

    where:

    * $LW^{\\uparrow}$ is upward longwave radiation at the surface i.e. $\\sigma T^4$ or `lwup_sfc` output by Isca.
        Units: $Wm^{-2}$.
    * $LW^{\\downarrow}$ is downward longwave radiation at the surface i.e. `lwdn_sfc` output by Isca.
        Units: $Wm^{-2}$.
    * $LH^{\\uparrow}$ is the upward latent heat radiation at the surface i.e. `flux_lhe` output by Isca.
        Units: $Wm^{-2}$.
    * $SH^{\\uparrow}$ is the upward sensible heat radiation at the surface i.e. `flux_t` output by Isca.
        Units: $Wm^{-2}$.

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`).
        temp: `float [n_time]`</br>
            Surface temperature, $T$ for each day in `time`. Assumes periodic so `temp[0]` is the temperature
            at time `time[-1]+1`.
        lambda_const: `float [n_lambda+1]`</br>
            The constants $\lambda_i$ used in the approximation.</br>
            `lambda_const[0]` is $\lambda_0$ and `lambda_const[i]` is $\lambda_{i}$ for $i>0$.
        lambda_time_lag: `float [n_lambda]`</br>
            The constants $\Lambda_i$ used in the approximation.</br>
            `lambda_time_lag[0]` is $\Lambda_1$ and `lambda_time_lag[i]` is $\Lambda_{i+1}$ for $i>0$.
        lambda_sq: The constant $\lambda_{sq}$ used in the approximation.
        temp_anom_squared: `float [n_time]`</br>
            The value of $\\left(T(t) - \overline{T}\\right)^2$ to use in the calculation. If `None`,
            will compute from `temp`.

    Returns:
        `float [n_time]`</br>
            The approximation $\Gamma^{\\uparrow} \\approx \lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i)
            + \lambda_{sq}\\left(T(t) - \overline{T}\\right)^2$ with units of $Wm^{-2}$.
    """
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
    if temp_anom_squared is None:
        temp_anom_squared = (temp-np.mean(temp))**2
    return lambda_const[0] + lambda_temp + lambda_sq * temp_anom_squared


def swdn_from_temp_fourier(time: np.ndarray, temp_fourier_amp: np.ndarray, temp_fourier_phase: np.ndarray,
                           heat_capacity: float, lambda_const: np.ndarray,
                           lambda_time_lag: Optional[np.ndarray] = None, lambda_sq: float = 0,
                           day_seconds: float = 86400, single_harmonic_squared: bool = False) -> np.ndarray:
    """
    This inverts the linearized surface energy budget to return an approximation for downward shortwave radiation
    at the surface, $F(t)$, given a Fourier approximation for surface temperature,
    $T(t) = \\frac{T_0}{2} + \\sum_{n=1}^{N} T_n\\cos(2n\\pi ft - \\phi_n)$:

    $$
    F(t) \\approx C\\frac{\partial T}{\partial t} + \lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) +
    \lambda_{sq}\\left(T(t) - \overline{T}\\right)^2
    $$


    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`).
        temp_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients for surface temperature: $T_n$.
        temp_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients for surface temperature: $\phi_n$.
        heat_capacity: $C$, the heat capacity of the surface in units of $JK^{-1}m^{-2}$.</br>
            Obtained from mixed layer depth of ocean using
            [`get_heat_capacity`](/code/utils/radiation/#isca_tools.utils.radiation.get_heat_capacity).
        lambda_const: `float [n_lambda+1]`</br>
            The constants $\lambda_i$ used in the approximation for
            $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$.</br>
            `lambda_const[0]` is $\lambda_0$ and `lambda_const[i]` is $\lambda_{i}$ for $i>0$.
        lambda_time_lag: `float [n_lambda]`</br>
            The constants $\Lambda_i$ used in the approximation for $\Gamma^{\\uparrow}$.</br>
            `lambda_time_lag[0]` is $\Lambda_1$ and `lambda_time_lag[i]` is $\Lambda_{i+1}$ for $i>0$.
        lambda_sq: The constant $\lambda_{sq}$ used in the approximation for $\Gamma^{\\uparrow}$.
        day_seconds: Duration of a day in seconds.
        single_harmonic_squared: If `True`, the $\lambda_{sq}T^2$ term in $\Gamma^{\\uparrow}$ will only
            use the first harmonic, not all harmonics.

    Returns:
        `float [n_time]`</br>
            Approximation for downward shortwave radiation at the surface.
            Units: $Wm^{-2}$.
    """
    n_year_days = len(time)
    temp_fourier = fourier.fourier_series(time, n_year_days, temp_fourier_amp, temp_fourier_phase)
    if single_harmonic_squared:
        temp_anom_squared = fourier.fourier_series(time, n_year_days, [0, temp_fourier_amp[1]],
                                                   [temp_fourier_phase[0]])**2
        gamma = gamma_linear_approx(time, temp_fourier, lambda_const,
                                    lambda_time_lag, lambda_sq, temp_anom_squared)
    else:
        gamma = gamma_linear_approx(time, temp_fourier, lambda_const,
                                    lambda_time_lag, lambda_sq)
    dtemp_dt = fourier.fourier_series_deriv(time, n_year_days, temp_fourier_amp, temp_fourier_phase, day_seconds)
    return heat_capacity * dtemp_dt + gamma


def get_temp_fourier(time: np.ndarray, swdn: np.ndarray, heat_capacity: float,
                     lambda_const: np.ndarray, lambda_time_lag: Optional[np.ndarray] = None,
                     lambda_sq: float = 0, n_harmonics: int = 2,
                     include_sw_phase: bool = False, numerical: bool = False,
                     day_seconds: float = 86400,
                     single_harmonic_squared: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seeks a fourier solution of the form $T(t) = \\frac{T_0}{2} + \\sum_{n=1}^{N} T_n\\cos(2n\\pi ft - \\phi_n)$
    to the linearized surface energy budget of the general form:

    $$
    C\\frac{\partial T}{\partial t} = F(t) - \lambda_0 - \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) -
    \lambda_{sq}\\left(T(t) - \overline{T}\\right)^2
    $$

    where:

    * $C$ is the heat capacity of the surface
    * $\overline{T} = T_0/2$ is the mean temperature.
    * $F(t) = \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi ft - \\varphi_n)$ is the Fourier representation
    of the downward shortwave radiation at the surface, $SW^{\downarrow}$.
    * $\lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) +
    \lambda_{sq}\\left(T(t) - \overline{T}\\right)^2$ is the approximation for
    $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$

    The solution is exact if $\lambda_{sq}=0$ and has the form:

    * $T_0 = (F_0-2\lambda_0)/\sum_{i=1}^{N_{\lambda}}\lambda_i$
    * $T_n = \\frac{F_n \cos(\\varphi_n)\sqrt{1+\\tan^2\phi_n}}{
    (2\pi nfC - \sum_i \lambda_i \sin \Phi_{ni})\\tan\phi_n + \sum_i \lambda_i \cos \Phi_{ni}}$
    * $\\tan \phi_n = \\frac{2\pi nfC + \\tan \\varphi_n \sum_i \lambda_i \cos \Phi_{ni} -
    \sum_i \lambda_i \sin \Phi_{ni}}{-2\pi nfC \\tan \\varphi_n + \sum_i \lambda_i \cos \Phi_{ni} -
    \\tan \\varphi_n \sum_i \lambda_i \sin \Phi_{ni}}$
    * $\Phi_{ni} = 2\pi nf \Lambda_i$ (units of radians, whereas $\Lambda_i$ is in units of time - days).

    If $\lambda_{sq}\\neq 0$, an approximate numerical solution will be obtained, still of the Fourier form.

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`).
        swdn: `float [n_time]`</br>
            Downward shortwave radiation at the surface, $SW^{\\downarrow}$. I.e. `swdn_sfc` output by Isca.
            Units: $Wm^{-2}$.
        heat_capacity: $C$, the heat capacity of the surface in units of $JK^{-1}m^{-2}$.</br>
            Obtained from mixed layer depth of ocean using
            [`get_heat_capacity`](/code/utils/radiation/#isca_tools.utils.radiation.get_heat_capacity).
        lambda_const: `float [n_lambda+1]`</br>
            The constants $\lambda_i$ used in the approximation for
            $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$.</br>
            `lambda_const[0]` is $\lambda_0$ and `lambda_const[i]` is $\lambda_{i}$ for $i>0$.
        lambda_time_lag: `float [n_lambda]`</br>
            The constants $\Lambda_i$ used in the approximation for $\Gamma^{\\uparrow}$.</br>
            `lambda_time_lag[0]` is $\Lambda_1$ and `lambda_time_lag[i]` is $\Lambda_{i+1}$ for $i>0$.
        lambda_sq: The constant $\lambda_{sq}$ used in the approximation for $\Gamma^{\\uparrow}$.
        n_harmonics: Number of harmonics to use to fit fourier series for both $T(t)$ and $F(t)$, $N$.
        include_sw_phase: If `False`, will set all phase factors, $\\varphi_n=0$, in Fourier expansion of $F(t)$.</br>
            These phase factors are usually very small, and it makes the solution for $T(t)$ more simple if they
            are set to 0, hence the option.
        numerical: If `True`, will compute solution for $T(t)$ numerically using [`scipy.optimize.curve_fit`](
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) rather than using the
            analytic solution. Will always return numerical solution if `lambda_sq` $\\neq 0$.
        day_seconds: Duration of a day in seconds.
        single_harmonic_squared: If `True`, the $\lambda_{sq}T^2$ term in $\Gamma^{\\uparrow}$ will only
            use the first harmonic, not all harmonics.

    Returns:
        temp_fourier: `float [n_time]`</br>
            The Fourier series solution that was found for surface temperature, $T$.
        temp_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients for surface temperature: $T_n$.
        temp_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients for surface temperature: $\phi_n$.
        sw_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients for shortwave radiation, $SW^{\\downarrow}$: $F_n$.
        sw_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients for shortwave radiation, $SW^{\\downarrow}$: $\\varphi_n$.
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
                                          lambda_time_lag, lambda_sq, day_seconds, single_harmonic_squared)

        # force positive phase coefficient to match analytic solution
        bounds_lower = [-np.inf] * (n_harmonics + 1) + [0] * n_harmonics
        bounds_upper = [np.inf] * (n_harmonics + 1) + [2 * np.pi] * n_harmonics
        try:
            args_found = optimize.curve_fit(fit_func, time, sw_fourier, np.ones(2 * n_harmonics + 1),
                                            bounds=(bounds_lower, bounds_upper))[0]
        except RuntimeError:
            warnings.warn('Hit Runtime Error, trying without bounds')
            args_found = optimize.curve_fit(fit_func, time, sw_fourier, np.ones(2 * n_harmonics + 1))[0]
        temp_fourier_amp = args_found[:n_harmonics + 1]
        temp_fourier_phase = args_found[n_harmonics + 1:]
    else:
        f = 1/(n_year_days*day_seconds)    # must have frequency in units of s^{-1} to deal with phase stuff in radians
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