import numpy as np
from ..utils import fourier, numerical
from scipy import optimize
from scipy.interpolate import CubicSpline
import warnings
from typing import Optional, Tuple, Union


def get_temp_exact(time: np.ndarray, swdn_sfc: np.ndarray, gamma: np.ndarray, heat_capacity: float,
                   temp0: Optional[float] = None,
                   day_seconds: int = 86400, seek_fourier: bool = False, n_harmonics: int = 10,
                   lambda_const_guess: float =3) -> np.ndarray:
    """
    For a given $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$,
    and downward shortwave radiation at the surface, $SW^{\\downarrow}$, this returns the exact surface
    temperature, $T$, by using a spline to integrate:

    $$
    \\frac{d T}{d t} = \\frac{SW^{\\downarrow} - \Gamma^{\\uparrow}}{C}
    $$

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`).
        swdn_sfc: `float [n_time]`</br>
            Downward shortwave radiation at the surface.</br>
            Units: $Wm^{-2}$.
        gamma: `float [n_time]`</br>
            $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$
            where $LW$ is longwave, $LH$ is latent heat and $SH$ is sensible heat.</br>
            Units: $Wm^{-2}$.
        heat_capacity: $C$, the heat capacity of the surface in units of $JK^{-1}m^{-2}$.</br>
            Obtained from mixed layer depth of ocean using
            [`get_heat_capacity`](/code/utils/radiation/#isca_tools.utils.radiation.get_heat_capacity).
        temp0: Temperature at `time[0]`.</br>
            If not given, will just return the anomaly i.e. $T - \overline{T}$.
            Units: $K$.
        day_seconds: Duration of a day in seconds.
        seek_fourier: If `True` will seek a Fourier solution of `n_harmonics` that best satisfies the differential
            equation. Otherwise will fit a spline and integrate it.
        n_harmonics: Number of harmonics to use in Fourier solution.
        lambda_const_guess: If $\Gamma = \lambda_0 + \lambda T(t)$ then an analytic fourier solution exists.
            This is a guess for $\lambda$ that will be used as starting point for seeking numerical fourier solution.

    Returns:
        `float [n_time]`</br>
            Exact value at temperature to satisfy the differential equation at each time.</br>
            Units: $K$.
    """
    dtemp_dt_exact = (swdn_sfc - gamma) / heat_capacity
    if not seek_fourier:
        temp_approx = numerical.spline_integral(time * day_seconds, dtemp_dt_exact,
                                                y0=0 if temp0 is None else temp0, periodic=True)
        if temp0 is None:
            return temp_approx - np.mean(temp_approx)
        else:
            return temp_approx
    else:
        def fit_func_deriv(time_array, *args):
            # first n_harmonic values in args are the temperature amplitude coefficients (excluding T_0)
            # last n_harmonic values in args are the temperature phase coefficients
            fourier_amp_coef = np.asarray([args[i] for i in range(n_harmonics)])
            # make first coefficient 0 as doesn't impact on derivative
            fourier_amp_coef = np.append([0], fourier_amp_coef)
            fourier_phase_coef = np.asarray([args[i] for i in range(n_harmonics, len(args))])
            return fourier.fourier_series_deriv(time_array, time_array.size, fourier_amp_coef,
                                                fourier_phase_coef, day_seconds)

        # Starting guess is linear 1 harmonic linear analytical solution given gamma=lambda_const_guess*temp
        # so only 1st harmonic coefficients of amplitude and phase are needed, rest are set to zero
        p0 = np.zeros(2 * n_harmonics)
        f = 1/(time.size*day_seconds)
        p0[n_harmonics] = np.arctan((2 * np.pi * f * heat_capacity) / lambda_const_guess)
        sw_fourier_amp = fourier.get_fourier_fit(time, swdn_sfc, time.size, 1)[1]
        p0[0] = sw_fourier_amp[1] / np.cos(p0[n_harmonics]) / (
                (2 * np.pi * f * heat_capacity) * np.tan(p0[n_harmonics]) + lambda_const_guess)
        args_found = optimize.curve_fit(fit_func_deriv, time, dtemp_dt_exact, p0)[0]
        temp_fourier_amp = np.append([0], args_found[:n_harmonics])
        temp_fourier_phase = args_found[n_harmonics:]
        temp_approx = fourier.fourier_series(time, time.size, temp_fourier_amp, temp_fourier_phase)
        if temp0 is None:
            return temp_approx      # fourier solution is anomaly so just return this
        else:
            return temp_approx - temp_approx[0] + temp0


def get_temp_fourier_analytic(time: np.ndarray, swdn_sfc: np.ndarray, heat_capacity: float,
                              lambda_const: float, lambda_phase: float = 0,
                              lambda_sq: float = 0, n_harmonics_sw: int = 2,
                              n_harmonics_temp: Optional[int] = None,
                              include_sw_phase: bool = False,
                              day_seconds: float = 86400,
                              phase_delay: float = 90) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seeks a fourier solution of the form $T'(t) = \\sum_{n=1}^{N} T_n\\cos(2n\\pi ft - \\phi_n)$
    to the linearized surface energy budget of the general form:

    $$
    C\\frac{\partial T'}{\partial t} = F(t) - \lambda_0 - \lambda T'(t) - \lambda_{phase}T'(t-\Lambda) -
    \lambda_{sq} T'^2(t)
    $$

    where:

    * $T' = T(t) - \overline{T}$ is the surface temperature anomaly
    * $C$ is the heat capacity of the surface
    * $\overline{T} = T_0/2$ is the mean temperature.
    * $F(t) = \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi ft - \\varphi_n)$ is the Fourier representation
    of the downward shortwave radiation at the surface, $SW^{\downarrow}$.
    * $\lambda_0 + \lambda T'(t) + \lambda_{phase}T'(t-\Lambda) + \lambda_{sq} T'^2(t)$ is the approximation for
    $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$

    The solution is exact if $\lambda_{sq}=0$ and has the form:

    * $T_0 = (F_0-2\lambda_0)/\sum_{i=1}^{N_{\lambda}}\lambda_i$
    * $T_n = \\frac{F_n \cos(\\varphi_n)\sqrt{1+\\tan^2\phi_n}}{
    (2\pi nfC - \sum_i \lambda_i \sin \Phi_{ni})\\tan\phi_n + \sum_i \lambda_i \cos \Phi_{ni}}$
    * $\\tan \phi_n = \\frac{2\pi nfC + \\tan \\varphi_n \sum_i \lambda_i \cos \Phi_{ni} -
    \sum_i \lambda_i \sin \Phi_{ni}}{-2\pi nfC \\tan \\varphi_n + \sum_i \lambda_i \cos \Phi_{ni} -
    \\tan \\varphi_n \sum_i \lambda_i \sin \Phi_{ni}}$
    * $\Phi_{ni} = 2\pi nf \Lambda_i$ (units of radians, whereas $\Lambda_i$ is in units of time - days).

    If $\lambda_{sq}\\neq 0$, an approximate numerical solution will be obtained, assuming
    $T'^2(t) \\approx (T_1\\cos(2\\pi ft - \\phi_1))^2$.

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
        lambda_nl: `float [n_lambda_nl]`
            The constants $\lambda_{nl}$ used in the approximation for $\Gamma^{\\uparrow}$.
            `[0]` is squared contribution, `[1]` is cubed, ...
        n_harmonics: Number of harmonics to use to fit fourier series for both $T(t)$ and $F(t)$, $N$.
        include_sw_phase: If `False`, will set all phase factors, $\\varphi_n=0$, in Fourier expansion of $F(t)$.</br>
            These phase factors are usually very small, and it makes the solution for $T(t)$ more simple if they
            are set to 0, hence the option.
        numerical: If `True`, will compute solution for $T(t)$ numerically using [`scipy.optimize.curve_fit`](
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) rather than using the
            analytic solution. Will always return numerical solution if `lambda_sq` $\\neq 0$.
        day_seconds: Duration of a day in seconds.
        single_harmonic_nl: If `True`, the $\lambda_{nl_j}T'^j$ terms in $\Gamma^{\\uparrow}$ will only
            use the first harmonic, not all harmonics.
        return_anomaly: If `True`, the first return variable, `temp_fourier` will be
            the temperature anomaly, i.e. it will not include $T_0$.

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
    if n_harmonics_temp is None:
        n_harmonics_temp = n_harmonics_sw

    if n_harmonics_temp == 1 and lambda_sq != 0:
        raise ValueError('Cannot solve for non-zero lambda_sq with single harmonic')

    # Get fourier representation of SW radiation
    if n_harmonics_sw > n_harmonics_temp:
        raise ValueError('Cannot have more harmonics for swdn_sfc than have for temperature')
    sw_fourier_amp = np.zeros(n_harmonics_temp + 1)
    sw_fourier_phase = np.zeros(n_harmonics_temp)
    sw_fourier_amp[:n_harmonics_sw+1], sw_fourier_phase[:n_harmonics_sw] = \
        fourier.get_fourier_fit(time, swdn_sfc, n_year_days, n_harmonics_sw)[1:]
    if not include_sw_phase:
        sw_fourier_phase = np.zeros(n_harmonics_temp)
    sw_tan = np.tan(sw_fourier_phase)
    sw_cos = np.cos(sw_fourier_phase)
    sw_sin = np.sin(sw_fourier_phase)
    f = 1 / (n_year_days * day_seconds)  # must have frequency in units of s^{-1} to deal with phase stuff in radians

    temp_fourier_amp = np.zeros(n_harmonics_temp + 1)  # 1st component will remain 0 so mean of temperature is 0
    temp_fourier_phase = np.zeros(n_harmonics_temp)

    for n in range(1, n_harmonics_temp + 1):
        # convert lambda time lag from days to units of radians
        phase_delay_radians = phase_delay * 2 * n * np.pi / n_year_days
        lambda_cos = lambda_const * np.cos(0) + lambda_phase * np.cos(phase_delay_radians)
        lambda_sin = lambda_const * np.sin(0) + lambda_phase * np.sin(phase_delay_radians)

        temp_fourier_phase[n - 1] = np.arctan(
            (2 * np.pi * n * f * heat_capacity + sw_tan[n - 1] * lambda_cos - lambda_sin) / (
                    -2 * np.pi * n * f * heat_capacity * sw_tan[n - 1] + lambda_cos - sw_tan[n - 1] * lambda_sin))
        temp_fourier_amp[n] = sw_fourier_amp[n] * sw_cos[n - 1] / np.cos(
            temp_fourier_phase[n - 1]) / (
                                      (2 * np.pi * n * f * heat_capacity - lambda_sin) * np.tan(
                                  temp_fourier_phase[n - 1]) + lambda_cos)

        if n == 1 and lambda_sq != 0:
            if include_sw_phase:
                raise ValueError('Solution not possible with sw phase and lambda_sq.')
            # Get analytic solution when include squared term in energy budget
            # Assuming squared term dominated by 1st harmonic
            sw_tan[1] = (sw_fourier_amp[2] * sw_sin[1] -
                         0.5 * lambda_sq * temp_fourier_amp[1] ** 2 * np.sin(2 * temp_fourier_phase[0])) / (
                                sw_fourier_amp[2] * sw_cos[1] -
                                0.5 * lambda_sq * temp_fourier_amp[1] ** 2 * np.cos(2 * temp_fourier_phase[0]))
            if sw_fourier_amp[2] == 0:
                sw_fourier_amp[2] = 1       # this term will cancel out, when do sw_fourier_amp[2]*sw_cos[1]
                                            # so is just so don't divide by zero below - exact number not important
            sw_cos[1] -= 0.5 * lambda_sq * temp_fourier_amp[1] ** 2 * np.cos(2 * temp_fourier_phase[0]
                                                                             ) / sw_fourier_amp[2]
    temp_fourier = fourier.fourier_series(time, n_year_days, temp_fourier_amp, temp_fourier_phase)
    return temp_fourier, temp_fourier_amp, temp_fourier_phase, sw_fourier_amp, sw_fourier_phase


def gamma_linear_approx(time: np.ndarray, temp: np.ndarray,
                        lambda_const: np.ndarray, lambda_time_lag: Optional[np.ndarray] = None,
                        lambda_nl: Optional[Union[float, np.ndarray]] = None,
                        temp_anom_nl: Optional[np.ndarray] = None) -> np.ndarray:
    """
    This approximates $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$ as:

    $$\Gamma^{\\uparrow} \\approx \lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) +
    \sum_{j=2}^{j_{max}}\lambda_{nl_j}\\left(T'^{j}(t) - \overline{T'^j}\\right)$$

    where:

    * $T' = T(t) - \overline{T}$ is the surface temperature anomaly
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
        lambda_nl: `float [n_lambda_nl]`
            The constants $\lambda_{nl}$ used in the approximation. `[0]` is squared contribution, `[1]` is cubed, ...
        temp_anom_nl: `float [n_time]`</br>
            The value of $T'(t)$ to use in the calculation of non-linear part:
            $\sum_{j=2}^{j_{max}}\lambda_{nl_j}\\left(T'^{j}(t) - \overline{T'^j}\\right)$.
            If `None`, will compute from `temp`.

    Returns:
        `float [n_time]`</br>
            The approximation $\Gamma^{\\uparrow} \\approx \lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i)
            + \lambda_{sq}\\left(T(t) - \overline{T}\\right)^2$ with units of $Wm^{-2}$.
    """
    if lambda_nl is not None:
        #  deal with case when float given as lambda_nl
        if not hasattr(lambda_nl, "__len__"):
            lambda_nl = np.asarray([lambda_nl])
    n_lambda = len(lambda_const) - 1
    if lambda_time_lag is None:
        lambda_temp = np.sum(lambda_const[1:]) * temp
    else:
        # To apply phase shift, need spline so can compute temp anomaly at times outside range in `time`.
        temp_spline_fit = CubicSpline(np.append(time, time[-1] + 1), np.append(temp, temp[0]),
                                      bc_type='periodic')
        lambda_temp = np.zeros_like(temp)
        for i in range(n_lambda):
            lambda_temp += lambda_const[1 + i] * temp_spline_fit(time - lambda_time_lag[i])
    gamma_nl = np.zeros_like(temp)
    if lambda_nl is not None:
        if temp_anom_nl is None:
            temp_anom_nl = temp - np.mean(temp)
        n_lambda_nl = len(lambda_nl)
        for j in range(n_lambda_nl):
            gamma_nl += lambda_nl[j] * (temp_anom_nl ** (j + 2) - np.mean(temp_anom_nl ** (j + 2)))
    return lambda_const[0] + lambda_temp + gamma_nl


def swdn_from_temp_fourier(time: np.ndarray, temp_fourier_amp: np.ndarray, temp_fourier_phase: np.ndarray,
                           heat_capacity: float, lambda_const: np.ndarray,
                           lambda_time_lag: Optional[np.ndarray] = None,
                           lambda_nl: Optional[Union[float, np.ndarray]] = None,
                           day_seconds: float = 86400, single_harmonic_nl: bool = False) -> np.ndarray:
    """
    This inverts the linearized surface energy budget to return an approximation for downward shortwave radiation
    at the surface, $F(t)$, given a Fourier approximation for surface temperature,
    $T(t) = \\frac{T_0}{2} + \\sum_{n=1}^{N} T_n\\cos(2n\\pi ft - \\phi_n)$:

    $$
    F(t) \\approx C\\frac{\partial T}{\partial t} + \lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) +
    \sum_{j=2}^{j_{max}}\lambda_{nl_j}\\left(T'^{j}(t) - \overline{T'^j}\\right)
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
        lambda_nl: `float [n_lambda_nl]`
            The constants $\lambda_{nl}$ used in the approximation for $\Gamma^{\\uparrow}$.
            `[0]` is squared contribution, `[1]` is cubed, ...
        day_seconds: Duration of a day in seconds.
        single_harmonic_nl: If `True`, the $\lambda_{nl_j}T'^j$ terms in $\Gamma^{\\uparrow}$ will only
            use the first harmonic, not all harmonics.

    Returns:
        `float [n_time]`</br>
            Approximation for downward shortwave radiation at the surface.
            Units: $Wm^{-2}$.
    """
    if lambda_nl is not None:
        #  deal with case when float given as lambda_nl
        if not hasattr(lambda_nl, "__len__"):
            lambda_nl = np.asarray([lambda_nl])
    n_year_days = len(time)
    temp_fourier = fourier.fourier_series(time, n_year_days, temp_fourier_amp, temp_fourier_phase)
    if single_harmonic_nl:
        temp_anom_nl = fourier.fourier_series(time, n_year_days, [0, temp_fourier_amp[1]],
                                              [temp_fourier_phase[0]])
        gamma = gamma_linear_approx(time, temp_fourier, lambda_const,
                                    lambda_time_lag, lambda_nl, temp_anom_nl)
    else:
        gamma = gamma_linear_approx(time, temp_fourier, lambda_const,
                                    lambda_time_lag, lambda_nl)
    dtemp_dt = fourier.fourier_series_deriv(time, n_year_days, temp_fourier_amp, temp_fourier_phase, day_seconds)
    return heat_capacity * dtemp_dt + gamma


def get_temp_fourier(time: np.ndarray, swdn: np.ndarray, heat_capacity: float,
                     lambda_const: np.ndarray, lambda_time_lag: Optional[np.ndarray] = None,
                     lambda_nl: Optional[Union[float, np.ndarray]] = None, n_harmonics: int = 2,
                     include_sw_phase: bool = False, numerical: bool = False,
                     day_seconds: float = 86400,
                     single_harmonic_nl: bool = False, return_anomaly: bool = True) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seeks a fourier solution of the form $T(t) = \\frac{T_0}{2} + \\sum_{n=1}^{N} T_n\\cos(2n\\pi ft - \\phi_n)$
    to the linearized surface energy budget of the general form:

    $$
    C\\frac{\partial T}{\partial t} = F(t) - \lambda_0 - \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) -
    \sum_{j=2}^{j_{max}}\lambda_{nl_j}\\left(T'^{j}(t) - \overline{T'^j}\\right)
    $$

    where:

    * $T' = T(t) - \overline{T}$ is the surface temperature anomaly
    * $C$ is the heat capacity of the surface
    * $\overline{T} = T_0/2$ is the mean temperature.
    * $F(t) = \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi ft - \\varphi_n)$ is the Fourier representation
    of the downward shortwave radiation at the surface, $SW^{\downarrow}$.
    * $\lambda_0 + \sum_{i=1}^{N_{\lambda}}\lambda_i T(t-\Lambda_i) +
    \sum_{j=2}^{j_{max}}\lambda_{nl_j}\\left(T'^{j}(t) - \overline{T'^j}\\right)$ is the approximation for
    $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$

    The solution is exact if $\lambda_{nl_j}=0 \\forall j$ and has the form:

    * $T_0 = (F_0-2\lambda_0)/\sum_{i=1}^{N_{\lambda}}\lambda_i$
    * $T_n = \\frac{F_n \cos(\\varphi_n)\sqrt{1+\\tan^2\phi_n}}{
    (2\pi nfC - \sum_i \lambda_i \sin \Phi_{ni})\\tan\phi_n + \sum_i \lambda_i \cos \Phi_{ni}}$
    * $\\tan \phi_n = \\frac{2\pi nfC + \\tan \\varphi_n \sum_i \lambda_i \cos \Phi_{ni} -
    \sum_i \lambda_i \sin \Phi_{ni}}{-2\pi nfC \\tan \\varphi_n + \sum_i \lambda_i \cos \Phi_{ni} -
    \\tan \\varphi_n \sum_i \lambda_i \sin \Phi_{ni}}$
    * $\Phi_{ni} = 2\pi nf \Lambda_i$ (units of radians, whereas $\Lambda_i$ is in units of time - days).

    If $\lambda_{nl_j}\\neq 0$, an approximate numerical solution will be obtained, still of the Fourier form.

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
        lambda_nl: `float [n_lambda_nl]`
            The constants $\lambda_{nl}$ used in the approximation for $\Gamma^{\\uparrow}$.
            `[0]` is squared contribution, `[1]` is cubed, ...
        n_harmonics: Number of harmonics to use to fit fourier series for both $T(t)$ and $F(t)$, $N$.
        include_sw_phase: If `False`, will set all phase factors, $\\varphi_n=0$, in Fourier expansion of $F(t)$.</br>
            These phase factors are usually very small, and it makes the solution for $T(t)$ more simple if they
            are set to 0, hence the option.
        numerical: If `True`, will compute solution for $T(t)$ numerically using [`scipy.optimize.curve_fit`](
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) rather than using the
            analytic solution. Will always return numerical solution if `lambda_sq` $\\neq 0$.
        day_seconds: Duration of a day in seconds.
        single_harmonic_nl: If `True`, the $\lambda_{nl_j}T'^j$ terms in $\Gamma^{\\uparrow}$ will only
            use the first harmonic, not all harmonics.
        return_anomaly: If `True`, the first return variable, `temp_fourier` will be
            the temperature anomaly, i.e. it will not include $T_0$.

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
    if lambda_nl is not None:
        #  deal with case when float given as lambda_nl
        if not hasattr(lambda_nl, "__len__"):
            lambda_nl = np.asarray([lambda_nl])
    n_year_days = len(time)
    n_lambda = len(lambda_const) - 1
    if lambda_time_lag is None:
        lambda_time_lag = np.zeros(n_lambda)
    else:
        if len(lambda_time_lag) != n_lambda:
            raise ValueError(f'Size of lambda_time_lag should be {n_lambda} not {len(lambda_time_lag)}.')

    # Get fourier representation of SW radiation
    sw_fourier_amp, sw_fourier_phase = fourier.get_fourier_fit(time, swdn, n_year_days, n_harmonics)[1:]
    if not include_sw_phase:
        sw_fourier_phase = np.zeros(n_harmonics)
    sw_fourier = fourier.fourier_series(time, n_year_days, sw_fourier_amp, sw_fourier_phase)
    sw_tan = np.tan(sw_fourier_phase)
    sw_cos = np.cos(sw_fourier_phase)
    sw_sin = np.sin(sw_fourier_phase)
    f = 1 / (
            n_year_days * day_seconds)  # must have frequency in units of s^{-1} to deal with phase stuff in radians

    if numerical or (lambda_nl is not None and not single_harmonic_nl):
        if not numerical:
            warnings.warn('Analytic solution not possible with lambda_nl non zero and single_hamrmonic_nl=False')

        def fit_func(time_array, *args):
            fourier_amp_coef = np.asarray([args[i] for i in range(n_harmonics + 1)])
            fourier_phase_coef = np.asarray([args[i] for i in range(n_harmonics + 1, len(args))])
            return swdn_from_temp_fourier(time_array, fourier_amp_coef, fourier_phase_coef, heat_capacity, lambda_const,
                                          lambda_time_lag, lambda_nl, day_seconds, single_harmonic_nl)

        # force positive phase coefficient to match analytic solution
        bounds_lower = [-np.inf] * (n_harmonics + 1) + [0] * n_harmonics
        bounds_upper = [np.inf] * (n_harmonics + 1) + [2 * np.pi] * n_harmonics

        # Starting solution is 1 harmonic analytical solution
        p0 = np.zeros(2 * n_harmonics + 1)
        lambda_phase_const = lambda_time_lag * 2 * np.pi / n_year_days
        lambda_cos = np.sum(lambda_const[1:] * np.cos(lambda_phase_const))
        lambda_sin = np.sum(lambda_const[1:] * np.sin(lambda_phase_const))
        p0[0] = (sw_fourier_amp[0] - 2 * lambda_const[0]) / np.sum(lambda_const[1:])
        p0[n_harmonics + 1] = np.arctan(
            (2 * np.pi * f * heat_capacity + sw_tan[0] * lambda_cos - lambda_sin) / (
                    -2 * np.pi * f * heat_capacity * sw_tan[0] + lambda_cos - sw_tan[0] * lambda_sin))
        p0[1] = sw_fourier_amp[1] * sw_cos[0] / np.cos(p0[n_harmonics + 1]) / (
                (2 * np.pi * f * heat_capacity - lambda_sin) * np.tan(
            p0[n_harmonics + 1]) + lambda_cos)

        try:
            args_found = optimize.curve_fit(fit_func, time, sw_fourier, p0,
                                            bounds=(bounds_lower, bounds_upper))[0]
        except RuntimeError:
            warnings.warn('Hit Runtime Error, trying without bounds')
            args_found = optimize.curve_fit(fit_func, time, sw_fourier, p0)[0]
        temp_fourier_amp = args_found[:n_harmonics + 1]
        temp_fourier_phase = args_found[n_harmonics + 1:]
    else:
        temp_fourier_amp = np.zeros(n_harmonics + 1)
        temp_fourier_phase = np.zeros(n_harmonics)
        temp_fourier_amp[0] = (sw_fourier_amp[0] - 2 * lambda_const[0]) / np.sum(lambda_const[1:])

        for n in range(1, n_harmonics + 1):
            # convert lambda time lag from days to units of radians
            lambda_phase_const = lambda_time_lag * 2 * n * np.pi / n_year_days
            lambda_cos = np.sum(lambda_const[1:] * np.cos(lambda_phase_const))
            lambda_sin = np.sum(lambda_const[1:] * np.sin(lambda_phase_const))

            temp_fourier_phase[n - 1] = np.arctan(
                (2 * np.pi * n * f * heat_capacity + sw_tan[n - 1] * lambda_cos - lambda_sin) / (
                        -2 * np.pi * n * f * heat_capacity * sw_tan[n - 1] + lambda_cos - sw_tan[n - 1] * lambda_sin))
            temp_fourier_amp[n] = sw_fourier_amp[n] * sw_cos[n - 1] / np.cos(
                temp_fourier_phase[n - 1]) / (
                                          (2 * np.pi * n * f * heat_capacity - lambda_sin) * np.tan(
                                      temp_fourier_phase[n - 1]) + lambda_cos)
            if n == 1 and lambda_nl is not None and single_harmonic_nl:
                if len(lambda_nl) > 1:
                    raise ValueError(f'Analytic solution only possible with 1 not {len(lambda_nl)} non-linear terms')
                # Get analytic solution when include squared term in energy budget
                # Assuming squared term dominated by 1st harmonic
                sw_tan[1] = (sw_fourier_amp[2] * sw_sin[1] -
                             0.5 * lambda_nl[0] * temp_fourier_amp[1] ** 2 * np.sin(2 * temp_fourier_phase[0])) / (
                                    sw_fourier_amp[2] * sw_cos[1] -
                                    0.5 * lambda_nl[0] * temp_fourier_amp[1] ** 2 * np.cos(2 * temp_fourier_phase[0]))
                sw_cos[1] -= 0.5 * lambda_nl[0] * temp_fourier_amp[1] ** 2 * np.cos(2 * temp_fourier_phase[0]
                                                                                    ) / sw_fourier_amp[2]
    if return_anomaly:
        temp_fourier = fourier.fourier_series(time, n_year_days, np.append([0], temp_fourier_amp[1:]),
                                              temp_fourier_phase)
    else:
        temp_fourier = fourier.fourier_series(time, n_year_days, temp_fourier_amp, temp_fourier_phase)
    return temp_fourier, temp_fourier_amp, temp_fourier_phase, sw_fourier_amp, sw_fourier_phase
