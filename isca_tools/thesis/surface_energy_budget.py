import numpy as np
from ..utils import fourier, numerical
from scipy import optimize
from scipy.interpolate import CubicSpline
import warnings
from typing import Optional, Tuple, Union, Literal
import xarray as xr


def get_temp_fourier_numerical(time: np.ndarray, temp_anom: np.ndarray, gamma: np.ndarray,
                               swdn_sfc: np.ndarray, heat_capacity: float,
                               n_harmonics_sw: int = 2, n_harmonics_temp: Optional[int] = None,
                               deg_gamma_fit: int = 8, phase_gamma_fit: bool = True,
                               resample: bool = False,
                               gamma_fourier_term: bool = False,
                               include_sw_phase: bool = False,
                               day_seconds: float = 86400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This uses [`scipy.optimize.curve_fit`](
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) to numerically seek
    a fourier solution of the form $T'(t) = \\sum_{n=1}^{N} T_n\\cos(2n\\pi t/\mathcal{T} - \\phi_n)$
    to the linearized surface energy budget of the general form:

    $$
    \\begin{align}
    \\begin{split}
    C\\frac{\partial T'}{\partial t} = &SW^{\\downarrow}(t) - \lambda_0 -
    \\frac{1}{2}\lambda_{phase}(T'(t-\mathcal{T}/4) - T'(t+\mathcal{T}/4)) -  \\\\
    &\\sum_{j=1}^{N_{\Gamma}}\lambda_j T'^{j}(t) - \\sum_{n=2}^N (\Lambda_{n, cos}\\cos(2n\\pi t/\mathcal{T}) +
    \Lambda_{n, sin}\\sin(2n\\pi t/\mathcal{T}))
    \\end{split}
    \\end{align}
    $$

    where:

    * $T' = T(t) - \overline{T}$ is the surface temperature anomaly
    * $C$ is the heat capacity of the surface
    * $\overline{T} = T_0/2$ is the mean temperature.
    * $SW^{\downarrow}$ is the downward shortwave radiation at the surface, $SW^{\downarrow}$.
    * $\lambda_0 + \\frac{1}{2}\lambda_{phase}(T'(t-\mathcal{T}/4) - T'(t+\mathcal{T}/4)) +
    \\sum_{j=1}^{N_{\Gamma}}\lambda_j T'^{j}(t) +$</br>
    $\\sum_{n=2}^N (\Lambda_{n, cos}\\cos(2n\\pi t/\mathcal{T}) + \Lambda_{n, sin}\\sin(2n\\pi t/\mathcal{T}))$
    is the approximation for $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$.
    * $\mathcal{T}$ is the period i.e. one year.


    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`).
        temp_anom: `float [n_time]`</br>
            Surface temperature anomaly, $T'(t)$, for each day in `time`. Used for approximating
            $\Gamma^{\\uparrow}$.</br>
            Assumes periodic so temp[0] is the temperature at time step immediately after `time[-1]`.
        gamma: `float [n_time]`</br>
            Simulated value of $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$
            where $LW$ is longwave, $LH$ is latent heat and $SH$ is sensible heat.</br>
            Units: $Wm^{-2}$.
        swdn_sfc: `float [n_time]`</br>
            Downward shortwave radiation at the surface.</br>
            Units: $Wm^{-2}$.
        heat_capacity: $C$, the heat capacity of the surface in units of $JK^{-1}m^{-2}$.</br>
            Obtained from mixed layer depth of ocean using
            [`get_heat_capacity`](../utils/radiation.md#isca_tools.utils.radiation.get_heat_capacity).
        n_harmonics_sw: Number of harmonics to use to fit fourier series for $SW^{\\downarrow}$.
            Cannot exceed `n_harmonics_temp` as extra harmonics would not be used.</n>
            Set to None, to use no approximation for $SW^{\\downarrow}$ - but we weary with comparing to analytic
            solution in this case.
        n_harmonics_temp: Number, $N$, of harmonics in fourier solution of temperature anomaly. If not given, will
            set to `n_harmonics_sw`.
        deg_gamma_fit: Power, $N_{\Gamma}$, to go up to in polyomial approximation of $\Gamma^{\\uparrow}$ seeked.
        phase_gamma_fit: If `False` will set $\lambda_{phase}=0$.
            Otherwise, will use [`polyfit_phase`](../utils/numerical.md#isca_tools.utils.numerical.polyfit_phase)
            to estimate it.
        resample: If `True`, will use [`resample_data`](../utils/numerical.md#isca_tools.utils.numerical.resample_data)
            to make data evenly spaced in $x$ before calling `np.polyfit`, when obtaining $\Gamma$ coefficients.
        gamma_fourier_term: Whether to fit the Fourier contribution
            $\\sum_{n=2}^N (\Lambda_{n, cos}\\cos(2n\\pi t/\mathcal{T}) + \Lambda_{n, sin}\\sin(2n\\pi t/\mathcal{T}))$
             to $\Gamma^{\\uparrow}$ with `fourier_harmonics=np.arange(2, n_harmonics+1)` in
            [`polyfit_phase`](../utils/numerical.md#isca_tools.utils.numerical.polyfit_phase). Idea behind this
            is to account for contribution of $\Gamma^{\\uparrow}$ that is not temperature dependent.
        include_sw_phase: If `False`, will set all phase factors, $\\varphi_n=0$, in Fourier expansion of
            $SW^{\\downarrow}$.</br>
            These phase factors are usually very small, and it makes the analytic solution for $T'(t)$ more simple
            if they are set to 0, hence the option. Only use if `n_harmonics_sw` not `None`.
        day_seconds: Duration of a day in seconds.

    Returns:
        temp_fourier `float [n_time]`</br>
            The Fourier series solution that was found for surface temperature anomaly.</br>
            Units: $K$.
        temp_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients for surface temperature: $T_n$.
        temp_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients for surface temperature: $\phi_n$.
    """
    if n_harmonics_temp is None:
        n_harmonics_temp = n_harmonics_sw
    if gamma_fourier_term and phase_gamma_fit:
        gamma_approx_coefs, gamma_fourier_term_coefs_amp, gamma_fourier_term_coefs_phase = \
            numerical.polyfit_phase(temp_anom, gamma, deg_gamma_fit, resample=resample,
                                    include_phase=phase_gamma_fit, fourier_harmonics=np.arange(2, n_harmonics_temp + 1))
    else:
        gamma_approx_coefs = numerical.polyfit_phase(temp_anom, gamma, deg_gamma_fit, resample=resample,
                                                     include_phase=phase_gamma_fit)
        gamma_fourier_term_coefs_amp = None
        gamma_fourier_term_coefs_phase = None

    def fit_func(time_array, *args):
        # first n_harmonic values in args are the temperature amplitude coefficients (excluding T_0)
        # last n_harmonic values in args are the temperature phase coefficients
        fourier_amp_coef = np.asarray([args[i] for i in range(n_harmonics_temp)])
        # make first coefficient 0 so gives anomaly
        fourier_amp_coef = np.append([0], fourier_amp_coef)
        fourier_phase_coef = np.asarray([args[i] for i in range(n_harmonics_temp, len(args))])
        temp_anom_fourier = fourier.fourier_series(time_array, fourier_amp_coef, fourier_phase_coef)
        dtemp_dt_fourier = fourier.fourier_series_deriv(time_array, fourier_amp_coef,
                                                        fourier_phase_coef, day_seconds)
        if phase_gamma_fit:
            gamma_approx = numerical.polyval_phase(gamma_approx_coefs, temp_anom_fourier,
                                                   coefs_fourier_amp=gamma_fourier_term_coefs_amp,
                                                   coefs_fourier_phase=gamma_fourier_term_coefs_phase)
        else:
            gamma_approx = np.polyval(gamma_approx_coefs, temp_anom_fourier)
        return heat_capacity * dtemp_dt_fourier + gamma_approx

    # Starting guess is linear 1 harmonic linear analytical solution given gamma=lambda_const_guess*temp
    # so only 1st harmonic coefficients of amplitude and phase are needed, rest are set to zero
    p0 = np.zeros(2 * n_harmonics_temp)
    f = 1 / (time.size * day_seconds)
    p0[n_harmonics_temp] = np.arctan((2 * np.pi * f * heat_capacity) / gamma_approx_coefs[-2])
    if n_harmonics_sw is None:
        sw_fourier_amp = fourier.get_fourier_fit(time, swdn_sfc, 1)[1]
        # find temperature solution which minimises error to full insolation, no fourier approx
        sw_fourier_fit = swdn_sfc
    else:
        if include_sw_phase:
            sw_fourier_fit, sw_fourier_amp = fourier.get_fourier_fit(time, swdn_sfc, n_harmonics_sw)[:2]
        else:
            sw_fourier_amp, sw_fourier_phase = fourier.get_fourier_fit(time, swdn_sfc, n_harmonics_sw)[1:]
            sw_fourier_fit = fourier.fourier_series(time, sw_fourier_amp, sw_fourier_phase * 0)
    p0[0] = sw_fourier_amp[1] / np.cos(p0[n_harmonics_temp]) / (
            (2 * np.pi * f * heat_capacity) * np.tan(p0[n_harmonics_temp]) + gamma_approx_coefs[-2])
    args_found = optimize.curve_fit(fit_func, time, sw_fourier_fit, p0)[0]
    temp_fourier_amp = np.append([0], args_found[:n_harmonics_temp])
    temp_fourier_phase = args_found[n_harmonics_temp:]
    return fourier.fourier_series(time, temp_fourier_amp, temp_fourier_phase), temp_fourier_amp, temp_fourier_phase


def get_temp_fourier_analytic(time: np.ndarray, swdn_sfc: np.ndarray, heat_capacity: float,
                              lambda_const: float, lambda_phase: float = 0,
                              lambda_sq: float = 0, lambda_cos: float = 0, lambda_sin: float = 0,
                              n_harmonics_sw: int = 2,
                              n_harmonics_temp: Optional[int] = None,
                              include_sw_phase: bool = False,
                              day_seconds: float = 86400, pad_coefs_phase: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seeks a fourier solution of the form $T'(t) = \\sum_{n=1}^{N} T_n\\cos(2n\\pi t/\mathcal{T} - \\phi_n)$
    to the surface energy budget of the general form:

    $$
    \\begin{align}
    \\begin{split}
    C\\frac{\partial T'}{\partial t} = &F(t) - \lambda_0 -
    \\frac{1}{2}\lambda_{phase}(T'(t-\mathcal{T}/4) - T'(t+\mathcal{T}/4)) -  \\\\
    &\lambda T'^(t) - \lambda_{sq} T'^{2}(t) - \Lambda_{cos}\\cos(4\\pi t/\mathcal{T}) -
    \Lambda_{sin}\\sin(4\\pi t/\mathcal{T})
    \\end{split}
    \\end{align}
    $$

    where:

    * $T' = T(t) - \overline{T}$ is the surface temperature anomaly
    * $C$ is the heat capacity of the surface
    * $\overline{T} = T_0/2$ is the mean temperature.
    * $F(t) = \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi t/\mathcal{T} - \\varphi_n)$ is the Fourier representation
    of the downward shortwave radiation at the surface, $SW^{\downarrow}$.
    * $\lambda_0 + \lambda T' - \lambda_{sq} T'^{2} +
    \\frac{1}{2}\lambda_{phase}(T'(t-\mathcal{T}/4) - T'(t+\mathcal{T}/4)) +$</br>
    $\Lambda_{cos}\\cos(4\\pi t/\mathcal{T}) + \Lambda_{sin}\\sin(4\\pi t/\mathcal{T})$
    is the approximation for $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$.
    * $\mathcal{T}$ is the period i.e. one year.

    The solution is exact if $\lambda_{sq}=0$ and has the form:

    * $T_0 = (F_0-2\lambda_0)/\sum_{i=1}^{N_{\lambda}}\lambda_i$
    * $T_n = \\frac{F_n \cos(\\varphi_n)\sqrt{1+\\tan^2\phi_n}}{
    (2\pi nfC - \sum_i \lambda_i \sin \Phi_{ni})\\tan\phi_n + \sum_i \lambda_i \cos \Phi_{ni}}$
    * $\\tan \phi_n = \\frac{2\pi nfC + \\tan \\varphi_n \sum_i \lambda_i \cos \Phi_{ni} -
    \sum_i \lambda_i \sin \Phi_{ni}}{-2\pi nfC \\tan \\varphi_n + \sum_i \lambda_i \cos \Phi_{ni} -
    \\tan \\varphi_n \sum_i \lambda_i \sin \Phi_{ni}}$
    * $i=1, 2$ with $\lambda_1=\lambda$ and $\lambda_2=\lambda_{phase}$.
    * $\Phi_{n1}=0$ and $\Phi_{n2} = n\pi/2$ (from $2n\pi / \mathcal{T} \\times \mathcal{T}/4$).

    If $\lambda_{sq}\\neq 0$, an approximate analytical solution will be obtained, assuming
    $T'^2(t) \\approx (T_1\\cos(2\\pi ft - \\phi_1))^2$.

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`).
        swdn_sfc: `float [n_time]`</br>
            Downward shortwave radiation at the surface, $SW^{\\downarrow}$. I.e. `swdn_sfc` output by Isca.
            Units: $Wm^{-2}$.
        heat_capacity: $C$, the heat capacity of the surface in units of $JK^{-1}m^{-2}$.</br>
            Obtained from mixed layer depth of ocean using
            [`get_heat_capacity`](../utils/radiation.md#isca_tools.utils.radiation.get_heat_capacity).
        lambda_const: The constant $\lambda$ used in the approximation for
            $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$.</br>
        lambda_phase: The constants $\lambda_{phase}$ used in the approximation for $\Gamma^{\\uparrow}$.
        lambda_sq: The constant $\lambda_{sq}$ used in the approximation for $\Gamma^{\\uparrow}$.
        lambda_cos: The constant $\Lambda_{cos}$ used in the approximation for $\Gamma^{\\uparrow}$.
        lambda_sin: The constant $\Lambda_{sin}$ used in the approximation for $\Gamma^{\\uparrow}$.
        n_harmonics_sw: Number of harmonics to use to fit fourier series for $SW^{\\downarrow}$.
            Cannot exceed `n_harmonics_temp` as extra harmonics would not be used.
        n_harmonics_temp: Number, $N$, of harmonics in fourier solution of temperature anomaly. If not given, will
            set to `n_harmonics_sw`.
        include_sw_phase: If `False`, will set all phase factors, $\\varphi_n=0$, in Fourier expansion of
            $SW^{\\downarrow}$.</br>
            These phase factors are usually very small, and it makes the solution for $T'(t)$ more simple if they
            are set to 0, hence the option.
        day_seconds: Duration of a day in seconds.
        pad_coefs_phase: If `True`, will set `temp_fourier_phase` and `sw_fourier_phase`
            to length `n_harmonics+1` to match `_fourier_amp`, with the first value as zero.
            Otherwise, it will be size `n_harmonics`.

    Returns:
        temp_fourier: `float [n_time]`</br>
            The Fourier series solution that was found for surface temperature, $T$.
        temp_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients for surface temperature: $T_n$.
        temp_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients for surface temperature: $\phi_n$.
            If `pad_coefs_phase`, it will be of the length `n_harmonics+1`, with the first value as zero.
        sw_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients for shortwave radiation, $SW^{\\downarrow}$: $F_n$.
        sw_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients for shortwave radiation, $SW^{\\downarrow}$: $\\varphi_n$.
            If `pad_coefs_phase`, it will be of the length `n_harmonics+1`, with the first value as zero.
    """
    n_year_days = len(time)
    if n_harmonics_temp is None:
        n_harmonics_temp = n_harmonics_sw

    if n_harmonics_temp == 1 and lambda_sq != 0:
        raise ValueError('Cannot solve for non-zero lambda_sq with single harmonic - '
                         'use get_temp_fourier_numerical instead')

    # Get fourier representation of SW radiation
    if n_harmonics_sw > n_harmonics_temp:
        raise ValueError('Cannot have more harmonics for swdn_sfc than have for temperature')
    sw_fourier_amp = np.zeros(n_harmonics_temp + 1)
    sw_fourier_phase = np.zeros(n_harmonics_temp)
    sw_fourier_amp[:n_harmonics_sw + 1], sw_fourier_phase[:n_harmonics_sw] = \
        fourier.get_fourier_fit(time, swdn_sfc, n_harmonics_sw)[1:]
    if not include_sw_phase:
        sw_fourier_phase = np.zeros(n_harmonics_temp)
    sw_tan = np.tan(sw_fourier_phase)
    sw_cos = np.cos(sw_fourier_phase)
    sw_sin = np.sin(sw_fourier_phase)
    if n_harmonics_temp >= 2:
        # For 2nd harmonic, have modification from cos and sin factors
        sw_cos[1] = (sw_fourier_amp[2] * sw_cos[1] - lambda_cos) / sw_fourier_amp[2]
        sw_sin[1] = (sw_fourier_amp[2] * sw_sin[1] - lambda_sin) / sw_fourier_amp[2]
        sw_tan[1] = sw_sin[1] / sw_cos[1]
    f = 1 / (n_year_days * day_seconds)  # must have frequency in units of s^{-1} to deal with phase stuff in radians

    temp_fourier_amp = np.zeros(n_harmonics_temp + 1)  # 1st component will remain 0 so mean of temperature is 0
    temp_fourier_phase = np.zeros(n_harmonics_temp)

    # 0.25 multiplier accounts for the phase delay i.e. shift of 90 days in 360 day year - same as in polyfit_phase
    phase_prefactor = 0.25
    for n in range(1, n_harmonics_temp + 1):
        lambda_term_cos = lambda_const
        lambda_term_sin = 0.5 * lambda_phase * (np.sin(phase_prefactor * 2 * n * np.pi) -
                                                np.sin(-phase_prefactor * 2 * n * np.pi))

        temp_fourier_phase[n - 1] = np.arctan(
            (2 * np.pi * n * f * heat_capacity + sw_tan[n - 1] * lambda_term_cos - lambda_term_sin) / (
                    -2 * np.pi * n * f * heat_capacity * sw_tan[n - 1] + lambda_term_cos - sw_tan[
                n - 1] * lambda_term_sin))
        temp_fourier_amp[n] = sw_fourier_amp[n] * sw_cos[n - 1] / np.cos(
            temp_fourier_phase[n - 1]) / (
                                      (2 * np.pi * n * f * heat_capacity - lambda_term_sin) * np.tan(
                                  temp_fourier_phase[n - 1]) + lambda_term_cos)

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
                sw_fourier_amp[2] = 1  # this term will cancel out, when do sw_fourier_amp[2]*sw_cos[1]
                # so is just so don't divide by zero below - exact number not important
            sw_cos[1] -= 0.5 * lambda_sq * temp_fourier_amp[1] ** 2 * np.cos(2 * temp_fourier_phase[0]
                                                                             ) / sw_fourier_amp[2]
    temp_fourier = fourier.fourier_series(time, temp_fourier_amp, temp_fourier_phase)
    if pad_coefs_phase:
        temp_fourier_phase = np.hstack((np.zeros(1), temp_fourier_phase))
        sw_fourier_phase = np.hstack((np.zeros(1), sw_fourier_phase))
    return temp_fourier, temp_fourier_amp, temp_fourier_phase, sw_fourier_amp, sw_fourier_phase


def get_param_dimensionless(var: Union[float, np.ndarray, xr.DataArray],
                            heat_capacity: Optional[Union[float, np.ndarray, xr.DataArray]]=None,
                            n_year_days: Optional[int]=None,
                            sw_fourier_amp1: Optional[Union[float, np.ndarray, xr.DataArray]]=None,
                            sw_fourier_amp2: Optional[Union[float, np.ndarray, xr.DataArray]]=None,
                            lambda_const: Optional[Union[float, np.ndarray, xr.DataArray]]=None,
                            day_seconds: float = 86400
                            ) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Returns dimensionless versions of empirical fitting parameter, without changing the sign.
    There are three possibilities for how `var` is made dimensionless:

    * If provide `heat_capacity`, $C$, and `n_year_days`, $\mathcal{T}$, then will assume `var` is $\lambda$ or
    $\lambda_{phase}$. Will return $\lambda' = \\frac{\lambda}{2\pi fC}$ where $f=1/\mathcal{T}$.
    * If provide `sw_fourier_amp1`, $F_1$, `sw_fourier_amp2`, $F_2$, and `lambda_const`, $\lambda$, then will
    assume `var` is $\lambda_{sq}$. Will return $\lambda_{sq}' = \\frac{\lambda_{sq}F_1^2}{2\lambda^2F_2}$.
    * If provide `sw_fourier_amp2`, $F_2$, only then will assume `var` is $\Lambda_{cos}$ or $\Lambda_{sin}$:
    $\Lambda_{cos}' = \Lambda_{cos}/F_2$.

    Args:
        var: Parameter to make dimensionless.
        heat_capacity: $C$, the heat capacity of the surface in units of $JK^{-1}m^{-2}$.</br>
            Obtained from mixed layer depth of ocean using
            [`get_heat_capacity`](../utils/radiation.md#isca_tools.utils.radiation.get_heat_capacity).
        n_year_days: Number of days in the year.
        sw_fourier_amp1: The first harmonic amplitude Fourier coefficient for shortwave radiation, $F_1$.
        sw_fourier_amp2: The second harmonic amplitude Fourier coefficients for shortwave radiation, $F_2$.
        lambda_const: The linear constant $\lambda$ used in the approximation for
            $\Gamma^{\\uparrow}. In normal form with units of Wm$^{-2}$K$^{-1}$.
        day_seconds: Duration of a day in seconds.

    Returns:
        var_dim: Dimensionless version of `var`.
    """
    # May be issue here with var changing sign, but don't think so as only negative is sw_fourier_amp1 and square that
    # But in southern hemisphere, may be different with sw_fourier_amp2
    if (heat_capacity is not None) and (n_year_days is not None):
        # lambda_phase or lambda_const
        f = 1/(n_year_days * day_seconds)
        var = var / (2*np.pi*f*heat_capacity)
    elif (sw_fourier_amp1 is not None) and (sw_fourier_amp2 is not None) and (lambda_const is not None):
        # lambda_sq
        var = var * sw_fourier_amp1**2 / (2 * lambda_const ** 2 * np.abs(sw_fourier_amp2))
    elif sw_fourier_amp2 is not None:
        # lambda_cos and lambda_sin
        var = var/np.abs(sw_fourier_amp2)
    else:
        raise ValueError("Combination of parameters provided not correct for any parameter")
    return var


def get_temp_fourier_analytic2(time: np.ndarray, swdn_sfc: np.ndarray, heat_capacity: float,
                               lambda_const: float, lambda_phase: float = 0,
                               lambda_sq: float = 0, lambda_cos: float = 0, lambda_sin: float = 0,
                               lambda_0: Optional[float] = None,
                               n_harmonics: Literal[1, 2] = 2,
                               day_seconds: float = 86400, pad_coefs_phase: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This is the same as `get_temp_fourier_analytic` but is constrained to only work with `n_harmonics=1` and
    `n_harmonics=2` and enforces `include_sw_phase=False`.
    Also, it is coded in a way to reflect the algebra more closely.

    Seeks a fourier solution of the form $T(t) = \\frac{T_0}{2} + \\sum_{n=1}^{2} T_n\\cos(2n\\pi t/\mathcal{T} - \\phi_n)$
    to the surface energy budget of the general form:

    $$
    \\begin{align}
    \\begin{split}
    C\\frac{\partial T}{\partial t} = &F(t) - \lambda_0 -
    \\frac{1}{2}\lambda_{phase}(T(t-\mathcal{T}/4) - T(t+\mathcal{T}/4)) -  \\\\
    &\lambda T(t) - \lambda_{sq} T'^{2}(t) - \Lambda_{cos}\\cos(4\\pi t/\mathcal{T}) -
    \Lambda_{sin}\\sin(4\\pi t/\mathcal{T})
    \\end{split}
    \\end{align}
    $$

    where:

    * $T' = T(t) - \overline{T}$ is the surface temperature anomaly
    * $C$ is the heat capacity of the surface
    * $\overline{T} = T_0/2$ is the mean temperature.
    * $F(t) = \\frac{F_0}{2} + \\sum_{n=1}^{N} F_n\\cos(2n\\pi t/\mathcal{T})$ is the Fourier representation
    of the downward shortwave radiation at the surface, $SW^{\downarrow}$.
    * $\lambda_0 + \lambda T - \lambda_{sq} T'^{2} +
    \\frac{1}{2}\lambda_{phase}(T(t-\mathcal{T}/4) - T(t+\mathcal{T}/4)) +$</br>
    $\Lambda_{cos}\\cos(4\\pi t/\mathcal{T}) + \Lambda_{sin}\\sin(4\\pi t/\mathcal{T})$
    is the approximation for $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$.
    * $\mathcal{T}$ is the period i.e. one year.

    The solution is exact if $\lambda_{sq}=0$, and otherwise approximate assuming first harmonic dominates $T'^2$
    such that $T'^2(t) \\approx (T_1\\cos(2\\pi ft - \\phi_1))^2 + T_2^2/2$:

    * $T_0 = (F_0-2\lambda_0)/\lambda - \\frac{\lambda_{sq}}{\lambda}(T_1^2 + T_2^2)$
    * $\\tan \phi_1 = x(1 - \\frac{\lambda_{phase}}{2\pi fC})$ where $x=\\frac{2\pi fC}{\lambda}$ and $f=1/\mathcal{T}$.
    * $T_1 = \\frac{F_1}{\lambda}\\frac{1}{1+\\tan^2\phi_1}$
    * $\\tan \phi_2 = 2x \\frac{1-\\frac{1}{2x}\\frac{\\alpha_2}{1-\\alpha_1}}{1+2x\\frac{\\alpha_2}{1-\\alpha_1}}$
    * $T_2 = \\frac{F_2(1-\Lambda_{cos}')}{\lambda}\\frac{1+\\tan^2\phi_2}{1+2x\\tan\phi_2}(1-\\alpha_1)$

    where we use the following dimensionless parameters:

    * $\Lambda_{cos}' = \Lambda_{cos}/F_2$
    * $\Lambda_{sin}' = \Lambda_{sin}/F_2$
    * $\lambda_{sq}' = \\frac{\lambda_{sq}F_1^2}{2\lambda^2F_2}$
    * $\\alpha_1 = \\frac{\lambda_{sq}'}{1-\Lambda_{cos}'}\\frac{1-\\tan^2(\phi_1)}{(1+\\tan^2(\phi_1))^2}$
    * $\\alpha_2 = \\frac{\Lambda_{sin}'}{1-\Lambda_{cos}'} + \\frac{2\lambda_{sq}'}{1-\Lambda_{cos}'}\\frac{\\tan(\phi_1)}{(1+\\tan^2(\phi_1))^2}$

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`).
        swdn_sfc: `float [n_time]`</br>
            Downward shortwave radiation at the surface, $SW^{\\downarrow}$. I.e. `swdn_sfc` output by Isca.
            Units: $Wm^{-2}$.
        heat_capacity: $C$, the heat capacity of the surface in units of $JK^{-1}m^{-2}$.</br>
            Obtained from mixed layer depth of ocean using
            [`get_heat_capacity`](../utils/radiation.md#isca_tools.utils.radiation.get_heat_capacity).
        lambda_const: The linear constant $\lambda$ used in the approximation for
            $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$.</br>
        lambda_phase: The constants $\lambda_{phase}$ used in the approximation for $\Gamma^{\\uparrow}$.
        lambda_sq: The constant $\lambda_{sq}$ used in the approximation for $\Gamma^{\\uparrow}$.
        lambda_cos: The constant $\Lambda_{cos}$ used in the approximation for $\Gamma^{\\uparrow}$.
        lambda_sin: The constant $\Lambda_{sin}$ used in the approximation for $\Gamma^{\\uparrow}$.
        lambda_0: The constant $\lambda_0$ used in the approximation for $\Gamma^{\\uparrow}$.
            Leave as `None` to set $T_0=0$ i.e., return the anomaly $T_s-\overline{T}_s$.
        n_harmonics: Number of harmonics to use to fit fourier series for $SW^{\\downarrow}$.
            Cannot exceed `n_harmonics_temp` as extra harmonics would not be used.
        day_seconds: Duration of a day in seconds.
        pad_coefs_phase: If `True`, will set `temp_fourier_phase` and `sw_fourier_phase`
            to length `n_harmonics+1` to match `_fourier_amp`, with the first value as zero.
            Otherwise, it will be size `n_harmonics`.

    Returns:
        temp_fourier: `float [n_time]`</br>
            The Fourier series solution that was found for surface temperature, $T$.
        temp_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients for surface temperature: $T_n$.
        temp_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients for surface temperature: $\phi_n$.
            If `pad_coefs_phase`, it will be of the length `n_harmonics+1`, with the first value as zero.
        sw_fourier_amp: `float [n_harmonics+1]`</br>
            The amplitude Fourier coefficients for shortwave radiation, $SW^{\\downarrow}$: $F_n$.
        sw_fourier_phase: `float [n_harmonics]`</br>
            The phase Fourier coefficients for shortwave radiation, $SW^{\\downarrow}$: $\\varphi_n$.
            If `pad_coefs_phase`, it will be of the length `n_harmonics+1`, with the first value as zero.
    """
    n_year_days = len(time)
    f = 1 / (n_year_days * day_seconds)  # must have frequency in units of s^{-1} to deal with phase stuff in radians

    if n_harmonics == 1 and lambda_sq != 0:
        raise ValueError('Cannot solve for non-zero lambda_sq with single harmonic - '
                         'use get_temp_fourier_numerical instead')
    # Get fourier representation of SW radiation. Note I force phase coefs to zero
    sw_fourier_amp = fourier.get_fourier_fit(time, swdn_sfc, n_harmonics)[1]

    temp_fourier_amp = np.zeros(n_harmonics + 1)  # 1st component will remain 0 so mean of temperature is 0
    temp_fourier_phase = np.zeros(n_harmonics + 1)

    # 1st Harmonic
    heat_cap_eff = heat_capacity * (1 - lambda_phase / (2 * np.pi * f * heat_capacity))
    x = 2 * np.pi * f * heat_capacity / lambda_const
    tan_phase1 = 2 * np.pi * f * heat_cap_eff / lambda_const
    temp_fourier_phase[1] = np.arctan(tan_phase1)
    temp_fourier_amp[1] = sw_fourier_amp[1] / lambda_const / np.sqrt(1 + tan_phase1 ** 2)

    # 2nd Harmonic - not exact if lambda_sq!=0, as approx T^2 given by first harmonic squared only
    if n_harmonics == 2:
        # Put empirical params in dimensionless form
        lambda_cos_dim = get_param_dimensionless(lambda_cos, sw_fourier_amp2=sw_fourier_amp[2]) * np.sign(sw_fourier_amp[2])
        lambda_sin_dim = get_param_dimensionless(lambda_sin, sw_fourier_amp2=sw_fourier_amp[2]) * np.sign(sw_fourier_amp[2])
        lambda_sq_dim = get_param_dimensionless(lambda_sq, sw_fourier_amp1=sw_fourier_amp[1],
                                                sw_fourier_amp2=sw_fourier_amp[2], lambda_const=lambda_const) * np.sign(sw_fourier_amp[2])

        # Combine to form other dimensionless factors
        alpha_1 = lambda_sq_dim / (1 - lambda_cos_dim) * (1 - tan_phase1 ** 2) / (1 + tan_phase1 ** 2) ** 2
        alpha_2 = lambda_sin_dim / (1 - lambda_cos_dim) + 2 * lambda_sq_dim / (1 - lambda_cos_dim) * tan_phase1 / (
                1 + tan_phase1 ** 2) ** 2
        phase_mod_factor = (1 - 1 / 2 / x * alpha_2 / (1 - alpha_1)) / (1 + 2 * x * (alpha_2 / (1 - alpha_1)))
        amp_mod_factor = (1 - lambda_cos_dim) * (1 - alpha_1)

        # Combine usual phase and amp factors with modification factors
        tan_phase2 = 2 * x * phase_mod_factor
        temp_fourier_phase[2] = np.arctan(tan_phase2)
        temp_fourier_amp[2] = sw_fourier_amp[2] / lambda_const * np.sqrt(1 + tan_phase2 ** 2) / (
                    1 + 2 * x * tan_phase2) * amp_mod_factor

        # sw_amp2_eff = sw_fourier_amp[2]-lambda_cos
        # lambda_sin_eff = lambda_sin/sw_amp2_eff
        # lambda_sq_eff = lambda_sq/2 * temp_fourier_amp[1]**2 / (2*sw_amp2_eff)
        # alpha_1 = lambda_sq_eff * (1-x**2)/(1+x**2)
        # alpha_2 = lambda_sin_eff + 2 * lambda_sq_eff * x/(1+x**2)
        # phase_mod_factor = (1-lambda_const/(4*np.pi*f*heat_capacity)*(alpha_2/(1-alpha_1)))/(
        #         1+4*np.pi*f*heat_capacity/lambda_const*(alpha_2/(1-alpha_1)))
        # x2 = phase_mod_factor * 4*np.pi*f*heat_capacity/lambda_const        # tan of phase_2
        # temp_fourier_phase[2] = np.arctan(x2)
        # temp_fourier_amp[2] = sw_amp2_eff * np.sqrt(1+x2**2) / (lambda_const + 4*np.pi*f*heat_capacity*x2) * (1-alpha_1)

    # 0th Harmonic
    if lambda_0 is not None:
        temp_fourier_amp[0] = (sw_fourier_amp[0] - 2 * lambda_0) / lambda_const
        if n_harmonics == 2:
            # If include squared term, then get an extra contribution to T_0
            temp_fourier_amp[0] -= lambda_sq / lambda_const * (temp_fourier_amp[1] ** 2 + temp_fourier_amp[2] ** 2)

    temp_fourier = fourier.fourier_series(time, temp_fourier_amp, temp_fourier_phase, pad_coefs_phase=True)
    if not pad_coefs_phase:
        temp_fourier_phase = temp_fourier_phase[1:]
    sw_fourier_phase = np.zeros_like(temp_fourier_phase)
    return temp_fourier, temp_fourier_amp, temp_fourier_phase, sw_fourier_amp, sw_fourier_phase


def get_temp_extrema_analytic(sw_fourier_amp1: Union[float, np.ndarray], heat_capacity: float,
                              lambda_const: Union[float, np.ndarray], lambda_phase: Union[float, np.ndarray] = 0,
                              lambda_sq: Union[float, np.ndarray] = 0, sw_fourier_amp2: Union[float, np.ndarray] = 0,
                              n_year_days: int = 360, day_seconds: float = 86400
                              ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray],
Union[float, np.ndarray]]:
    """
    This will return the analytic solution for the times and amplitudes of the extrema of the fourier
    solution of $T'$ in the following form of the surface energy budget:

    $$
    C\\frac{\partial T'}{\partial t} = F(t) - \lambda_0 - \lambda T'(t) - \lambda_{phase}T'(t-\mathcal{T}/4) -
    \lambda_{sq} T'^2(t)
    $$

    Args:
        sw_fourier_amp1: `float [n_regions]`</br>
            The first harmonic amplitude fourier coefficients for shortwave radiation, $SW^{\\downarrow}$: $F_1$.
        heat_capacity: $C$, the heat capacity of the surface in units of $JK^{-1}m^{-2}$.</br>
            Obtained from mixed layer depth of ocean using
            [`get_heat_capacity`](../utils/radiation.md#isca_tools.utils.radiation.get_heat_capacity).
        lambda_const: The constant $\lambda$ used in the approximation for
            $\Gamma^{\\uparrow} = LW^{\\uparrow} - LW^{\\downarrow} + LH^{\\uparrow} + SH^{\\uparrow}$.</br>
        lambda_phase: The constants $\lambda_{phase}$ used in the approximation for $\Gamma^{\\uparrow}$.
        lambda_sq: The constant $\lambda_{sq}$ used in the approximation for $\Gamma^{\\uparrow}$.
        sw_fourier_amp2: `float [n_regions]`</br>
            The second harmonic amplitude fourier coefficients for shortwave radiation, $SW^{\\downarrow}$: $F_1$.
        n_year_days: Number of days in a year.
        day_seconds: Duration of a day in seconds.

    Returns:
        time_extrema1: Time in days of extrema to occur first
        time_extrema2: Time in days of extrema to occur last
        amp_extrema1: Absolute amplitude of extrema to occur first
        amp_extrema2: Absolute amplitude of extrema to occur last
    """
    f = 1 / (n_year_days * day_seconds)
    if sw_fourier_amp2 == 0:
        if lambda_sq != 0:
            raise ValueError('Cannot solve for non-zero lambda_sq with single harmonic - '
                             'get extrema numerically instead')
        tan_phase = (2 * np.pi * heat_capacity * f - lambda_phase) / lambda_const
        time_extrema1 = np.arctan(tan_phase) / (2 * np.pi) * n_year_days
        time_extrema2 = time_extrema1 + n_year_days / 2
        amp_extrema1 = np.abs(sw_fourier_amp1 / lambda_const) / (np.sqrt(1 + tan_phase ** 2))
        amp_extrema2 = amp_extrema1
    return time_extrema1, time_extrema2, amp_extrema1, amp_extrema2


def get_temp_extrema_numerical(time: np.ndarray, temp: np.ndarray, smooth_window: int = 1,
                               smooth_method: str = 'convolve') -> Tuple[float, float, float, float]:
    """
    Given the temperature `temp`, this will return the times and amplitudes of the maxima and minima. The extrema
    will be returned in time order i.e. if minima occurs first, it will be returned first.

    Args:
        time: `float [n_time]`</br>
            Time in days (assumes periodic e.g. annual mean, so `time = np.arange(360)`)
        temp: `float [n_time]`</br>
            Value of temperature at each time. Again, assume periodic.
        smooth_window: Number of time steps to use to smooth `temp` before finding extrema.
            Smaller equals more accurate fit. `1` is perfect fit.
        smooth_method: `convolve` or `spline`</br>
            If `convolve`, will smooth via convolution with window of length `smooth_window`.
            If `spline`, will fit a spline using every `smooth_window` values of `time`.

    Returns:
        time_extrema1: Time in days of extrema to occur first
        time_extrema2: Time in days of extrema to occur last
        amp_extrema1: Absolute amplitude of extrema to occur first
        amp_extrema2: Absolute amplitude of extrema to occur last
    """
    time_extrema = {}
    amp_extrema = {}
    for key in ['min', 'max']:
        var_use, spline_use = numerical.get_var_extrema_date(time, temp - np.mean(temp),
                                                             smooth_window=smooth_window, type=key, max_extrema=1,
                                                             smooth_method=smooth_method)
        time_extrema[key] = var_use[0]
        amp_extrema[key] = np.abs(spline_use(time_extrema[key]))
    # Put output in time order
    if time_extrema['min'] <= time_extrema['max']:
        return time_extrema['min'], time_extrema['max'], amp_extrema['min'], amp_extrema['max']
    else:
        return time_extrema['max'], time_extrema['min'], amp_extrema['max'], amp_extrema['min']


def gamma_linear_approx(time: np.ndarray, temp: np.ndarray,
                        lambda_const: np.ndarray, lambda_time_lag: Optional[np.ndarray] = None,
                        lambda_nl: Optional[Union[float, np.ndarray]] = None,
                        temp_anom_nl: Optional[np.ndarray] = None) -> np.ndarray:
    """
    OUTDATED FUNCTION - USED FOR `get_temp_fourier` but now use `get_temp_fourier_numerical`

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
    OUTDATED FUNCTION - USED FOR `get_temp_fourier` but now use `get_temp_fourier_numerical`

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
            [`get_heat_capacity`](../utils/radiation.md#isca_tools.utils.radiation.get_heat_capacity).
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
    temp_fourier = fourier.fourier_series(time, temp_fourier_amp, temp_fourier_phase)
    if single_harmonic_nl:
        temp_anom_nl = fourier.fourier_series(time, [0, temp_fourier_amp[1]],
                                              [temp_fourier_phase[0]])
        gamma = gamma_linear_approx(time, temp_fourier, lambda_const,
                                    lambda_time_lag, lambda_nl, temp_anom_nl)
    else:
        gamma = gamma_linear_approx(time, temp_fourier, lambda_const,
                                    lambda_time_lag, lambda_nl)
    dtemp_dt = fourier.fourier_series_deriv(time, temp_fourier_amp, temp_fourier_phase, day_seconds)
    return heat_capacity * dtemp_dt + gamma


def get_temp_fourier(time: np.ndarray, swdn: np.ndarray, heat_capacity: float,
                     lambda_const: np.ndarray, lambda_time_lag: Optional[np.ndarray] = None,
                     lambda_nl: Optional[Union[float, np.ndarray]] = None, n_harmonics: int = 2,
                     include_sw_phase: bool = False, numerical: bool = False,
                     day_seconds: float = 86400,
                     single_harmonic_nl: bool = False, return_anomaly: bool = True) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OUTDATED FUNCTION - NOW USE `get_temp_fourier_analytical` or `get_temp_fourier_numerical`

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
            [`get_heat_capacity`](../utils/radiation.md#isca_tools.utils.radiation.get_heat_capacity).
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
    sw_fourier_amp, sw_fourier_phase = fourier.get_fourier_fit(time, swdn, n_harmonics)[1:]
    if not include_sw_phase:
        sw_fourier_phase = np.zeros(n_harmonics)
    sw_fourier = fourier.fourier_series(time, sw_fourier_amp, sw_fourier_phase)
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
        temp_fourier = fourier.fourier_series(time, np.append([0], temp_fourier_amp[1:]),
                                              temp_fourier_phase)
    else:
        temp_fourier = fourier.fourier_series(time, temp_fourier_amp, temp_fourier_phase)
    return temp_fourier, temp_fourier_amp, temp_fourier_phase, sw_fourier_amp, sw_fourier_phase


def phase_coef_conversion(coef_linear: Union[float, np.ndarray, xr.DataArray],
                          coef_phase: Union[float, np.ndarray, xr.DataArray],
                          to_time: bool = True
                          ) -> Tuple[Union[float, np.ndarray, xr.DataArray], Union[float, np.ndarray, xr.DataArray]]:
    """
    Method to convert between interpretation of phase empirical coefficients. If `to_time=True`, expect
    input to be $\lambda$ and $\lambda_{ph}$, and will return
    $\lambda_{mod}$, and $2\pi ft_{ph}$. If `to_time=False`, expect the opposite.

    Conversion performed according to:

    \\begin{align}
        \lambda &= \lambda_{\\text{mod}}\cos(2 \pi ft_{\\text{ph}})
        \\\\
        \lambda_{\\text{ph}} &= \lambda_{\\text{mod}}\sin(2 \pi ft_{\\text{ph}})
    \end{align}

    Args:
        coef_linear:
            Cosine/linear-amplitude coefficient associated with the in-phase component.
            If ``to_time=True`` this is interpreted as \(\lambda\); if ``to_time=False`` this is
            interpreted as \(\lambda_{\mathrm{mod}}\).
            Can be a scalar, ``numpy.ndarray``, or ``xarray.DataArray``.
        coef_phase:
            Phase-related coefficient.
            If ``to_time=True`` this is interpreted as \(\lambda_{\mathrm{ph}}\) (sine coefficient);
            if ``to_time=False`` this is interpreted as the phase angle \(2\pi f t_{\mathrm{ph}}\)
            (radians) used inside \(\cos(\cdot)\) / \(\sin(\cdot)\).
            Can be a scalar, ``numpy.ndarray``, or ``xarray.DataArray``.
        to_time:
            Direction of conversion.
            If ``True``, convert \((\lambda,\lambda_{\mathrm{ph}})\\rightarrow
            (\lambda_{\mathrm{mod}},2\pi f t_{\mathrm{ph}})\).
            If ``False``, convert \((\lambda_{\mathrm{mod}},2\pi f t_{\mathrm{ph}})
            \\rightarrow (\lambda,\lambda_{\mathrm{ph}})\).

    Returns:
        coef_linear_out:
            Output amplitude coefficient, same container type as inputs.
            Interpreted as \(\lambda_{\mathrm{mod}}\) if ``to_time=True``; otherwise \(\lambda\).
        coef_phase_out:
            Output phase quantity, same container type as inputs.
            Interpreted as \(2\pi f t_{\mathrm{ph}}\) (radians) if ``to_time=True``; otherwise
            \(\lambda_{\mathrm{ph}}\).
    """
    if to_time:
        coef_linear_out, coef_phase_out = fourier.coef_conversion(sin_coef=coef_phase, cos_coef=coef_linear)
        # if freq is not None:
        #     coef_phase_out = coef_phase_out / (2 * np.pi * freq)  # convert to seconds
    else:
        coef_linear_out = coef_linear * np.cos(coef_phase)
        coef_phase_out = coef_linear * np.sign(coef_phase)
    return coef_linear_out, coef_phase_out
