import copy

import numpy as np
import scipy.optimize
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v, R, g
from typing import Tuple, Union, Optional


def temp_adiabat_fit_func(temp_ft_adiabat: float, temp_surf: float, sphum_surf: float,
                          pressure_surf: float, pressure_ft: float) -> float:
    """
    Adiabatic Free Troposphere temperature, $T_{A,FT}$, is defined such that surface moist static energy, $h$
    is equal to the saturated moist static energy, $h^*$, evaluated at $T_{A,FT}$ and free troposphere pressure,
    $p_{FT}$ i.e. $h = h^*(T_{A,FT}, p_{FT})$.

    Using the following approximate relationship between $z_{A, FT}$ and $T_{A, FT}$:

    $$z_{A,FT} - z_s \\approx \\frac{R^{\\dagger}}{g}(T_s + T_{A, FT})$$

    where $R^{\dagger} = \\ln(p_s/p_{FT})/2$, we can obtain $T_{A, FT}$
    by solving the following equation for modified MSE, $h^{\\dagger}$:

    $$h^{\\dagger} = (c_p - R^{\\dagger})T_s + L_v q_s =
    (c_p + R^{\\dagger})T_{A, FT} + L_vq^*(T_{A, FT})$$

    This function returns the LHS minus the RHS of this equation to then give to `scipy.optimize.fsolve` to find
    $T_{A, FT}$.

    Args:
        temp_ft_adiabat: float
            Adiabatic temperature at `pressure_ft` in Kelvin.
        temp_surf:
            Actual temperature at `pressure_surf` in Kelvin.
        sphum_surf:
            Actual specific humidity at `pressure_surf` in *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
    Returns:
        MSE discrepancy: difference between surface and free troposphere saturated adiabatic MSE.
    """
    R_mod = R * np.log(pressure_surf / pressure_ft) / 2
    mse_mod_surf = moist_static_energy(temp_surf, sphum_surf, height=0, c_p_const=c_p - R_mod)
    mse_mod_ft = moist_static_energy(temp_ft_adiabat, sphum_sat(temp_ft_adiabat, pressure_ft), height=0,
                                     c_p_const=c_p + R_mod)
    return mse_mod_surf - mse_mod_ft


def get_temp_adiabat(temp_surf: float, sphum_surf: float, pressure_surf: float, pressure_ft: float,
                     guess_temp_adiabat: float = 273):
    """
    This returns the adiabatic temperature at `pressure_ft`, $T_{A, FT}$, such that surface moist static
    energy equals free troposphere saturated moist static energy: $h = h^*(T_{A, FT}, p_{FT})$.

    Args:
        temp_surf:
            Temperature at `pressure_surf` in Kelvin.
        sphum_surf:
            Specific humidity at `pressure_surf` in *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        guess_temp_adiabat:
            Initial guess for what adiabatic temperature at `pressure_ft` should be.
    Returns:
        Adiabatic temperature at `pressure_ft` in Kelvin.
    """
    return scipy.optimize.fsolve(temp_adiabat_fit_func, guess_temp_adiabat,
                                 args=(temp_surf, sphum_surf, pressure_surf, pressure_ft))


def decompose_temp_adiabat_anomaly(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                   sphum_quant: np.ndarray, temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray,
                                   pressure_surf, pressure_ft) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The theory for $\delta T(x)$ involves the adiabatic temperature anomaly, $\Delta T_A$. This can be decomposed
    into more physically meaningful quantities:

    $$\Delta T_A(x) = T_A(x) - \overline{T_A} = \overline{T_{CE}} - T_{CE}(x) + \Delta T_{FT}(x)$$

    where:

    * $\overline{T_{CE}} = \overline{T_{FT}} - \overline{T_A}$ represents the deviation of the mean free tropospheric
    temperature from the adiabatic temperature. If at convective equilibrium, this would be zero. If the mean day had
    CAPE, this would be negative, as the lapse rate would be steeper than that expected by convection.
    * $T_{CE}(x) = T_{FT}(x) - T_A(x)$ represents the deviation of the free tropospheric
    temperature from the adiabatic temperature conditioned on percentile $x$ of near-surface temperature.
    * $\Delta T_{FT}(x) = T_{FT}(x) - \overline{T_{FT}}$ represents the gradient of the free tropospheric temperature.
    Near the tropics, we expect a weak temperature gradient (WTG) so this term would be small.

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        temp_ft_mean: `float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: `float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.

    Returns:
        `temp_adiabat_anom`: `float [n_exp, n_quant]`</br>
            Adiabatic temperature anomaly at `pressure_ft`, $\Delta T_A(x) = T_A(x) - \overline{T_A}$.
        `temp_ce_mean`: `float [n_exp]`</br>
            Deviation of mean temperature at `pressure_ft` from mean adiabatic temperature:
            $\overline{T_{CE}} = \overline{T_{FT}} - \overline{T_A}$.
        `temp_ce_quant`: `float [n_exp, n_quant]`</br>
            Deviation of temperature at `pressure_ft` from adiabatic temperature: $T_{CE}(x) = T_{FT}(x) - T_A(x)$.
            Conditioned on percentile of near-surface temperature.
        `temp_ft_anom`: `float [n_exp, n_quant]`</br>
            Temperature anomaly at `pressure_ft`, $\Delta T_{FT}(x) = T_{FT}(x) - \overline{T_{FT}}$.

    """
    n_exp, n_quant = temp_surf_quant.shape
    temp_adiabat_mean = np.zeros_like(temp_surf_mean)
    temp_adiabat_quant = np.zeros_like(temp_surf_quant)
    for i in range(n_exp):
        temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)
        for j in range(n_quant):
            temp_adiabat_quant[i, j] = get_temp_adiabat(temp_surf_quant[i, j], sphum_quant[i, j], pressure_surf,
                                                        pressure_ft)
    temp_adiabat_anom = temp_adiabat_quant - temp_adiabat_mean[:, np.newaxis]
    temp_ce_quant = temp_ft_quant - temp_adiabat_quant
    temp_ce_mean = temp_ft_mean - temp_adiabat_mean
    temp_ft_anom = temp_ft_quant - temp_ft_mean[:, np.newaxis]
    return temp_adiabat_anom, temp_ce_mean, temp_ce_quant, temp_ft_anom


def get_theory_prefactor_terms(temp: Union[np.ndarray, float], pressure_surf: float,
                               pressure_ft: float, sphum: Optional[Union[np.ndarray, float]] = None
                               ) -> Tuple[float, Union[float, np.ndarray], Union[float, np.ndarray],
Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Returns prefactors to do modified moist static energy, $\delta h^{\dagger}$ taylor expansions.

    Note units of returned variables are in *J* rather than *kJ* unlike most MSE stuff.

    Args:
        temp: `float` or `float [n_temp]`</br>
            Temperatures to compute prefactor terms for.
        pressure_surf:
            Pressure at near-surface, $p_s$ in *Pa*.
        pressure_ft:
            Pressure at free troposphere level, $p_{FT}$ in *Pa*.
        sphum: If given, will return surface prefactor terms, otherwise will return free troposphere values.</br>
            `float` or `float [n_temp]`</br>
            Specific humidity corresponding to the temperature i.e. specific humidity conditioned on same
            days as temperature given.
    Returns:
        `R_mod`: Modified gas constant, $R^{\dagger} = R\\ln(p_s/p_{FT})/2$</br>
            Units: *J/kg/K*
        `q_sat`: `float` or `float [n_temp]`</br>
            Saturated specific humidity. $q^*(T, p_s)$ if `sphum` given, otherwise $q^*(T, p_{FT})$</br>
            Units: *kg/kg*
        `alpha`: `float` or `float [n_temp]`</br>
            Clausius clapeyron parameter. $\\alpha(T, p_s)$ if `sphum` given, otherwise $\\alpha(T, p_{FT})$</br>
            Units: K$^{-1}$
        `beta_1`: `float` or `float [n_temp]`</br>
            $\\frac{dh_s^{\\dagger}}{dT_s}(T, p_s)$ if `sphum` given, otherwise
            $\\frac{h_s^{\\dagger}}{dT_A}(T, p_{FT})$.</br>
            Units: *J/kg/K*
        `beta_2`: `float` or `float [n_temp]`</br>
            $T_s\\frac{d^2h_s^{\\dagger}}{dT_s^2}(T, p_s)$ if `sphum` given, otherwise
            $T_A\\frac{d^2h_s^{\\dagger}}{dT_A^2}(T, p_{FT})$.</br>
            Units: *J/kg/K*
        `beta_3`: `float` or `float [n_temp]`</br>
            $T_s^2\\frac{d^3h_s^{\\dagger}}{dT_s^3}(T, p_s)$ if `sphum` given, otherwise
            $T_A^2\\frac{d^3h_s^{\\dagger}}{dT_A^3}(T, p_{FT})$.</br>
            Units: *J/kg/K*

    """
    R_mod = R * np.log(pressure_surf / pressure_ft) / 2
    if sphum is None:
        # if sphum not given, then free troposphere used
        c_p_use = c_p + R_mod
        pressure_use = pressure_ft
        sphum = sphum_sat(temp, pressure_use)
    else:
        c_p_use = c_p - R_mod
        pressure_use = pressure_surf

    alpha = clausius_clapeyron_factor(temp, pressure_use)
    beta_1 = c_p_use + L_v * alpha * sphum
    beta_2 = L_v * alpha * sphum * (alpha * temp - 2)
    beta_3 = L_v * alpha * sphum * ((alpha * temp) ** 2 - 6 * alpha * temp + 6)
    q_sat = sphum_sat(temp, pressure_use)
    return R_mod, q_sat, alpha, beta_1, beta_2, beta_3


def get_gamma_factors(temp_surf: float, sphum: float, pressure_surf: float, pressure_ft: float,
                      epsilon_form: bool = True) -> Tuple[
    float, float, float, float, float, float, float, float, float, float, float, float, float, float,
    float, float, float]:
    """
    Calculates the sensitivity $\gamma$ parameters such that in the simplest linear case, in `epsilon_form`, we have:

    $$
    \\begin{align}
    \\begin{split}
    \\frac{\delta T_s(x)}{\delta \overline{T_s}} \\approx
    1 &+ \gamma_{T_s}\\frac{\Delta T_s(x)}{\overline{T_s}} +
    \gamma_{r_s} \\frac{\Delta r_s(x)}{\overline{r_s}} +
    \gamma_{\epsilon}\Delta \epsilon(x) +
    \gamma_{\delta r_s} \\frac{\delta \Delta r_s(x)}{\overline{r_s}\delta \overline{T_s}} \\\\
    &+ \gamma_{\delta T_{FT}} \\frac{\delta \Delta T_{FT}(x)}{\delta \overline{T_s}} +
    \gamma_{\delta \epsilon} \\frac{\delta \Delta \epsilon(x)}{\delta \overline{T_s}}
    \\end{split}
    \\end{align}
    $$

    If non-linear and squared terms are included, the LHS becomes
    $\\left[1+\mu(x)\\frac{\delta r_s(x)}{\overline{r_s}}\\right]\\frac{\delta T_s(x)}{\delta \overline{T_s}}$
    where $\\mu(x) = \\frac{L_v \\alpha_s(x) q_s(x)}{\\beta_{s1}(x)}$, and extra terms e.g.
    $\gamma_{T_s^2}(\\frac{\Delta T_s}{\overline{T_s}})^2$ are included on the RHS.

    Args:
        temp_surf:
            Temperature at `pressure_surf` in *K*.
        sphum:
            Specific humidity at `pressure_surf` in *kg/kg*.
        pressure_surf:
            Pressure at near-surface, $p_s$ in *Pa*.
        pressure_ft:
            Pressure at free troposphere level, $p_{FT}$ in *Pa*.
        epsilon_form:
            Affects the gamma parameters with `conv` in name. If `True`, deviation from convective equilibrium
            is quantified in MSE space through: $\epsilon = h_s - h_{FT}^*$.
            Otherwise, it is quantified in temperature space through $T_{CE} = T_{FT} - T_A$.
    Returns:
        `gamma_temp_s`:
            $\gamma_{T_s}$ sensitivty parameter, such that $\gamma_{T_s}\\frac{\Delta T_s}{\overline{T_s}}$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_humidity`:
            $\gamma_{r_s}$ sensitivty parameter, such that $\gamma_{r_s}\\frac{\Delta r_s}{\overline{r_s}}$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_conv`:
            $\gamma_{\epsilon}$ sensitivty parameter, such that $\gamma_{\epsilon}\Delta \epsilon$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$. Will be 0 if `epsilon_form=False`.
        `gamma_r_change`:
            $\gamma_{\delta r_s}$ sensitivity parameter, such that
            $\gamma_{\delta r_s}\\frac{\delta \Delta r_s}{\overline{r_s}\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_ft_change`:
            $\gamma_{\delta T_{FT}}$ sensitivty parameter, such that
            $\gamma_{\delta T_{FT}}\\frac{\delta \Delta T_{FT}}{\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_conv_change`:
            $\gamma_{\delta \epsilon}$ sensitivty parameter, such that
            $\gamma_{\delta \epsilon}\\frac{\delta \Delta \epsilon}{\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$. If `epsilon_form=False`, replace $\Delta \epsilon$ with
            $\Delta T_{CE}$ in this equation, and value of this gamma variable will change.
        `gamma_temp_s_squared`:
            $\gamma_{T_s^2}$ sensitivty parameter, such that $\gamma_{T_s^2}(\\frac{\Delta T_s}{\overline{T_s}})^2$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_humidity_squared`:
            $\gamma_{r_s^2}$ sensitivty parameter, such that $\gamma_{r_s^2}(\\frac{\Delta r_s}{\overline{r_s}})^2$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_humidity`:
            $\gamma_{T_sr_s}$ senstivitiy parameter, such that
            $\gamma_{T_sr_s}\\frac{\Delta T_s}{\overline{T_s}} \\frac{\Delta r_s}{\overline{r_s}}$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_squared_humidity`:
            $\gamma_{T_s^2r_s}$ senstivitiy parameter, such that
            $\gamma_{T_s^2r_s}(\\frac{\Delta T_s}{\overline{T_s}})^2 \\frac{\Delta r_s}{\overline{r_s}}$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_humidity_squared`:
            $\gamma_{T_sr_s^2}$ senstivitiy parameter, such that
            $\gamma_{T_s^2r_s}\\frac{\Delta T_s}{\overline{T_s}} (\\frac{\Delta r_s}{\overline{r_s}})^2$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_r_mean_change`:
            $\gamma_{T_s\delta \overline{r_s}}$ senstivitiy parameter, such that
            $\gamma_{T_s\delta \overline{r_s}}\\frac{\Delta T_s}{\overline{T_s}}
            \\frac{\delta \overline{r_s}}{\overline{r_s}\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_r_change`:
            $\gamma_{T_s\delta r_s}$ senstivitiy parameter, such that
            $\gamma_{T_s\delta r_s}\\frac{\Delta T_s}{\overline{T_s}}
            \\frac{\delta \Delta r_s}{\overline{r_s}\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_r_mean_change_temp_mean_change`:
            $\gamma_{\delta \overline{r_s} \delta \overline{T_s}}$ senstivitiy parameter, such that
            $\gamma_{\delta \overline{r_s} \delta \overline{T_s}} \\frac{\delta \overline{r_s}}{\overline{r_s}}$
            is the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_ft_change_temp_mean_change`:
            $\gamma_{\delta T_{FT} \delta \overline{T_s}}$ senstivitiy parameter, such that
            $\gamma_{\delta T_{FT} \delta \overline{T_s}} \delta \Delta T_{FT}$
            is the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_conv_change_temp_mean_change`:
            $\gamma_{\delta \epsilon \delta \overline{T_s}}$ senstivitiy parameter, such that
            $\gamma_{\delta \epsilon \delta \overline{T_s}} \delta \Delta \epsilon$
            is the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$. If `epsilon_form=False`, replace
            $\Delta \epsilon$ with $\Delta T_{CE}$ in this equation, and value of this gamma variable will change.
        `gamma_conv_temp_mean_change_squared`:
            $\gamma_{\epsilon \delta \overline{T_s}^2}$ senstivitiy parameter, such that
            $\gamma_{\epsilon \delta \overline{T_s}^2} \Delta \epsilon \delta \overline{T_s}$
            is the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$. Will be 0 if `epsilon_form=False`.
            If `epsilon_form=False`, replace $\Delta \epsilon$ with $\Delta T_{CE}$ in this equation, and value of this
            gamma variable will change.
    """
    temp_adiabat = get_temp_adiabat(temp_surf, sphum, pressure_surf, pressure_ft)

    # Get parameters required for prefactors in the theory
    _, _, _, beta_a1, beta_a2, beta_a3 = get_theory_prefactor_terms(temp_adiabat, pressure_surf, pressure_ft)
    _, q_sat_surf, alpha_s, beta_s1, beta_s2, beta_s3 = get_theory_prefactor_terms(temp_surf, pressure_surf, pressure_ft,
                                                                             sphum)

    # Record coefficients of each term in equation for delta T_s(x)
    # label is anomaly that causes variation with x.

    gamma_temp_s = beta_a2 * beta_s1 / beta_a1 ** 2 * temp_surf / temp_adiabat - beta_s2 / beta_s1
    gamma_humidity = (beta_a2 / (beta_a1 ** 2 * temp_adiabat) -
                      alpha_s / beta_s1) * L_v * sphum
    gamma_r_change = -L_v * sphum / beta_s1
    gamma_temp_ft_change = beta_a1 / beta_s1
    if epsilon_form:
        gamma_conv_change = 1 / beta_s1
        gamma_conv = -beta_a2 / (beta_a1**2 * temp_adiabat)
    else:
        gamma_conv_change = -gamma_temp_ft_change
        gamma_conv = 0

    # non-linear contributions
    gamma_temp_s_squared = beta_a2 * beta_s2 / (2 * beta_a1**2) * temp_surf/temp_adiabat + (beta_s2/beta_s1)**2 - \
                           0.5*beta_s3/beta_s1
    gamma_humidity_squared = (L_v * alpha_s * sphum / beta_s1)**2
    gamma_temp_s_humidity = beta_a2 * L_v * sphum * alpha_s / beta_a1**2 * temp_surf/temp_adiabat + \
                            2 * beta_s2 * L_v * alpha_s * sphum/beta_s1**2 - beta_s2/beta_s1
    gamma_temp_s_squared_humidity = 2 * (beta_s2/beta_s1)**2 - 0.5*beta_s3/beta_s1
    gamma_temp_s_humidity_squared = 2 * L_v * alpha_s * sphum * beta_s2 / beta_s1**2

    gamma_temp_s_r_mean_change = -L_v * alpha_s * sphum * temp_surf / beta_s1
    gamma_temp_s_r_change = gamma_temp_s_r_mean_change

    gamma_r_mean_change_temp_mean_change = L_v * alpha_s * sphum / beta_s1
    gamma_temp_ft_change_temp_mean_change = beta_a2 / (beta_a1 * temp_adiabat)
    if epsilon_form:
        gamma_conv_change_temp_mean_change = beta_a2 / (beta_a1**2 * temp_adiabat)
        gamma_conv_temp_mean_change_squared = -beta_a2**2 * beta_s1 / (beta_a1**4 * temp_adiabat**2)
    else:
        gamma_conv_change_temp_mean_change = -gamma_temp_ft_change_temp_mean_change
        gamma_conv_temp_mean_change_squared = 0

    # # These terms come from keeping \Delta T_A^2 term in expansion of \Delta h_s^{\dagger}
    # # I.e. when \Delta T_A is large in reference climate.
    # gamma_temp_s_squared += 0.5 * beta_a3 * beta_s1**2 / beta_a1**3 * (temp_surf/temp_adiabat)**2
    # gamma_humidity_squared += 0.5 * beta_a3 * (L_v * sphum)**2 / beta_a1**3 / temp_adiabat**2
    # gamma_temp_s_humidity += beta_a3 * beta_s1 * L_v * sphum * temp_surf / beta_a1**3 / temp_adiabat**2

    return gamma_temp_s, gamma_humidity, gamma_conv, gamma_r_change, gamma_temp_ft_change, gamma_conv_change, \
        gamma_temp_s_squared, gamma_humidity_squared, gamma_temp_s_humidity, gamma_temp_s_squared_humidity, \
        gamma_temp_s_humidity_squared, gamma_temp_s_r_mean_change, gamma_temp_s_r_change, \
        gamma_r_mean_change_temp_mean_change, gamma_temp_ft_change_temp_mean_change, \
        gamma_conv_change_temp_mean_change, gamma_conv_temp_mean_change_squared


def get_gamma_factors2(temp_surf: float, sphum: float, temp_ft: float, pressure_surf: float,
                       pressure_ft: float) -> Tuple[
    float, float, float, float, float, float, float, float, float, float, float, float, float, float,
    float, float, float]:
    """
    Calculates the sensitivity $\gamma$ parameters such that in the simplest linear case, in `epsilon_form`, we have:

    $$
    \\begin{align}
    \\begin{split}
    \\frac{\delta T_s(x)}{\delta \overline{T_s}} \\approx
    1 &+ \gamma_{T_s}\\frac{\Delta T_s(x)}{\overline{T_s}} +
    \gamma_{r_s} \\frac{\Delta r_s(x)}{\overline{r_s}} +
    \gamma_{\epsilon}\Delta \epsilon(x) +
    \gamma_{\delta r_s} \\frac{\delta \Delta r_s(x)}{\overline{r_s}\delta \overline{T_s}} \\\\
    &+ \gamma_{\delta T_{FT}} \\frac{\delta \Delta T_{FT}(x)}{\delta \overline{T_s}} +
    \gamma_{\delta \epsilon} \\frac{\delta \Delta \epsilon(x)}{\delta \overline{T_s}}
    \\end{split}
    \\end{align}
    $$

    If non-linear and squared terms are included, the LHS becomes
    $\\left[1+\mu(x)\\frac{\delta r_s(x)}{\overline{r_s}}\\right]\\frac{\delta T_s(x)}{\delta \overline{T_s}}$
    where $\\mu(x) = \\frac{L_v \\alpha_s(x) q_s(x)}{\\beta_{s1}(x)}$, and extra terms e.g.
    $\gamma_{T_s^2}(\\frac{\Delta T_s}{\overline{T_s}})^2$ are included on the RHS.

    Args:
        temp_surf:
            Temperature at `pressure_surf` in *K*.
        sphum:
            Specific humidity at `pressure_surf` in *kg/kg*.
        pressure_surf:
            Pressure at near-surface, $p_s$ in *Pa*.
        pressure_ft:
            Pressure at free troposphere level, $p_{FT}$ in *Pa*.
        epsilon_form:
            Affects the gamma parameters with `conv` in name. If `True`, deviation from convective equilibrium
            is quantified in MSE space through: $\epsilon = h_s - h_{FT}^*$.
            Otherwise, it is quantified in temperature space through $T_{CE} = T_{FT} - T_A$.
    Returns:
        `gamma_temp_s`:
            $\gamma_{T_s}$ sensitivty parameter, such that $\gamma_{T_s}\\frac{\Delta T_s}{\overline{T_s}}$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_humidity`:
            $\gamma_{r_s}$ sensitivty parameter, such that $\gamma_{r_s}\\frac{\Delta r_s}{\overline{r_s}}$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_conv`:
            $\gamma_{\epsilon}$ sensitivty parameter, such that $\gamma_{\epsilon}\Delta \epsilon$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$. Will be 0 if `epsilon_form=False`.
        `gamma_r_change`:
            $\gamma_{\delta r_s}$ sensitivity parameter, such that
            $\gamma_{\delta r_s}\\frac{\delta \Delta r_s}{\overline{r_s}\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_ft_change`:
            $\gamma_{\delta T_{FT}}$ sensitivty parameter, such that
            $\gamma_{\delta T_{FT}}\\frac{\delta \Delta T_{FT}}{\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_conv_change`:
            $\gamma_{\delta \epsilon}$ sensitivty parameter, such that
            $\gamma_{\delta \epsilon}\\frac{\delta \Delta \epsilon}{\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$. If `epsilon_form=False`, replace $\Delta \epsilon$ with
            $\Delta T_{CE}$ in this equation, and value of this gamma variable will change.
        `gamma_temp_s_squared`:
            $\gamma_{T_s^2}$ sensitivty parameter, such that $\gamma_{T_s^2}(\\frac{\Delta T_s}{\overline{T_s}})^2$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_humidity_squared`:
            $\gamma_{r_s^2}$ sensitivty parameter, such that $\gamma_{r_s^2}(\\frac{\Delta r_s}{\overline{r_s}})^2$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_humidity`:
            $\gamma_{T_sr_s}$ senstivitiy parameter, such that
            $\gamma_{T_sr_s}\\frac{\Delta T_s}{\overline{T_s}} \\frac{\Delta r_s}{\overline{r_s}}$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_squared_humidity`:
            $\gamma_{T_s^2r_s}$ senstivitiy parameter, such that
            $\gamma_{T_s^2r_s}(\\frac{\Delta T_s}{\overline{T_s}})^2 \\frac{\Delta r_s}{\overline{r_s}}$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_humidity_squared`:
            $\gamma_{T_sr_s^2}$ senstivitiy parameter, such that
            $\gamma_{T_s^2r_s}\\frac{\Delta T_s}{\overline{T_s}} (\\frac{\Delta r_s}{\overline{r_s}})^2$ is
            the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_r_mean_change`:
            $\gamma_{T_s\delta \overline{r_s}}$ senstivitiy parameter, such that
            $\gamma_{T_s\delta \overline{r_s}}\\frac{\Delta T_s}{\overline{T_s}}
            \\frac{\delta \overline{r_s}}{\overline{r_s}\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_s_r_change`:
            $\gamma_{T_s\delta r_s}$ senstivitiy parameter, such that
            $\gamma_{T_s\delta r_s}\\frac{\Delta T_s}{\overline{T_s}}
            \\frac{\delta \Delta r_s}{\overline{r_s}\delta \overline{T_s}}$ is the contribution to
            $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_r_mean_change_temp_mean_change`:
            $\gamma_{\delta \overline{r_s} \delta \overline{T_s}}$ senstivitiy parameter, such that
            $\gamma_{\delta \overline{r_s} \delta \overline{T_s}} \\frac{\delta \overline{r_s}}{\overline{r_s}}$
            is the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_temp_ft_change_temp_mean_change`:
            $\gamma_{\delta T_{FT} \delta \overline{T_s}}$ senstivitiy parameter, such that
            $\gamma_{\delta T_{FT} \delta \overline{T_s}} \delta \Delta T_{FT}$
            is the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$.
        `gamma_conv_change_temp_mean_change`:
            $\gamma_{\delta \epsilon \delta \overline{T_s}}$ senstivitiy parameter, such that
            $\gamma_{\delta \epsilon \delta \overline{T_s}} \delta \Delta \epsilon$
            is the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$. If `epsilon_form=False`, replace
            $\Delta \epsilon$ with $\Delta T_{CE}$ in this equation, and value of this gamma variable will change.
        `gamma_conv_temp_mean_change_squared`:
            $\gamma_{\epsilon \delta \overline{T_s}^2}$ senstivitiy parameter, such that
            $\gamma_{\epsilon \delta \overline{T_s}^2} \Delta \epsilon \delta \overline{T_s}$
            is the contribution to $\\frac{\delta T_s(x)}{\delta \overline{T_s}}$. Will be 0 if `epsilon_form=False`.
            If `epsilon_form=False`, replace $\Delta \epsilon$ with $\Delta T_{CE}$ in this equation, and value of this
            gamma variable will change.
    """
    # Get parameters required for prefactors in the theory
    _, _, _, beta_ft1, beta_ft2, beta_ft3 = get_theory_prefactor_terms(temp_ft, pressure_surf, pressure_ft)
    _, q_sat_surf, alpha_s, beta_s1, beta_s2, beta_s3 = get_theory_prefactor_terms(temp_surf, pressure_surf,
                                                                                   pressure_ft, sphum)

    # Record coefficients of each term in equation for delta T_s(x)
    # label is anomaly that causes variation with x.
    gamma_temp_s = beta_ft2 * beta_s1 / beta_ft1 ** 2 * temp_surf / temp_ft - beta_s2 / beta_s1
    gamma_r = (alpha_s / beta_s1 - beta_ft2 / (beta_ft1 ** 2 * temp_ft)) * L_v * sphum
    gamma_epsilon = beta_ft2 / (beta_ft1**2 * temp_ft)
    gamma_r_change = L_v * sphum / beta_s1
    gamma_temp_ft_change = beta_ft1 / beta_s1
    gamma_epsilon_change = 1/beta_s1

    # # These terms come from keeping \Delta T_A^2 term in expansion of \Delta h_s^{\dagger}
    # # I.e. when \Delta T_A is large in reference climate.
    # gamma_temp_s_squared += 0.5 * beta_a3 * beta_s1**2 / beta_a1**3 * (temp_surf/temp_adiabat)**2
    # gamma_humidity_squared += 0.5 * beta_a3 * (L_v * sphum)**2 / beta_a1**3 / temp_adiabat**2
    # gamma_temp_s_humidity += beta_a3 * beta_s1 * L_v * sphum * temp_surf / beta_a1**3 / temp_adiabat**2

    return gamma_temp_s, gamma_r, gamma_epsilon, gamma_r_change, gamma_temp_ft_change, gamma_epsilon_change

# def get_delta_mse_mod_anom_theory(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
#                                   sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
#                                   taylor_terms: str = 'linear', mse_mod_mean_change: Optional[float] = None
#                                   ) -> Tuple[np.ndarray, dict, np.ndarray]:
#     """
#     This function returns an approximation in the change in modified MSE anomaly,
#     $\delta \Delta h_s^{\dagger} = \delta (h_s^{\dagger}(x) - \overline{h_s^{\dagger}})$, with warming -
#     the basis of a theory for $\delta T_s(x)$.
#
#     Doing a second order taylor expansion of $h_s^{\dagger}$ in the base climate,
#     about adiabatic temperature $\overline{T_A}$, we can get:
#
#     $$\\Delta h^{\\dagger}(x) \\approx \\beta_{A1} \\Delta T_A + \\frac{1}{2\\overline{T_A}}
#     \\beta_{A2} \\Delta T_A^2$$
#
#     Terms in equation:
#         * $h_s^{\\dagger} = (c_p - R^{\\dagger})T_s + L_v q_s = (c_p + R^{\\dagger})T_A + L_vq^*(T_A, p_{FT})$ where
#         $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
#         * $\\Delta T_A = T_A(x) - \overline{T_A}$
#         * $\\beta_{A1} = \\frac{d\\overline{h_s^{\\dagger}}}{d\overline{T_A}} = c_p + R^{\dagger} + L_v \\alpha_A q_A^*$
#         * $\\beta_{A2} = \overline{T_A} \\frac{d^2\\overline{h_s^{\\dagger}}}{d\overline{T_A}^2} =
#          \overline{T_A}\\frac{d\\beta_1}{d\overline{T_A}} = L_v \\alpha_A q_A^*(\\alpha_A \\overline{T_A} - 2)$
#         * All terms on RHS are evaluated at the free tropospheric adiabatic temperature, $T_A$. I.e.
#         $q^* = q^*(T_A, p_{FT})$ where $p_{FT}$ is the free tropospheric pressure.
#
#     Doing a second taylor expansion on this equation for a change with warming between simulations, $\delta$, we can
#     decompose $\delta \\Delta h_s^{\\dagger}(x)$ into terms involving $\delta \Delta T_A$ and $\delta \overline{T_A}$
#
#     We can then use a third taylor expansion to relate $\delta \overline{T_A}$ to $\delta \overline{h^{\\dagger}}$:
#
#     $$\\delta \\overline{T_A} \\approx \\frac{\\delta \\overline{h^{\\dagger}}}{\\beta_{A1}} -
#     \\frac{1}{2} \\frac{\\beta_{A2}}{\\beta_{A1}^3 \\overline{T_A}} (\\delta \\overline{h_s^{\\dagger}})^2$$
#
#     Overall, we get $\delta \Delta h_s^{\dagger}$ as a function of $\delta \Delta T_A$,
#     $\delta \overline{h_s^{\\dagger}}$ and quantities evaluated at the base climate.
#     The `taylor_terms` variable can be used to specify how many terms we want to keep.
#
#     The simplest equation with `taylor_terms = 'linear'` is:
#
#     $$\\delta \\Delta h_s^{\\dagger} \\approx \\beta_{A1} \\delta \\Delta T_A +
#     \\frac{\\beta_{A2}}{\\beta_{A1}}\\frac{\\Delta T_A}{\\overline{T_A}} \\delta \\overline{h_s^{\\dagger}}$$
#
#     Args:
#         temp_surf_mean: `float [n_exp]`</br>
#             Average near surface temperature of each simulation, corresponding to a different
#             optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
#         temp_surf_quant: `float [n_exp, n_quant]`</br>
#             `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
#             experiment `i`. Units: *K*.</br>
#             Note that `quant_use` is not provided as not needed by this function, but is likely to be
#             `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
#             of a quantity.
#         sphum_mean: `float [n_exp]`</br>
#             Average near surface specific humidity of each simulation. Units: *kg/kg*.
#         sphum_quant: `float [n_exp, n_quant]`</br>
#             `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
#              corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
#         pressure_surf:
#             Pressure at near-surface in *Pa*.
#         pressure_ft:
#             Pressure at free troposphere level in *Pa*.
#         taylor_terms:
#             The approximations in this equation arise from the three taylor series mentioned above, we can specify
#             how many terms we want to keep, with one of the 3 options below:
#
#             * `linear`: Only keep the two terms which are linear in all three taylor series i.e.
#             $\\delta \\Delta h_s^{\\dagger} \\approx \\beta_{A1} \\delta \\Delta T_A +
#             \\frac{\\beta_{A2}}{\\beta_{A1}}\\frac{\\Delta T_A}{\\overline{T_A}} \\delta \\overline{h_s^{\\dagger}}$
#             * `non_linear`: Keep *LL*, *LLL* and *LNL* terms.
#             * `squared_0`: Keep terms linear and squared in first expansion and then just linear terms:
#             *LL*, *LLL*, *SL*, *SLL*.
#             * `squared`: Keep four additional terms corresponding to *LLS*, *LSL*, *LNL* and *SNL* terms in the
#             taylor series. SNL means second order in the first taylor series mentioned above, non-linear
#             (i.e. $\\delta \\Delta T_A\\delta \\overline{h^{\\dagger}}$ terms)  in the second
#             and linear in the third. These 5 terms are the most significant non-linear terms.
#             * `full`: In addition to the terms in `squared`, we keep the usually small *LSS* and *SLS* terms.
#         mse_mod_mean_change: `float [n_exp]`</br>
#             Can provide the $\\delta \\overline{h^{\\dagger}}$ in J/kg. Use this if wan't to use a particular taylor
#             approximation for this.
#     Returns:
#         `delta_mse_mod_anomaly`: `float [n_quant]`</br>
#             $\delta \Delta h_s^{\dagger}$ conditioned on each quantile of near-surface temperature. Units: *kJ/kg*.
#         `info_dict`: Dictionary with 5 keys: `temp_adiabat_anom`, `mse_mod_mean`, `mse_mod_mean_squared`,
#             `mse_mod_mean_cubed`, `non_linear`.</br>
#             For each key, a list containing a prefactor computed in the base climate and a change between simulations is
#             returned. I.e. for `info_dict[non_linear][1]` would be $\\delta \\Delta T_A\\delta \\overline{h^{\\dagger}}$
#             and the total contribution of non-linear terms to $\delta \Delta h^{\dagger}$ would be
#             `info_dict[non_linear][0] * info_dict[non_linear][1]`. In the `linear` case this would be zero,
#             and `info_dict[temp_adiabat_anom][0]`$=\\beta_{A1}$ and `info_dict[mse_mod_mean][0]`$=
#             \\frac{\\beta_{A2}}{\\beta_{A1}}\\frac{\\Delta T_A}{\\overline{T_A}}$ would be the only non-zero prefactors.
#             Units of prefactor multiplied by change is *kJ/kg*.
#         `temp_adiabat_anom`: `float [n_exp, n_quant]`</br>
#             The adiabatic free troposphere temperature anomaly, $\Delta T_A$ for each experiment, as may be of use.
#             Units: *Kelvin*.
#     """
#     # Compute adiabatic temperatures
#     n_exp, n_quant = temp_surf_quant.shape
#     temp_adiabat_mean = np.zeros_like(temp_surf_mean)
#     temp_adiabat_quant = np.zeros_like(temp_surf_quant)
#     for i in range(n_exp):
#         temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)
#         for j in range(n_quant):
#             temp_adiabat_quant[i, j] = get_temp_adiabat(temp_surf_quant[i, j], sphum_quant[i, j], pressure_surf,
#                                                         pressure_ft)
#     temp_adiabat_anom = temp_adiabat_quant - temp_adiabat_mean[:, np.newaxis]
#     delta_temp_adiabat_anom = temp_adiabat_anom[1] - temp_adiabat_anom[0]
#
#     # Parameters needed for taylor expansions - most compute using adiabatic temperature in free troposphere.
#     R_mod, _, _, beta_1, beta_2, beta_3 = get_theory_prefactor_terms(temp_adiabat_mean[0], pressure_surf,
#                                                                      pressure_ft)
#
#     # Compute modified MSE - need in units of J/kg at the moment hence multiply by 1000
#     if mse_mod_mean_change is None:
#         mse_mod_mean = moist_static_energy(temp_surf_mean, sphum_mean, height=0, c_p_const=c_p - R_mod) * 1000
#         mse_mod_mean_change = mse_mod_mean[1] - mse_mod_mean[0]
#
#     # Decompose Taylor Expansions - 3 in total
#     # l means linear, s means squared and n means non-linear
#     # first index is for Delta expansion i.e. base climate - quantile about mean
#     # second index is for delta expansion i.e. difference between climates
#     # third index is for conversion between delta_temp_adiabat_mean and delta_mse_mod_mean
#     # I neglect all terms that are more than squared in two or more of these taylor expansions
#     if taylor_terms.lower() not in ['linear', 'non_linear', 'squared_0', 'squared', 'full']:
#         raise ValueError(f'taylor_terms given is {taylor_terms}, but must be linear, squared_0, squared or full.')
#
#     term_ll = beta_1 * delta_temp_adiabat_anom
#     term_lll = beta_2 / beta_1 * temp_adiabat_anom[0] / temp_adiabat_mean[0] * mse_mod_mean_change
#     if taylor_terms == 'squared_0':
#         # term_sl = beta_2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] * delta_temp_adiabat_anom
#         term_sl = 0
#         term_sll = 0.5 * beta_3 / beta_1 * (temp_adiabat_anom[0] / temp_adiabat_mean[0]) ** 2 * mse_mod_mean_change
#         term_lls = 0
#         term_lsl = 0
#         term_lnl = 0
#         term_snl = 0
#     elif 'linear' not in taylor_terms:
#         term_sl = beta_2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] * delta_temp_adiabat_anom
#         term_sll = 0.5 * beta_3 / beta_1 * (temp_adiabat_anom[0] / temp_adiabat_mean[0]) ** 2 * mse_mod_mean_change
#         term_lls = -0.5 * beta_2 ** 2 / beta_1 ** 3 * temp_adiabat_anom[0] / temp_adiabat_mean[
#             0] ** 2 * mse_mod_mean_change ** 2
#         term_lsl = 0.5 * beta_3 / beta_1 ** 2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] ** 2 * mse_mod_mean_change ** 2
#         term_lnl = beta_2 / beta_1 / temp_adiabat_mean[0] * delta_temp_adiabat_anom * mse_mod_mean_change
#         term_snl = beta_3 / beta_1 * temp_adiabat_anom[0] / temp_adiabat_mean[
#             0] ** 2 * delta_temp_adiabat_anom * mse_mod_mean_change
#     else:
#         term_sl = 0
#         term_sll = 0
#         term_lls = 0
#         term_lsl = 0
#         term_lnl = 0 if taylor_terms == 'linear' else beta_2 / beta_1 / temp_adiabat_mean[0] * delta_temp_adiabat_anom * mse_mod_mean_change
#         term_snl = 0
#     # Extra squared-squared terms
#     if taylor_terms == 'full':
#         term_lss = -0.5 * beta_3 * beta_2 / beta_1 ** 4 * temp_adiabat_anom[0] / temp_adiabat_mean[
#             0] ** 3 * mse_mod_mean_change ** 3
#         term_sls = -0.25 * beta_3 * beta_2 / beta_1 ** 3 * temp_adiabat_anom[0] ** 2 / temp_adiabat_mean[
#             0] ** 3 * mse_mod_mean_change ** 2
#         # The two below are very small so should exclude
#         # term_ss = 0.5 * beta_2/temp_adiabat_mean[0] * delta_temp_anom**2
#         # term_lns = -0.5 * beta_2**2/beta_1**3/temp_adiabat_mean[0]**2 * delta_temp_anom * delta_mse**2
#     else:
#         term_lss = 0
#         term_sls = 0
#
#     # Keep track of contribution to different changes
#     # Have a prefactor based on current climate and a change between simulations for each factor.
#     info_dict = {'temp_adiabat_anom': [(term_ll + term_sl) / delta_temp_adiabat_anom / 1000, delta_temp_adiabat_anom],
#                  'mse_mod_mean': [(term_lll + term_sll) / mse_mod_mean_change / 1000, mse_mod_mean_change],
#                  'mse_mod_mean_squared': [(term_lls + term_lsl + term_sls) / mse_mod_mean_change ** 2 / 1000,
#                                           mse_mod_mean_change ** 2],
#                  'mse_mod_mean_cubed': [term_lss / mse_mod_mean_change ** 3 / 1000, mse_mod_mean_change ** 3],
#                  'non_linear': [(term_lnl + term_snl) / (delta_temp_adiabat_anom * mse_mod_mean_change) / 1000,
#                                 delta_temp_adiabat_anom * mse_mod_mean_change]
#                  }
#
#     final_answer = term_ll + term_lll + term_sl + term_lls + term_sll + term_lsl + term_lnl + term_snl + term_lss + \
#                    term_sls
#     final_answer = final_answer / 1000  # convert to units of kJ/kg
#     return final_answer, info_dict, temp_adiabat_anom


def mse_mod_anom_change_ft_expansion(temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray,
                                     pressure_surf: float, pressure_ft: float,
                                     taylor_terms: str = 'linear', mse_mod_mean_change: Optional[float] = None,
                                     temp_ft_anom0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    This function returns an approximation in the change in modified MSE anomaly,
    $\delta \Delta h_s^{\dagger} = \delta (h_s^{\dagger}(x) - \overline{h_s^{\dagger}})$, with warming -
    the basis of a theory for $\delta T_s(x)$.

    Doing a second order taylor expansion of $h_s^{\dagger}$ in the base climate,
    about adiabatic temperature $\overline{T_A}$, we can get:

    $$\\Delta h^{\\dagger}(x) \\approx \\beta_{A1} \\Delta T_A + \\frac{1}{2\\overline{T_A}}
    \\beta_{A2} \\Delta T_A^2$$

    Terms in equation:
        * $h_s^{\\dagger} = (c_p - R^{\\dagger})T_s + L_v q_s = (c_p + R^{\\dagger})T_A + L_vq^*(T_A, p_{FT})$ where
        $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\\Delta T_A = T_A(x) - \overline{T_A}$
        * $\\beta_{A1} = \\frac{d\\overline{h_s^{\\dagger}}}{d\overline{T_A}} = c_p + R^{\dagger} + L_v \\alpha_A q_A^*$
        * $\\beta_{A2} = \overline{T_A} \\frac{d^2\\overline{h_s^{\\dagger}}}{d\overline{T_A}^2} =
         \overline{T_A}\\frac{d\\beta_1}{d\overline{T_A}} = L_v \\alpha_A q_A^*(\\alpha_A \\overline{T_A} - 2)$
        * All terms on RHS are evaluated at the free tropospheric adiabatic temperature, $T_A$. I.e.
        $q^* = q^*(T_A, p_{FT})$ where $p_{FT}$ is the free tropospheric pressure.

    Doing a second taylor expansion on this equation for a change with warming between simulations, $\delta$, we can
    decompose $\delta \\Delta h_s^{\\dagger}(x)$ into terms involving $\delta \Delta T_A$ and $\delta \overline{T_A}$

    We can then use a third taylor expansion to relate $\delta \overline{T_A}$ to $\delta \overline{h^{\\dagger}}$:

    $$\\delta \\overline{T_A} \\approx \\frac{\\delta \\overline{h^{\\dagger}}}{\\beta_{A1}} -
    \\frac{1}{2} \\frac{\\beta_{A2}}{\\beta_{A1}^3 \\overline{T_A}} (\\delta \\overline{h_s^{\\dagger}})^2$$

    Overall, we get $\delta \Delta h_s^{\dagger}$ as a function of $\delta \Delta T_A$,
    $\delta \overline{h_s^{\\dagger}}$ and quantities evaluated at the base climate.
    The `taylor_terms` variable can be used to specify how many terms we want to keep.

    The simplest equation with `taylor_terms = 'linear'` is:

    $$\\delta \\Delta h_s^{\\dagger} \\approx \\beta_{A1} \\delta \\Delta T_A +
    \\frac{\\beta_{A2}}{\\beta_{A1}}\\frac{\\Delta T_A}{\\overline{T_A}} \\delta \\overline{h_s^{\\dagger}}$$

    Args:
        temp_ft_mean: `float [n_exp]`</br>
            Average free tropospheric temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_ft_quant: `float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is free tropospheric temperature, averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        taylor_terms:
            The approximations in this equation arise from the three taylor series mentioned above, we can specify
            how many terms we want to keep, with one of the 3 options below:

            * `linear`: Only keep the two terms which are linear in all three taylor series i.e.
            $\\delta \\Delta h_s^{\\dagger} \\approx \\beta_{A1} \\delta \\Delta T_A +
            \\frac{\\beta_{A2}}{\\beta_{A1}}\\frac{\\Delta T_A}{\\overline{T_A}} \\delta \\overline{h_s^{\\dagger}}$
            * `non_linear`: Keep *LL*, *LLL* and *LNL* terms.
            * `squared_0`: Keep terms linear and squared in first expansion and then just linear terms:
            *LL*, *LLL*, *SL*, *SLL*.
            * `squared`: Keep four additional terms corresponding to *LLS*, *LSL*, *LNL* and *SNL* terms in the
            taylor series. SNL means second order in the first taylor series mentioned above, non-linear
            (i.e. $\\delta \\Delta T_A\\delta \\overline{h^{\\dagger}}$ terms)  in the second
            and linear in the third. These 5 terms are the most significant non-linear terms.
            * `full`: In addition to the terms in `squared`, we keep the usually small *LSS* and *SLS* terms.
        mse_mod_mean_change: `float [n_exp]`</br>
            Can provide the $\\delta \\overline{h^{\\dagger}}$ in J/kg. Use this if wan't to use a particular taylor
            approximation for this.
    Returns:
        `delta_mse_mod_anomaly`: `float [n_quant]`</br>
            $\delta \Delta h_s^{\dagger}$ conditioned on each quantile of near-surface temperature. Units: *kJ/kg*.
        `info_dict`: Dictionary with 5 keys: `temp_ft_anom`, `mse_mod_mean`, `mse_mod_mean_squared`,
            `mse_mod_mean_cubed`, `non_linear`.</br>
            For each key, a list containing a prefactor computed in the base climate and a change between simulations is
            returned. I.e. for `info_dict[non_linear][1]` would be $\\delta \\Delta T_A\\delta \\overline{h^{\\dagger}}$
            and the total contribution of non-linear terms to $\delta \Delta h^{\dagger}$ would be
            `info_dict[non_linear][0] * info_dict[non_linear][1]`. In the `linear` case this would be zero,
            and `info_dict[temp_adiabat_anom][0]`$=\\beta_{A1}$ and `info_dict[mse_mod_mean][0]`$=
            \\frac{\\beta_{A2}}{\\beta_{A1}}\\frac{\\Delta T_A}{\\overline{T_A}}$ would be the only non-zero prefactors.
            Units of prefactor multiplied by change is *kJ/kg*.
    """
    temp_ft_anom = temp_ft_quant - temp_ft_mean[:, np.newaxis]
    delta_temp_ft_anom = temp_ft_anom[1] - temp_ft_anom[0]
    if temp_ft_anom0 is None:
        temp_ft_anom0 = temp_ft_anom[0]

    # Parameters needed for taylor expansions - most compute using adiabatic temperature in free troposphere.
    R_mod, _, _, beta_1, beta_2, beta_3 = get_theory_prefactor_terms(temp_ft_mean[0], pressure_surf, pressure_ft)

    # Compute modified MSE - need in units of J/kg at the moment hence multiply by 1000
    if mse_mod_mean_change is None:
        mse_mod_mean = moist_static_energy(temp_ft_mean, sphum_sat(temp_ft_mean, pressure_ft), height=0,
                                           c_p_const=c_p + R_mod) * 1000
        mse_mod_mean_change = mse_mod_mean[1] - mse_mod_mean[0]

    # Decompose Taylor Expansions - 3 in total
    # l means linear, s means squared and n means non-linear
    # first index is for Delta expansion i.e. base climate - quantile about mean
    # second index is for delta expansion i.e. difference between climates
    # third index is for conversion between delta_temp_adiabat_mean and delta_mse_mod_mean
    # I neglect all terms that are more than squared in two or more of these taylor expansions
    if taylor_terms.lower() not in ['linear', 'non_linear', 'squared_0', 'squared', 'full']:
        raise ValueError(f'taylor_terms given is {taylor_terms}, but must be linear, squared_0, squared or full.')

    term_ll = beta_1 * delta_temp_ft_anom
    term_lll = beta_2 / beta_1 * temp_ft_anom0 / temp_ft_mean[0] * mse_mod_mean_change
    if taylor_terms == 'squared_0':
        # term_sl = beta_2 * temp_adiabat_anom[0] / temp_adiabat_mean[0] * delta_temp_adiabat_anom
        term_sl = 0
        term_sll = 0.5 * beta_3 / beta_1 * (temp_ft_anom0 / temp_ft_mean[0]) ** 2 * mse_mod_mean_change
        term_lls = 0
        term_lsl = 0
        term_lnl = 0
        term_snl = 0
    elif 'linear' not in taylor_terms:
        term_sl = beta_2 * temp_ft_anom0 / temp_ft_mean[0] * delta_temp_ft_anom
        term_sll = 0.5 * beta_3 / beta_1 * (temp_ft_anom0 / temp_ft_mean[0]) ** 2 * mse_mod_mean_change
        term_lls = -0.5 * beta_2 ** 2 / beta_1 ** 3 * temp_ft_anom0 / temp_ft_mean[
            0] ** 2 * mse_mod_mean_change ** 2
        term_lsl = 0.5 * beta_3 / beta_1 ** 2 * temp_ft_anom0 / temp_ft_mean[0] ** 2 * mse_mod_mean_change ** 2
        term_lnl = beta_2 / beta_1 / temp_ft_mean[0] * delta_temp_ft_anom * mse_mod_mean_change
        term_snl = beta_3 / beta_1 * temp_ft_anom0 / temp_ft_mean[
            0] ** 2 * delta_temp_ft_anom * mse_mod_mean_change
    else:
        term_sl = 0
        term_sll = 0
        term_lls = 0
        term_lsl = 0
        term_lnl = 0 if taylor_terms == 'linear' else beta_2 / beta_1 / temp_ft_mean[0] * delta_temp_ft_anom * mse_mod_mean_change
        term_snl = 0
    # Extra squared-squared terms
    if taylor_terms == 'full':
        term_lss = -0.5 * beta_3 * beta_2 / beta_1 ** 4 * temp_ft_anom0 / temp_ft_mean[
            0] ** 3 * mse_mod_mean_change ** 3
        term_sls = -0.25 * beta_3 * beta_2 / beta_1 ** 3 * temp_ft_anom0 ** 2 / temp_ft_mean[
            0] ** 3 * mse_mod_mean_change ** 2
        # The two below are very small so should exclude
        # term_ss = 0.5 * beta_2/temp_adiabat_mean[0] * delta_temp_anom**2
        # term_lns = -0.5 * beta_2**2/beta_1**3/temp_adiabat_mean[0]**2 * delta_temp_anom * delta_mse**2
    else:
        term_lss = 0
        term_sls = 0

    # Keep track of contribution to different changes
    # Have a prefactor based on current climate and a change between simulations for each factor.
    info_dict = {'temp_ft_anom': [(term_ll + term_sl) / delta_temp_ft_anom / 1000, delta_temp_ft_anom],
                 'mse_mod_mean': [(term_lll + term_sll) / mse_mod_mean_change / 1000, mse_mod_mean_change],
                 'mse_mod_mean_squared': [(term_lls + term_lsl + term_sls) / mse_mod_mean_change ** 2 / 1000,
                                          mse_mod_mean_change ** 2],
                 'mse_mod_mean_cubed': [term_lss / mse_mod_mean_change ** 3 / 1000, mse_mod_mean_change ** 3],
                 'non_linear': [(term_lnl + term_snl) / (delta_temp_ft_anom * mse_mod_mean_change) / 1000,
                                delta_temp_ft_anom * mse_mod_mean_change]
                 }

    final_answer = term_ll + term_lll + term_sl + term_lls + term_sll + term_lsl + term_lnl + term_snl + term_lss + \
                   term_sls
    final_answer = final_answer / 1000  # convert to units of kJ/kg
    return final_answer, info_dict


def mse_mod_change_surf_expansion(temp_surf: np.ndarray, sphum_surf: np.ndarray, epsilon: np.ndarray,
                                  pressure_surf: float, pressure_ft: float, taylor_terms: str = 'linear',
                                  temp_use_rh_term: Optional[Union[np.ndarray, float]] = None
                                  ) -> Tuple[Union[np.ndarray, float], dict]:
    """
    Does a taylor expansion of the change in modified moist static energy, $\delta h^{\dagger}$:

    $$\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s + L_v q^* \\delta r +
    0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$$

    In terms of $\\beta$ parameters, this can be written as:

    $$\\delta h^{\\dagger} \\approx \\beta_{s1} \delta T_s + L_v q^* \\delta r +
     \\frac{1}{2 T_s} \\beta_{s2} \delta T_s^2$$

    The last term is only included if `taylor_terms == 'squared'`.

    Args:
        temp_surf: `float [n_exp]` or `float [n_exp, n_quant]` </br>
            Near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        sphum_surf: `float [n_exp]` or `float [n_exp, n_quant]` </br>
            Near surface specific humidity of each simulation. Units: *kg/kg*.
        epsilon: `float [n_exp]` or `float [n_exp, n_quant]` </br>
            $h_s - h^*_{FT}$. Units: *kJ/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        taylor_terms:
            How many taylor series terms to keep in the expansions for changes in modified moist static energy:

            * `linear`: $\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s +
                L_v \\alpha q^* \\delta r$
            * `squared`: Includes the additional term $0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$
        temp_use_rh_term: `float` or `float [n_quant]` </br>
            Can specify temperature to evaluate $q^*(T, p_s)$ in the $L_v q^* \\delta r$ term. May want to do
            this if want to approximate $q^*(T(x), p_s) \\approx q^*(\overline{T}, p_s)$ for this term, so can combine
            $\\delta r(x)$ and $\\delta \overline{r}$ terms. If `None`, then will use `temp_surf[0]`.

    Returns:
        `delta_mse_mod`: `float` or `float [n_quant]` </br>
            Approximation of $\delta h^{\\dagger}$. Units are *kJ/kg*.
        `info_dict`: Dictionary containing 3 keys: `rh`, `temp`, `temp_squared`. </br>
            For each key, there is a list with first value being the prefactor and second the change in the expansion.
            The sum of all these prefactors multiplied by the changes equals the full theory.
            Units of prefactors are *kJ/kg* divided by units of the change term.
    """
    _, _, _, beta_s1, beta_s2, _ = get_theory_prefactor_terms(temp_surf[0], pressure_surf, pressure_ft, sphum_surf[0])

    delta_temp = temp_surf[1] - temp_surf[0]
    rh = sphum_surf / sphum_sat(temp_surf, pressure_surf)
    delta_rh = rh[1] - rh[0]
    delta_epsilon = epsilon[1] - epsilon[0]

    if temp_use_rh_term is None:
        temp_use_rh_term = temp_surf[0]
    q_sat_s = sphum_sat(temp_use_rh_term, pressure_surf)
    alpha_s = clausius_clapeyron_factor(temp_use_rh_term, pressure_surf)
    coef_rh = L_v * q_sat_s
    coef_temp = beta_s1

    if taylor_terms == 'squared':
        # Add extra term in taylor expansion of delta_mse_mod if requested
        coef_non_linear = L_v * q_sat_s * alpha_s
        coef_temp_squared = 0.5 * beta_s2 / temp_surf[0]
    elif taylor_terms == 'non_linear':
        coef_non_linear = L_v * q_sat_s * alpha_s
        coef_temp_squared = sphum_surf[0] * 0
    elif taylor_terms == 'linear':
        coef_non_linear = sphum_surf[0] * 0
        coef_temp_squared = sphum_surf[0] * 0  # set to 0 for all values
    else:
        raise ValueError(f"taylor_terms given is {taylor_terms}, but must be 'linear', 'non_linear' or 'squared'")

    final_answer = (coef_rh * delta_rh + coef_temp * delta_temp + coef_non_linear * delta_rh * delta_temp +
                    coef_temp_squared * delta_temp ** 2) / 1000 - delta_epsilon
    info_dict = {'rh': [coef_rh / 1000, delta_rh], 'temp': [coef_temp / 1000, delta_temp],
                 'epsilon': [-1, delta_epsilon],
                 'non_linear': [coef_non_linear / 1000, delta_temp * delta_rh],
                 'temp_squared': [coef_temp_squared / 1000, delta_temp ** 2]}
    return final_answer, info_dict


# def do_delta_mse_mod_taylor_expansion(temp_surf: np.ndarray, sphum_surf: np.ndarray,
#                                       pressure_surf: float, pressure_ft: float, taylor_terms: str = 'linear',
#                                       temp_use_rh_term: Optional[Union[np.ndarray, float]] = None
#                                       ) -> Tuple[Union[np.ndarray, float], dict]:
#     """
#     Does a taylor expansion of the change in modified moist static energy, $\delta h^{\dagger}$:
#
#     $$\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s + L_v q^* \\delta r +
#     0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$$
#
#     In terms of $\\beta$ parameters, this can be written as:
#
#     $$\\delta h^{\\dagger} \\approx \\beta_{s1} \delta T_s + L_v q^* \\delta r +
#      \\frac{1}{2 T_s} \\beta_{s2} \delta T_s^2$$
#
#     The last term is only included if `taylor_terms == 'squared'`.
#
#     Args:
#         temp_surf: `float [n_exp]` or `float [n_exp, n_quant]` </br>
#             Near surface temperature of each simulation, corresponding to a different
#             optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
#         sphum_surf: `float [n_exp]` or `float [n_exp, n_quant]` </br>
#             Near surface specific humidity of each simulation. Units: *kg/kg*.
#         pressure_surf:
#             Pressure at near-surface in *Pa*.
#         pressure_ft:
#             Pressure at free troposphere level in *Pa*.
#         taylor_terms:
#             How many taylor series terms to keep in the expansions for changes in modified moist static energy:
#
#             * `linear`: $\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s +
#                 L_v \\alpha q^* \\delta r$
#             * `squared`: Includes the additional term $0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$
#         temp_use_rh_term: `float` or `float [n_quant]` </br>
#             Can specify temperature to evaluate $q^*(T, p_s)$ in the $L_v q^* \\delta r$ term. May want to do
#             this if want to approximate $q^*(T(x), p_s) \\approx q^*(\overline{T}, p_s)$ for this term, so can combine
#             $\\delta r(x)$ and $\\delta \overline{r}$ terms. If `None`, then will use `temp_surf[0]`.
#
#     Returns:
#         `delta_mse_mod`: `float` or `float [n_quant]` </br>
#             Approximation of $\delta h^{\\dagger}$. Units are *kJ/kg*.
#         `info_dict`: Dictionary containing 3 keys: `rh`, `temp`, `temp_squared`. </br>
#             For each key, there is a list with first value being the prefactor and second the change in the expansion.
#             The sum of all these prefactors multiplied by the changes equals the full theory.
#             Units of prefactors are *kJ/kg* divided by units of the change term.
#     """
#     _, _, _, beta_s1, beta_s2, _ = get_theory_prefactor_terms(temp_surf[0], pressure_surf, pressure_ft,
#                                                                           sphum_surf[0])
#
#     delta_temp = temp_surf[1] - temp_surf[0]
#     rh = sphum_surf / sphum_sat(temp_surf, pressure_surf)
#     delta_rh = rh[1] - rh[0]
#
#     if temp_use_rh_term is None:
#         temp_use_rh_term = temp_surf[0]
#     q_sat_s = sphum_sat(temp_use_rh_term, pressure_surf)
#     alpha_s = clausius_clapeyron_factor(temp_use_rh_term, pressure_surf)
#     coef_rh = L_v * q_sat_s
#     coef_temp = beta_s1
#
#     if taylor_terms == 'squared':
#         # Add extra term in taylor expansion of delta_mse_mod if requested
#         coef_non_linear = L_v * q_sat_s * alpha_s
#         coef_temp_squared = 0.5 * beta_s2 / temp_surf[0]
#     elif taylor_terms == 'non_linear':
#         coef_non_linear = L_v * q_sat_s * alpha_s
#         coef_temp_squared = sphum_surf[0] * 0
#     elif taylor_terms == 'linear':
#         coef_non_linear = sphum_surf[0] * 0
#         coef_temp_squared = sphum_surf[0] * 0  # set to 0 for all values
#     else:
#         raise ValueError(f"taylor_terms given is {taylor_terms}, but must be 'linear', 'non_linear' or 'squared'")
#
#     final_answer = (coef_rh * delta_rh + coef_temp * delta_temp + coef_non_linear * delta_rh * delta_temp +
#                     coef_temp_squared * delta_temp ** 2) / 1000
#     info_dict = {'rh': [coef_rh / 1000, delta_rh], 'temp': [coef_temp / 1000, delta_temp],
#                  'non_linear': [coef_non_linear / 1000, delta_temp * delta_rh],
#                  'temp_squared': [coef_temp_squared / 1000, delta_temp ** 2]}
#     return final_answer, info_dict


def get_delta_temp_quant_theory(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                temp_ft_mean: Optional[np.ndarray] = None, temp_ft_quant: Optional[np.ndarray] = None,
                                z_ft_mean: Optional[np.ndarray] = None, z_ft_quant: Optional[np.ndarray] = None,
                                taylor_terms_delta_mse_mod_anom: str = 'linear',
                                taylor_terms_delta_mse_mod: str = 'linear', rh_option: str = 'full',
                                sphum_option: str = 'full') -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Returns a theoretical prediction for change in a given percentile, $x$, of near-surface temperature. In the simplest
    case with `taylor_terms_delta_mse_mod_anom = 'linear'`, `taylor_terms_delta_mse_mod='linear'`,
    `rh_option = 'approx_anomaly'` and `sphum_option = 'approx'`, the equation for the theory is:

    $$\delta T_s(x) \\approx \\left(\\frac{\overline{\\beta_{s1}}}{\\beta_{s1}(x)} +
    \\frac{\\beta_{A2}}{\\beta_{A1}} \\frac{\Delta T_A(x)}{\overline{T_A}} \\right) \delta \overline{T_s} -
    \\frac{L_v \overline{q_s^*}}{\overline{\\beta_{s1}}}\delta \Delta r_s(x) +
    \\frac{\\beta_{A1}}{\overline{\\beta_{s1}}} \delta \Delta T_A(x)$$

    If `z_quant` is given (with other parameters the same), then the simplest $z$ form of the theory will be computed:

    $$
    \\begin{align}
    \\begin{split}
    \delta T_s(x) & \\approx \\left(\\frac{\overline{\\beta_{s1}}+\\beta_{A1}}{\\beta_{s1}(x)+\\beta_{A1}} +
    \\frac{\\beta_{A2}}{\\beta_{A1}} \\frac{\overline{\\beta_{s1}}}{\overline{\\beta_{s1}}+\\beta_{A1}}
    \\frac{\Delta T_A(x)}{\overline{T_A}}\\right) \delta \overline{T_s} \\\\
    & - \\frac{L_v \overline{q_s^*}}{\overline{\\beta_{s1}}+\\beta_{A1}}\delta \Delta r_s(x)
    + \\frac{\\beta_{A1}}{\overline{\\beta_{s1}}+\\beta_{A1}} \delta \Delta T_A'(x)
    \\end{split}
    \\end{align}
    $$

    A simpler theory with $\\Delta T_A = \\delta \\Delta T_A = 0$, is also returned.

    Terms in equation:
        * $\\beta_{s1}(x) = \\frac{dh^{\dagger}_s(x)}{dT_s(x)} = c_p - R^{\dagger} + L_v \\alpha_s(x)q_s(x)$
        * $\\beta_{A1} = \\frac{d\\overline{h_s^{\\dagger}}}{d\overline{T_A}} = c_p + R^{\dagger} + L_v \\alpha_A q_A^*$
        * $\\beta_{A2} = \overline{T_A} \\frac{d^2\\overline{h_s^{\\dagger}}}{d\overline{T_A}^2} =
         \overline{T_A}\\frac{d\\beta_1}{d\overline{T_A}} = L_v \\alpha_A q_A^*(\\alpha_A \\overline{T_A} - 2)$
        * $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\Delta T_A = T_A(x) - \overline{T_A} = \overline{T_{CE}} - T_{CE}(x) + \Delta T_{FT}(x)$
        where $T_A$ is the adiabatic temperature at $p_{FT}$.
        * $\Delta T_A' = \overline{T_{CE}} - T_{CE}(x) + \\frac{g}{R^{\dagger}} \Delta z_{FT}(x)$ is used for the $z$
        theory to replace $\Delta T_{FT}(x)$.
        * All terms in $\\beta_A$ are evaluated at $\overline{T_A}$. I.e. $q_A^* = q^*(\overline{T_A}, p_{FT})$, while
        all terms in $\\beta_s$ are evaluated at the surface i.e. $\\alpha_s(x) = \\alpha(T_s(x), p_s)$.

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        temp_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        z_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average geopotential height at `pressure_ft` in *m*.
        z_ft_quant: IF GIVEN, WILL RETURN $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `z_ft_quant[i, j]` is geopotential height at `pressure_ft`, averaged over all days with near-surface
            temperature corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *m*.
        taylor_terms_delta_mse_mod_anom:
            How many taylor series terms to keep in the `get_delta_mse_mod_anom_theory` function to obtain approximation
            for change in modified moist static energy anomaly: $\\delta \\Delta h^{\\dagger}$.
        taylor_terms_delta_mse_mod:
            How many taylor series terms to keep in the expansions for changes in modified moist static energy:
            $\\delta h^{\\dagger}(x)$ and $\\delta \overline{h^{\\dagger}}$ ($\\alpha$ and $q^*$ are both evaluated
            at the surface i.e. $\\alpha(T_s, p_s)$).

            * `linear`: $\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha q)\\delta T_s +
                L_v q^* \\delta r$
            * `squared`: Includes the additional term $0.5 L_v \\alpha q (\\alpha - 2 / T_s) \\delta T_s^2$
        rh_option:
            Relative humidity changes, $\delta r$, provide a secondary effect, so we give a number of options to
            approximate them:

            * `full`: No approximations to $\delta r$ terms.
            * `approx`: Neglect all $\\Delta T_A \delta r$ terms, in essence non-linear terms.
            * `none`: Set all relative humidity changes to zero.
            * If `anomaly` is also included e.g. `full_anomaly` or `approx_anomaly`, the $\delta r(x)$ term will be
                multiplied by $\overline{q^*}$ rather than $q^*(x)$. This then gives a $\delta (r(x) - \overline{r})$
                term rather than two separate relative humidity terms.
        sphum_option:
            Only relavent if `taylor_terms_delta_mse_mod = 'linear`. Specific humidity as function of quantile, $q(x)$
            appears in denominator due to $\\beta_{s1}(x)$ term. Below, give options to replace this with
            $\overline{\\beta_{s1}}$ in some cases to remove $x$ dependence.

            * `full`: No approximations to $\\beta_{s1}(x)$ terms.
            * `approx`: Keep $\\beta_{s1}(x)$ as denominator for the primary $\delta \overline{T_s}$ term, but
                replace with $\overline{\\beta_{s1}}$ elsewhere (where $\delta r$, $\Delta T_A$ or
                $\delta \Delta T_A$ appear).
    Returns:
        `delta_temp_quant`: `float [n_quant]`</br>
            `delta_temp_quant[i]` refers to the theoretical temperature difference between experiments
            for percentile `quant_use[j]`.
        `delta_temp_quant_old`: `float [n_quant]`</br>
            Theoretical temperature difference using old theory which assumes $\\Delta T_A = \\delta \\Delta T_A = 0$.
        `info_coef`: Dictionary with 13 keys for each term in the theory: `temp_s_mean`, `r_mean`,
            `r_quant`, `temp_s_mean_squared`, `temp_a`, `temp_s_mean_temp_a0`, `r_mean_temp_a0`,
            `temp_s_mean_squared_temp_a0`, `temp_s_mean_cubed_temp_a0`, `r_mean_squared_temp_a0`,
            `nl_temp_s_mean_temp_a`, `nl_r_mean_temp_a` and `nl_temp_s_mean_r_mean`.</br>
            This gives the prefactor for the term indicated such that `info_coef[var]` $\\times$ `info_change[var]`
            is the contribution for that term. Sum of all contributions equals `delta_temp_quant`
            (only if `taylor_terms_delta_mse_mod='linear'`).</br>
            Terms with `temp_a0` have the $\Delta T_A$ term in the prefactor. I isolate them from the terms independent
            of adiabatic temperature anomaly but `info_change['temp_s_mean'] = info_change['temp_s_mean_temp_a0']`.</br>
            `temp_a` is the $\delta \Delta T_A$ term and `nl` refers to non-linear terms.
        `info_change`: Complementary dictionary to `info_coef` with same keys that gives the relavent change to a
            quantity. I.e. the $\delta$ term so
            `info_change['nl_temp_s_mean_temp_a']` $=\delta \overline{T_s} \Delta T_A$.
    """
    # Have equation delta_mse_mod(x) = delta_mse_mod_mean + delta_mse_mod_anom(x)
    # Rearrange and gather terms such that:
    # coef_temp_quant_squared * delta_temp_quant**2 + beta_s1 * delta_temp_quant = sum(info_coef * info_change)
    info_coef = {key: 0 for key in ['temp_s_mean', 'r_mean', 'r_quant', 'temp_s_mean_squared', 'temp_a',
                                    'temp_s_mean_temp_a0', 'r_mean_temp_a0', 'temp_s_mean_squared_temp_a0',
                                    'temp_s_mean_cubed_temp_a0', 'r_mean_squared_temp_a0', 'nl_temp_s_mean_temp_a',
                                    'nl_r_mean_temp_a', 'nl_temp_s_mean_r_mean']}
    info_change = copy.deepcopy(info_coef)

    # LHS is due to delta_mse_mod(x), shift r(x) term to RHS as independent of temperature
    info_taylor_quant = \
        do_delta_mse_mod_taylor_expansion(temp_surf_quant, sphum_quant, pressure_surf,
                                          pressure_ft, taylor_terms_delta_mse_mod,
                                          temp_use_rh_term=temp_surf_mean[0] if 'anomaly' in rh_option else None)[1]
    for key in info_taylor_quant:
        info_taylor_quant[key][0] = info_taylor_quant[key][0] * 1000  # turn coefficient units into J/kg
    beta_s1 = info_taylor_quant['temp'][0]
    coef_temp_quant_squared = info_taylor_quant['temp_squared'][0]
    info_coef['r_quant'] = -info_taylor_quant['rh'][0]
    info_change['r_quant'] = np.zeros_like(info_coef['r_quant']) if 'none' in rh_option else info_taylor_quant['rh'][1]

    # First term on RHS is delta_mse_mod_mean term - taylor expansion in terms of surface quantities
    # No adiabatic temperature at all.
    info_taylor_mean = do_delta_mse_mod_taylor_expansion(temp_surf_mean, sphum_mean, pressure_surf, pressure_ft,
                                                         taylor_terms_delta_mse_mod)[1]
    for key in info_taylor_mean:
        info_taylor_mean[key][0] = info_taylor_mean[key][0] * 1000  # turn coefficient units into J/kg
    beta_s1_mean = info_taylor_mean['temp'][0]
    info_coef['temp_s_mean'] = beta_s1_mean
    info_change['temp_s_mean'] = info_taylor_mean['temp'][1]
    info_coef['r_mean'] = info_taylor_mean['rh'][0]
    info_change['r_mean'] = 0 if 'none' in rh_option else info_taylor_mean['rh'][1]
    info_coef['temp_s_mean_squared'] = info_taylor_mean['temp_squared'][0]
    info_change['temp_s_mean_squared'] = info_taylor_mean['temp_squared'][1]

    # MSE_MOD_ANOMALY i.e. adiabatic temp stuff - start
    # Other terms on RHS come from delta_mse_mod_anom(x) expansion in terms of adiabatic temperature anomaly (temp_a)
    # changes and mse_mod_mean changes.
    # mse_mod_mean changes here have a climatological adiabatic temperature anomaly (temp_a0) in the coefficient
    # so are considered separately to the mse_mod_mean term above.
    # Need to further decompose mse_mod_mean changes into temp_s_mean and rh contributions using expansion given by
    # info_taylor_mean. Hence, each coefficient is two terms multiplied together
    _, info_mse_mod_anom, temp_adiabat_anom = get_delta_mse_mod_anom_theory(temp_surf_mean, temp_surf_quant, sphum_mean,
                                                                            sphum_quant, pressure_surf, pressure_ft,
                                                                            taylor_terms_delta_mse_mod_anom)
    for key in info_mse_mod_anom:
        info_mse_mod_anom[key][0] = info_mse_mod_anom[key][0] * 1000  # turn coefficient units into J/kg
    beta_a1 = info_mse_mod_anom['temp_adiabat_anom'][0]
    info_coef['temp_a'] = beta_a1
    info_change['temp_a'] = info_mse_mod_anom['temp_adiabat_anom'][1]

    info_coef['temp_s_mean_temp_a0'] = info_mse_mod_anom['mse_mod_mean'][0] * beta_s1_mean
    info_change['temp_s_mean_temp_a0'] = info_change['temp_s_mean']

    # Squared term on RHS come from coef_mse_mod_mean_squared * delta_mse_mod_mean**2
    # So in coefficient, need to combine temperature or rh coefficient in delta_mse_mod_mean**2 with
    # coef_mse_mod_mean_squared. Similar with cubed.
    # delta_mse_mod_mean has squared terms as well, so need to account for those here - hence addition.
    # Neglect the non-linear delta_temp_s_mean x delta_r_mean terms
    info_coef['temp_s_mean_squared_temp_a0'] = info_mse_mod_anom['mse_mod_mean_squared'][0] * beta_s1_mean ** 2 + \
                                               info_mse_mod_anom['mse_mod_mean'][0] * info_coef['temp_s_mean_squared']
    info_change['temp_s_mean_squared_temp_a0'] = info_change['temp_s_mean'] ** 2
    info_coef['temp_s_mean_cubed_temp_a0'] = info_mse_mod_anom['mse_mod_mean_cubed'][0] * beta_s1_mean ** 3 + \
                                             2 * info_mse_mod_anom['mse_mod_mean_squared'][0] * beta_s1_mean * \
                                             info_coef['temp_s_mean_squared']
    info_change['temp_s_mean_cubed_temp_a0'] = info_change['temp_s_mean'] ** 3

    coef_mse_mod_mean_temp_a = info_mse_mod_anom['non_linear'][0]
    info_coef['nl_temp_s_mean_temp_a'] = coef_mse_mod_mean_temp_a * beta_s1_mean
    info_change['nl_temp_s_mean_temp_a'] = info_change['temp_s_mean'] * info_change['temp_a']

    # RH contribution to delta_mse_mod_anom
    if 'full' in rh_option:
        info_coef['r_mean_temp_a0'] = info_mse_mod_anom['mse_mod_mean'][0] * info_coef['r_mean']
        info_coef['r_mean_squared_temp_a0'] = info_mse_mod_anom['mse_mod_mean_squared'][0] * info_coef['r_mean'] ** 2
        info_coef['nl_r_mean_temp_a'] = coef_mse_mod_mean_temp_a * info_coef['r_mean']
        info_coef['nl_temp_s_mean_r_mean'] = info_mse_mod_anom['mse_mod_mean_squared'][0] * \
                                             2 * beta_s1_mean * info_coef['r_mean']
    else:
        # Neglect rh terms when multiplied by a temp_a0 term, as very small.
        info_coef['r_mean_temp_a0'] = 0
        info_coef['r_mean_squared_temp_a0'] = 0
        info_coef['nl_r_mean_temp_a'] = 0
        info_coef['nl_temp_s_mean_r_mean'] = 0
    info_change['r_mean_temp_a0'] = info_change['r_mean']
    info_change['r_mean_squared_temp_a0'] = info_change['r_mean'] ** 2
    info_change['nl_r_mean_temp_a'] = info_change['r_mean'] * info_change['temp_a']
    info_change['nl_temp_s_mean_r_mean'] = info_change['temp_s_mean'] * info_change['r_mean']
    # MSE_MOD_ANOMALY - end

    # what to divide info_coef by - only used if taylor_terms_delta_mse_mod is linear.
    if sphum_option == 'full':
        beta_s1_use = {key: beta_s1 for key in info_change}
    elif sphum_option == 'approx':
        beta_s1_use = {key: beta_s1_mean for key in info_change}
        beta_s1_use['temp_s_mean'] = beta_s1
    else:
        raise ValueError(f'sphum_option needs to be full or approx but was {sphum_option}')

    if z_ft_quant is not None:
        # If provide z, will compute z version of the theory - replace temp_ft anomaly with z_ft anomaly
        # in delta temp_adiabat_anom term.
        temp_ce_mean, temp_ce_quant = \
            decompose_temp_adiabat_anomaly(temp_surf_mean, temp_surf_quant, sphum_mean,
                                           sphum_quant, temp_ft_mean, temp_ft_quant, pressure_surf, pressure_ft)[1:3]
        delta_temp_ce_mean = temp_ce_mean[1] - temp_ce_mean[0]
        delta_temp_ce_quant = temp_ce_quant[1] - temp_ce_quant[0]
        z_ft_anom = z_ft_quant - z_ft_mean[:, np.newaxis]
        delta_z_ft_anom = z_ft_anom[1] - z_ft_anom[0]
        R_mod = R * np.log(pressure_surf / pressure_ft) / 2
        info_change['temp_a'] = delta_temp_ce_mean - delta_temp_ce_quant + g / R_mod * delta_z_ft_anom
        # extra delta_temp_s(x) term in z form means extra term in denominator
        for var in beta_s1_use:
            beta_s1_use[var] = beta_s1_use[var] + beta_a1
        info_coef['temp_s_mean'] += beta_a1  # extra delta_temp_s_mean term in z form

    # Convert info_coef into temperature form by dividing by denominator
    if taylor_terms_delta_mse_mod == 'squared':
        # If squared term, don't use beta_s1_use
        # Divide each term in equation by beta_s1
        for var in info_coef:
            info_coef[var] = info_coef[var] / beta_s1
        coef_temp_quant_squared = coef_temp_quant_squared / beta_s1
        coef_temp_quant = 1
    else:
        for var in info_coef:
            info_coef[var] = info_coef[var] / beta_s1_use[var]

    final_answer = {}
    for key in ['full', 'old']:
        if taylor_terms_delta_mse_mod == 'squared':
            if key == 'old':
                # have to use sum not np.sum for sum of list like this
                coef_rhs = sum([info_coef[var] * info_change[var] for var in ['temp_s_mean', 'r_mean', 'r_quant',
                                                                              'temp_s_mean_squared']])
            else:
                coef_rhs = sum([info_coef[var] * info_change[var] for var in info_coef])
            # Solve quadratic equation, taking the positive solution
            final_answer[key] = (-coef_temp_quant + np.sqrt(coef_temp_quant ** 2 - 4 * coef_temp_quant_squared *
                                                            (-coef_rhs))) / (2 * coef_temp_quant_squared)
        else:
            if key == 'old':
                final_answer[key] = sum([info_coef[var] * info_change[var] for var in
                                         ['temp_s_mean', 'r_mean', 'r_quant', 'temp_s_mean_squared']])
            else:
                final_answer[key] = sum([info_coef[var] * info_change[var] for var in info_coef])

    return final_answer['full'], final_answer['old'], info_coef, info_change


def get_delta_temp_quant_theory_simple(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                       sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                       temp_ft_mean: Optional[np.ndarray] = None,
                                       temp_ft_quant: Optional[np.ndarray] = None,
                                       z_ft_mean: Optional[np.ndarray] = None, z_ft_quant: Optional[np.ndarray] = None,
                                       ignore_rh: bool = False) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    This performs the same calculation as `get_delta_temp_quant_theory` with the following parameters, i.e. the simplest
    form:

    * `taylor_terms_delta_mse_mod_anom = 'linear'`
    * `taylor_terms_delta_mse_mod = 'linear'`
    * `rh_option = 'none' if ignore_rh else 'approx_anomaly'`
    * `sphum_option = 'approx'`

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        temp_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        z_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average geopotential height at `pressure_ft` in *m*.
        z_ft_quant: IF GIVEN, WILL RETURN $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `z_ft_quant[i, j]` is geopotential height at `pressure_ft`, averaged over all days with near-surface
            temperature corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *m*.
        ignore_rh: If `True`, will set $\delta r_s(x) = \delta \overline{r_s} = 0$.
    Returns:
        `delta_temp_quant`: `float [n_quant]`</br>
            `delta_temp_quant[i]` refers to the theoretical temperature difference between experiments
            for percentile `quant_use[j]`.
        `delta_temp_quant_old`: `float [n_quant]`</br>
            Theoretical temperature difference using old theory which assumes $\Delta T_A = \delta \Delta T_A = 0$.
        `info_coef`: Dictionary with 5 keys for each term in the simple version of the theory: `temp_s_mean`, `r_mean`,
            `r_quant`, `temp_a` and `temp_s_mean_temp_a0`.</br>
            This gives the prefactor for the term indicated such that `info_coef[var]` $\\times$ `info_change[var]`
            is the contribution for that term. Sum of all contributions equals `delta_temp_quant`.</br>
            `temp_s_mean_temp_a0` has the $\Delta T_A$ term in the prefactor. I isolate it from the `temp_s_mean` term
             which is independent of adiabatic temperature anomaly but
             `info_change['temp_s_mean'] = info_change['temp_s_mean_temp_a0']`.</br>
            `temp_a` is the $\delta \Delta T_A$ term.
        `info_change`: Complementary dictionary to `info_coef` with same keys that gives the relavent change to a
            quantity. I.e. the $\delta$ term so `info_change['r_quant']` $=\delta r_s(x)$.
    """
    # Compute adiabatic temperatures
    n_exp, n_quant = temp_surf_quant.shape
    temp_adiabat_mean = np.zeros_like(temp_surf_mean)
    temp_adiabat_quant = np.zeros_like(temp_surf_quant)
    for i in range(n_exp):
        temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)
        for j in range(n_quant):
            temp_adiabat_quant[i, j] = get_temp_adiabat(temp_surf_quant[i, j], sphum_quant[i, j], pressure_surf,
                                                        pressure_ft)
    temp_adiabat_anom = temp_adiabat_quant - temp_adiabat_mean[:, np.newaxis]

    # Compute relative humidities
    r_mean = sphum_mean / sphum_sat(temp_surf_mean, pressure_surf)
    r_quant = sphum_quant / sphum_sat(temp_surf_quant, pressure_surf)

    # Get parameters required for prefactors in the theory
    R_mod, _, _, beta_a1, beta_a2, _ = get_theory_prefactor_terms(temp_adiabat_mean[0], pressure_surf, pressure_ft)
    beta_s1 = get_theory_prefactor_terms(temp_surf_quant[0], pressure_surf, pressure_ft, sphum_quant[0])[3]
    _, q_sat_surf_mean, _, beta_s1_mean, _, _ = get_theory_prefactor_terms(temp_surf_mean[0], pressure_surf,
                                                                           pressure_ft, sphum_mean[0])

    # Record numerator of coefficients
    info_coef = {'temp_s_mean': beta_s1_mean, 'r_quant': 0 if ignore_rh else -L_v * q_sat_surf_mean,
                 'temp_a': beta_a1,
                 'temp_s_mean_temp_a0': beta_a2 / beta_a1 * beta_s1_mean * temp_adiabat_anom[0] / temp_adiabat_mean[0]}
    # theory has rh anomaly term so coefficient for mean is same but negative of rh_quant
    info_coef['r_mean'] = -info_coef['r_quant']

    # Record denominator of coefficients
    beta_s1_use = {key: beta_s1_mean for key in info_coef}
    beta_s1_use['temp_s_mean'] = beta_s1  # only keep x dependence for leading temp_s_mean term

    if z_ft_quant is not None:
        # If provide z, will compute z version of the theory - replace temp_ft anomaly with z_ft anomaly
        # in delta temp_adiabat_anom term.
        temp_ce_mean, temp_ce_quant = \
            decompose_temp_adiabat_anomaly(temp_surf_mean, temp_surf_quant, sphum_mean,
                                           sphum_quant, temp_ft_mean, temp_ft_quant, pressure_surf, pressure_ft)[1:3]
        delta_temp_ce_mean = temp_ce_mean[1] - temp_ce_mean[0]
        delta_temp_ce_quant = temp_ce_quant[1] - temp_ce_quant[0]
        z_ft_anom = z_ft_quant - z_ft_mean[:, np.newaxis]
        delta_z_ft_anom = z_ft_anom[1] - z_ft_anom[0]
        # write adiabatic temp anomaly change in terms of z anomaly rather than free troposphere temp anomaly
        delta_temp_adiabat_anom = delta_temp_ce_mean - delta_temp_ce_quant + g / R_mod * delta_z_ft_anom
        info_coef['temp_s_mean'] += beta_a1  # extra delta_temp_s_mean term in z form
        for var in beta_s1_use:
            beta_s1_use[var] = beta_s1_use[var] + beta_a1  # extra term in denominator in z form
    else:
        delta_temp_adiabat_anom = temp_adiabat_anom[1] - temp_adiabat_anom[0]

    # Get full coefficients by dividing numerator by denominator
    for var in info_coef:
        info_coef[var] = info_coef[var] / beta_s1_use[var]

    info_change = {'temp_s_mean': temp_surf_mean[1] - temp_surf_mean[0],
                   'r_mean': 0 if ignore_rh else r_mean[1] - r_mean[0],
                   'r_quant': 0 if ignore_rh else r_quant[1] - r_quant[0],
                   'temp_a': delta_temp_adiabat_anom}
    info_change['temp_s_mean_temp_a0'] = info_change['temp_s_mean']

    final_answer = {'full': sum([info_coef[var] * info_change[var] for var in info_coef]),
                    'old': sum([info_coef[var] * info_change[var] for var in ['temp_s_mean', 'r_mean', 'r_quant']])}
    return final_answer['full'], final_answer['old'], info_coef, info_change


def get_delta_temp_quant_theory_final(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                      sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                      temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray, z_ft_mean: np.ndarray,
                                      z_ft_quant: np.ndarray, z_form: bool = False, epsilon_form: bool = True,
                                      ignore_rh: bool = False, include_squared_terms: bool = False) -> Tuple[
    np.ndarray, dict, dict, np.ndarray]:
    """
    Calculates the theoretical near-surface temperature change for percentile $x$, $\delta T_s(x)$,
    such that in the simplest linear case (`include_squared_terms=False`) with `epsilon_form=True`, we have:

    $$
    \\begin{align}
    \\begin{split}
    \delta T_s(x) \\approx
    1 &+ \gamma_{T_s}\\frac{\Delta T_s(x)}{\overline{T_s}} \delta \overline{T_s}+
    \gamma_{r_s} \\frac{\Delta r_s(x)}{\overline{r_s}} \delta \overline{T_s}+
    \gamma_{\epsilon} \Delta \epsilon(x) \delta \overline{T_s} \\\\
    &+ \gamma_{\delta r_s} \\frac{\delta \Delta r_s(x)}{\overline{r_s}} +
    \gamma_{\delta T_{FT}} \delta \Delta T_{FT}(x) + \gamma_{\delta \epsilon} \delta \Delta \epsilon(x)
    \\end{split}
    \\end{align}
    $$

    If `include_squared_terms=True`, then non-linear and squared terms are included. The LHS becomes
    $\\left[1+\mu(x)\\frac{\delta r_s(x)}{\overline{r_s}}\\right]\delta T_s(x)$
    where $\\mu(x) = \\frac{L_v \\alpha_s(x) q_s(x)}{\\beta_{s1}(x)}$, and extra terms e.g.
    $\gamma_{T_s^2}(\\frac{\Delta T_s}{\overline{T_s}})^2 \delta \overline{T_s}$ are included on the RHS.

    If `z_quant` is given (with other parameters the same), then the $z$ form of the theory will be computed
    (not possible with `include_squared_terms=True` because gets too complicated):

    $$
    \\begin{align}
    \\begin{split}
    \delta T_s(x) \\approx
    1 + \\frac{\overline{\\beta_{s1}}}{\overline{\\beta_{s1}} + \\beta_{A1}} \\bigg(
    &\gamma_{T_s}\\frac{\Delta T_s(x)}{\overline{T_s}} \delta \overline{T_s}+
    \gamma_{r_s} \\frac{\Delta r_s(x)}{\overline{r_s}} \delta \overline{T_s}+
    \gamma_{\epsilon} \Delta \epsilon(x) \delta \overline{T_s} \\\\
    &+ \gamma_{\delta r_s} \\frac{\delta \Delta r_s(x)}{\overline{r_s}} +
    \gamma_{\delta T_{FT}} \\frac{g}{R^{\dagger}} \delta \Delta z_{FT}(x) +
    \gamma_{\delta \epsilon} \delta \Delta \epsilon(x) \\bigg)
    \\end{split}
    \\end{align}
    $$

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        temp_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        z_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average geopotential height at `pressure_ft` in *m*.
        z_ft_quant: IF GIVEN, WILL RETURN $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `z_ft_quant[i, j]` is geopotential height at `pressure_ft`, averaged over all days with near-surface
            temperature corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *m*.
        z_form: If `True`, will return $z$ version of theory.
        epsilon_form: If `True`, will quantify deviation from convective equilibrium through moist static energy:
            $\\epsilon = h_s - h_{FT}^*$. Otherwise, will quantify it in temperature space through
            $T_{CE} = T_{FT} - T_A$.
        ignore_rh: If `True`, will set $\delta r_s(x) = \delta \overline{r_s} = 0$.
        include_squared_terms: If `True`, will include $\Delta T_s^2\delta \overline{T_s}$,
            $\Delta r_s^2\delta \overline{T_s}$, $\Delta T_s \Delta r_s \delta \overline{T_s}$,
            $\Delta T_s^2 \Delta r_s \delta \overline{T_s}$, $\Delta T_s \Delta r_s^2 \delta \overline{T_s}$,
            $\Delta T_s \delta \overline{r_s}$, $\Delta T_s \delta \Delta r_s$,
            $\delta \overline{r_s} \delta \overline{T_s}$ and $\delta \Delta T_A \delta \overline{T_s}$ terms in theory.
    Returns:
        `delta_temp_quant`: `float [n_quant]`</br>
            `delta_temp_quant[i]` refers to the theoretical temperature difference between experiments
            for percentile `quant_use[j]`.
        `info_coef`: Dictionary with 4 keys for each term in the simple version of the theory: `temp_s`, `humidity`,
            `r_change`, `temp_a_change`. The key refers to the variable that causes the variation with $x$.</br>
            This gives the prefactor for the term indicated such that `info_coef[var]` $\\times$ `info_change[var]`
            is the contribution for that term. Sum of all contributions equals $\delta T_s(x)-\delta \overline{T_s}$.
        `info_change`: Complementary dictionary to `info_coef` with same keys that gives the relavent change to a
            quantity i.e. the $\delta$ term. For both `temp_s` and `sphum`, this is $\delta \overline{T_s}$.
        `mu`: `float [n_quant]`</br>
            $\mu(x)$ factor. Will be all zeros if `include_squared_terms=False`.
            Otherwise, will be $\\mu(x) = \\frac{L_v \\alpha_s(x) q_s(x)}{\\beta_{s1}(x)}$.
    """
    # Compute adiabatic temperatures
    n_exp, n_quant = temp_surf_quant.shape
    temp_adiabat_mean = np.zeros_like(temp_surf_mean)
    for i in range(n_exp):
        temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)

    # Compute relative humidities
    r_mean = sphum_mean / sphum_sat(temp_surf_mean, pressure_surf)
    r_quant = sphum_quant / sphum_sat(temp_surf_quant, pressure_surf)
    r_anom = r_quant - r_mean[:, np.newaxis]

    # Compute epsilon
    if epsilon_form:
        # Quantify deviation from convective equilibrium in MSE space
        epsilon_mean = moist_static_energy(temp_surf_mean, sphum_mean, height=0) - \
                       moist_static_energy(temp_ft_mean, sphum_sat(temp_ft_mean, pressure_ft), z_ft_mean)
        epsilon_quant = moist_static_energy(temp_surf_quant, sphum_quant, height=0) - \
                        moist_static_energy(temp_ft_quant, sphum_sat(temp_ft_quant, pressure_ft), z_ft_quant)
        conv_anom = (epsilon_quant - epsilon_mean[:, np.newaxis]) * 1000
    else:
        # Quantify deviation from convective equilibrium in temperature space
        temp_ce_mean, temp_ce_quant = \
            decompose_temp_adiabat_anomaly(temp_surf_mean, temp_surf_quant, sphum_mean,
                                           sphum_quant, temp_ft_mean, temp_ft_quant, pressure_surf, pressure_ft)[1:3]
        conv_anom = temp_ce_quant - temp_ce_mean[:, np.newaxis]

    temp_surf_anom_norm0 = (temp_surf_quant - temp_surf_mean[:, np.newaxis])[0] / temp_surf_mean[0]
    r_anom_norm0 = r_anom[0] / r_mean[0]

    # Record coefficients of each term in equation for delta T_s(x)
    # label is anomaly that causes variation with x.
    gamma_temp_s, gamma_humidity, gamma_conv, gamma_r_change, gamma_temp_ft_change, gamma_conv_change, \
        gamma_temp_s_squared, gamma_humidity_squared, gamma_temp_s_humidity, gamma_temp_s_squared_humidity, \
        gamma_temp_s_humidity_squared, gamma_temp_s_r_mean_change, gamma_temp_s_r_change, \
        gamma_r_mean_change_temp_mean_change, gamma_temp_ft_change_temp_mean_change, \
        gamma_conv_change_temp_mean_change, gamma_conv_temp_mean_change_squared = \
        get_gamma_factors(temp_surf_mean[0], sphum_mean[0], pressure_surf, pressure_ft, epsilon_form)

    info_coef = {'temp_s': gamma_temp_s * temp_surf_anom_norm0,
                 'humidity': gamma_humidity * r_anom_norm0,
                 'conv': gamma_conv * conv_anom[0],             # will be zero if epsilon_form=False
                 'r_change': 0 if ignore_rh else gamma_r_change / r_mean[0],
                 'temp_ft_change': gamma_temp_ft_change,
                 'conv_change': gamma_conv_change}

    if include_squared_terms:
        info_coef['temp_s_squared'] = gamma_temp_s_squared * temp_surf_anom_norm0**2
        info_coef['humidity_squared'] = gamma_humidity_squared * r_anom_norm0**2
        info_coef['temp_s_humidity'] = gamma_temp_s_humidity * temp_surf_anom_norm0 * r_anom_norm0
        info_coef['temp_s_squared_humidity'] = gamma_temp_s_squared_humidity * temp_surf_anom_norm0**2 * r_anom_norm0
        info_coef['temp_s_humidity_squared'] = gamma_temp_s_humidity_squared * temp_surf_anom_norm0 * r_anom_norm0**2

        info_coef['temp_s_r_mean_change'] = gamma_temp_s_r_mean_change * temp_surf_anom_norm0 / r_mean[0]
        info_coef['temp_s_r_change'] = gamma_temp_s_r_change * temp_surf_anom_norm0 / r_mean[0]

        info_coef['r_mean_change_temp_mean_change'] = gamma_r_mean_change_temp_mean_change / r_mean[0]
        info_coef['temp_ft_change_temp_mean_change'] = gamma_temp_ft_change_temp_mean_change
        info_coef['conv_change_temp_mean_change'] = gamma_conv_change_temp_mean_change
        # below will be zero if epsilon_form=False
        info_coef['conv_temp_mean_change_squared'] = gamma_conv_temp_mean_change_squared * conv_anom[0]

    if z_form:
        if include_squared_terms:
            raise ValueError('Conversion to z form of theory is too complicated with squared terms')
        # Get parameters required for conversion to z form of theory
        R_mod, _, _, beta_a1, _, _ = get_theory_prefactor_terms(temp_adiabat_mean[0], pressure_surf, pressure_ft)
        beta_s1_mean = get_theory_prefactor_terms(temp_surf_mean[0], pressure_surf, pressure_ft, sphum_mean[0])[3]

        # If provide z, will compute z version of the theory - replace temp_ft anomaly with z_ft anomaly
        z_ft_anom = z_ft_quant - z_ft_mean[:, np.newaxis]
        delta_z_ft_anom = z_ft_anom[1] - z_ft_anom[0]
        # write adiabatic temp anomaly change in terms of z anomaly rather than free troposphere temp anomaly
        delta_temp_ft_anom = g / R_mod * delta_z_ft_anom

        # In z-form, need to multiply each term by constant prefactor
        for var in info_coef:
            info_coef[var] = info_coef[var] * beta_s1_mean / (beta_s1_mean + beta_a1)
    else:
        temp_ft_anom = temp_ft_quant - temp_ft_mean[:, np.newaxis]
        delta_temp_ft_anom = temp_ft_anom[1] - temp_ft_anom[0]

    info_change = {'temp_s': temp_surf_mean[1] - temp_surf_mean[0],
                   'humidity': temp_surf_mean[1] - temp_surf_mean[0],
                   'conv': temp_surf_mean[1] - temp_surf_mean[0],
                   'r_change': 0 if ignore_rh else r_anom[1] - r_anom[0],
                   'temp_ft_change': delta_temp_ft_anom,
                   'conv_change': conv_anom[1] - conv_anom[0]}
    if include_squared_terms:
        for var in ['temp_s_squared', 'humidity_squared', 'temp_s_humidity', 'temp_s_squared_humidity',
                    'temp_s_humidity_squared']:
            info_change[var] = temp_surf_mean[1] - temp_surf_mean[0]
        info_change['temp_s_r_mean_change'] = 0 if ignore_rh else r_mean[1] - r_mean[0]
        info_change['temp_s_r_change'] = 0 if ignore_rh else r_anom[1] - r_anom[0]
        info_change['r_mean_change_temp_mean_change'] = (r_mean[1] - r_mean[0]) * (temp_surf_mean[1] -
                                                                                   temp_surf_mean[0])
        info_change['temp_ft_change_temp_mean_change'] = delta_temp_ft_anom * (temp_surf_mean[1] - temp_surf_mean[0])
        info_change['conv_change_temp_mean_change'] = (conv_anom[1] - conv_anom[0]) * (
                temp_surf_mean[1] - temp_surf_mean[0])
        info_change['conv_temp_mean_change_squared'] = (temp_surf_mean[1] - temp_surf_mean[0])**2

        _, _, alpha_s_x, beta_s1_x, _, _ = get_theory_prefactor_terms(temp_surf_quant[0], pressure_surf, pressure_ft,
                                                                      sphum_quant[0])
        mu_factor = L_v * alpha_s_x * sphum_quant[0] / beta_s1_x * (r_quant[1] - r_quant[0]) / r_quant[0]
    else:
        mu_factor = np.zeros(n_quant)
    final_answer = temp_surf_mean[1] - temp_surf_mean[0] + sum([info_coef[var] * info_change[var] for var in info_coef])
    final_answer = final_answer / (1+mu_factor)
    return final_answer, info_coef, info_change, mu_factor


def get_delta_temp_quant_theory_final2(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                       sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                       temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray, z_ft_mean: np.ndarray,
                                       z_ft_quant: np.ndarray, z_form: bool = False,
                                       gamma_from_temp_adiabat: bool = True,
                                       ignore_rh: bool = False, include_squared_terms: bool = False) -> Tuple[
    np.ndarray, dict, dict, np.ndarray]:
    """
    Calculates the theoretical near-surface temperature change for percentile $x$, $\delta T_s(x)$,
    such that in the simplest linear case (`include_squared_terms=False`) with `epsilon_form=True`, we have:

    $$
    \\begin{align}
    \\begin{split}
    \delta T_s(x) \\approx
    1 &+ \gamma_{T_s}\\frac{\Delta T_s(x)}{\overline{T_s}} \delta \overline{T_s}+
    \gamma_{r_s} \\frac{\Delta r_s(x)}{\overline{r_s}} \delta \overline{T_s}+
    \gamma_{\epsilon} \Delta \epsilon(x) \delta \overline{T_s} \\\\
    &+ \gamma_{\delta r_s} \\frac{\delta \Delta r_s(x)}{\overline{r_s}} +
    \gamma_{\delta T_{FT}} \delta \Delta T_{FT}(x) + \gamma_{\delta \epsilon} \delta \Delta \epsilon(x)
    \\end{split}
    \\end{align}
    $$

    If `include_squared_terms=True`, then non-linear and squared terms are included. The LHS becomes
    $\\left[1+\mu(x)\\frac{\delta r_s(x)}{\overline{r_s}}\\right]\delta T_s(x)$
    where $\\mu(x) = \\frac{L_v \\alpha_s(x) q_s(x)}{\\beta_{s1}(x)}$, and extra terms e.g.
    $\gamma_{T_s^2}(\\frac{\Delta T_s}{\overline{T_s}})^2 \delta \overline{T_s}$ are included on the RHS.

    If `z_quant` is given (with other parameters the same), then the $z$ form of the theory will be computed
    (not possible with `include_squared_terms=True` because gets too complicated):

    $$
    \\begin{align}
    \\begin{split}
    \delta T_s(x) \\approx
    1 + \\frac{\overline{\\beta_{s1}}}{\overline{\\beta_{s1}} + \\beta_{A1}} \\bigg(
    &\gamma_{T_s}\\frac{\Delta T_s(x)}{\overline{T_s}} \delta \overline{T_s}+
    \gamma_{r_s} \\frac{\Delta r_s(x)}{\overline{r_s}} \delta \overline{T_s}+
    \gamma_{\epsilon} \Delta \epsilon(x) \delta \overline{T_s} \\\\
    &+ \gamma_{\delta r_s} \\frac{\delta \Delta r_s(x)}{\overline{r_s}} +
    \gamma_{\delta T_{FT}} \\frac{g}{R^{\dagger}} \delta \Delta z_{FT}(x) +
    \gamma_{\delta \epsilon} \delta \Delta \epsilon(x) \\bigg)
    \\end{split}
    \\end{align}
    $$

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        temp_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        z_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average geopotential height at `pressure_ft` in *m*.
        z_ft_quant: IF GIVEN, WILL RETURN $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `z_ft_quant[i, j]` is geopotential height at `pressure_ft`, averaged over all days with near-surface
            temperature corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *m*.
        z_form: If `True`, will return $z$ version of theory.
        epsilon_form: If `True`, will quantify deviation from convective equilibrium through moist static energy:
            $\\epsilon = h_s - h_{FT}^*$. Otherwise, will quantify it in temperature space through
            $T_{CE} = T_{FT} - T_A$.
        ignore_rh: If `True`, will set $\delta r_s(x) = \delta \overline{r_s} = 0$.
        include_squared_terms: If `True`, will include $\Delta T_s^2\delta \overline{T_s}$,
            $\Delta r_s^2\delta \overline{T_s}$, $\Delta T_s \Delta r_s \delta \overline{T_s}$,
            $\Delta T_s^2 \Delta r_s \delta \overline{T_s}$, $\Delta T_s \Delta r_s^2 \delta \overline{T_s}$,
            $\Delta T_s \delta \overline{r_s}$, $\Delta T_s \delta \Delta r_s$,
            $\delta \overline{r_s} \delta \overline{T_s}$ and $\delta \Delta T_A \delta \overline{T_s}$ terms in theory.
    Returns:
        `delta_temp_quant`: `float [n_quant]`</br>
            `delta_temp_quant[i]` refers to the theoretical temperature difference between experiments
            for percentile `quant_use[j]`.
        `info_coef`: Dictionary with 4 keys for each term in the simple version of the theory: `temp_s`, `humidity`,
            `r_change`, `temp_a_change`. The key refers to the variable that causes the variation with $x$.</br>
            This gives the prefactor for the term indicated such that `info_coef[var]` $\\times$ `info_change[var]`
            is the contribution for that term. Sum of all contributions equals $\delta T_s(x)-\delta \overline{T_s}$.
        `info_change`: Complementary dictionary to `info_coef` with same keys that gives the relavent change to a
            quantity i.e. the $\delta$ term. For both `temp_s` and `sphum`, this is $\delta \overline{T_s}$.
        `mu`: `float [n_quant]`</br>
            $\mu(x)$ factor. Will be all zeros if `include_squared_terms=False`.
            Otherwise, will be $\\mu(x) = \\frac{L_v \\alpha_s(x) q_s(x)}{\\beta_{s1}(x)}$.
    """
    # Compute adiabatic temperatures
    n_exp, n_quant = temp_surf_quant.shape
    temp_adiabat_mean = np.zeros_like(temp_surf_mean)
    for i in range(n_exp):
        temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)

    # Compute relative humidities
    r_mean = sphum_mean / sphum_sat(temp_surf_mean, pressure_surf)
    r_quant = sphum_quant / sphum_sat(temp_surf_quant, pressure_surf)
    r_anom = r_quant - r_mean[:, np.newaxis]

    # Compute epsilon
    # Quantify deviation from convective equilibrium in MSE space
    epsilon_mean = (moist_static_energy(temp_surf_mean, sphum_mean, height=0) -
                    moist_static_energy(temp_ft_mean, sphum_sat(temp_ft_mean, pressure_ft), z_ft_mean))*1000
    epsilon_quant = (moist_static_energy(temp_surf_quant, sphum_quant, height=0) -
                     moist_static_energy(temp_ft_quant, sphum_sat(temp_ft_quant, pressure_ft), z_ft_quant))*1000
    epsilon_anom = epsilon_quant - epsilon_mean[:, np.newaxis]

    temp_surf_anom_norm0 = (temp_surf_quant - temp_surf_mean[:, np.newaxis])[0] / temp_surf_mean[0]
    r_anom_norm0 = r_anom[0] / r_mean[0]

    # Record coefficients of each term in equation for delta T_s(x)
    # label is anomaly that causes variation with x.
    gamma_temp_s, gamma_r, gamma_e, gamma_r_change, gamma_temp_ft_change, gamma_e_change = \
        get_gamma_factors2(temp_surf_mean[0], sphum_mean[0],
                           temp_adiabat_mean[0] if gamma_from_temp_adiabat else temp_ft_mean[0],
                           pressure_surf, pressure_ft)

    _, _, alpha_s, beta_s1_mean, _, _ = get_theory_prefactor_terms(temp_surf_mean[0], pressure_surf, pressure_ft,
                                                                         sphum_mean[0])
    info_coef = {'t_s0': gamma_temp_s * temp_surf_anom_norm0,
                 'r0': -gamma_r * r_anom_norm0,
                 'e0': -gamma_e * epsilon_anom[0],
                 'r_change': -gamma_r_change / r_mean[0],
                 't_s0_r_change': -gamma_r_change * alpha_s * temp_surf_mean[0] * temp_surf_anom_norm0 / r_mean[0],
                 't_ft_change': gamma_temp_ft_change,
                 'e_change': gamma_e_change}

    if include_squared_terms:
        info_coef['t_s0_r_mean_change'] = (gamma_temp_s - alpha_s * temp_surf_mean[0]
                                             ) * temp_surf_anom_norm0 * gamma_r_change / r_mean[0]
        info_coef['r0_r_mean_change'] = -gamma_r * r_anom_norm0 * gamma_r_change / r_mean[0]
        info_coef['e0_r_mean_change'] = -gamma_e * epsilon_anom[0] * gamma_r_change / r_mean[0]

        info_coef['t_s0_e_mean_change'] = -gamma_temp_s * temp_surf_anom_norm0 * gamma_e_change
        info_coef['r0_e_mean_change'] = gamma_r * r_anom_norm0 * gamma_e_change
        info_coef['e0_e_mean_change'] = gamma_e * epsilon_anom[0] * gamma_e_change

    if z_form:
        if include_squared_terms:
            raise ValueError('Conversion to z form of theory is too complicated with squared terms')
        # Get parameters required for conversion to z form of theory
        R_mod, _, _, beta_a1, _, _ = get_theory_prefactor_terms(temp_adiabat_mean[0], pressure_surf, pressure_ft)

        # If provide z, will compute z version of the theory - replace temp_ft anomaly with z_ft anomaly
        z_ft_anom = z_ft_quant - z_ft_mean[:, np.newaxis]
        delta_z_ft_anom = z_ft_anom[1] - z_ft_anom[0]
        # write adiabatic temp anomaly change in terms of z anomaly rather than free troposphere temp anomaly
        delta_temp_ft_anom = g / R_mod * delta_z_ft_anom

        # In z-form, need to multiply each term by constant prefactor
        for var in info_coef:
            info_coef[var] = info_coef[var] * beta_s1_mean / (beta_s1_mean + beta_a1)
    else:
        temp_ft_anom = temp_ft_quant - temp_ft_mean[:, np.newaxis]
        delta_temp_ft_anom = temp_ft_anom[1] - temp_ft_anom[0]

    info_change = {}
    for var in info_coef:
        if 'change' not in var:
            info_change[var] = temp_surf_mean[1] - temp_surf_mean[0]
        elif 'r_change' in var:
            info_change[var] = 0 if ignore_rh else r_anom[1] - r_anom[0]
        elif 't_ft_change' in var:
            info_change[var] = delta_temp_ft_anom
        elif 'e_change' in var:
            info_change[var] = epsilon_anom[1] - epsilon_anom[0]
        elif 'r_mean_change' in var:
            info_change[var] = 0 if ignore_rh else r_mean[1] - r_mean[0]
        elif 'e_mean_change' in var:
            info_change[var] = 0 if ignore_rh else epsilon_mean[1] - epsilon_mean[0]
        else:
            raise ValueError(f'Dont know what change to use for {var}')
    if include_squared_terms:
        _, _, alpha_s_x, beta_s1_x, _, _ = get_theory_prefactor_terms(temp_surf_quant[0], pressure_surf, pressure_ft,
                                                                      sphum_quant[0])
        mu_factor = L_v * alpha_s_x * sphum_quant[0] / beta_s1_x * (r_quant[1] - r_quant[0]) / r_quant[0]
        mu_factor = np.zeros(n_quant)
    else:
        mu_factor = np.zeros(n_quant)
    final_answer = temp_surf_mean[1] - temp_surf_mean[0] + sum([info_coef[var] * info_change[var] for var in info_coef])
    final_answer = final_answer / (1+mu_factor)
    return final_answer, info_coef, info_change, mu_factor


def get_delta_temp_quant_theory_final3(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                                       sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                                       temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray, z_ft_mean: np.ndarray,
                                       z_ft_quant: np.ndarray, beta_approx: Optional[list] = None) -> Tuple[
    np.ndarray, dict, dict, np.ndarray]:
    """
    Calculates the theoretical near-surface temperature change for percentile $x$, $\delta T_s(x)$,
    such that in the simplest linear case (`include_squared_terms=False`) with `epsilon_form=True`, we have:

    $$
    \\begin{align}
    \\begin{split}
    \delta T_s(x) \\approx
    1 &+ \gamma_{T_s}\\frac{\Delta T_s(x)}{\overline{T_s}} \delta \overline{T_s}+
    \gamma_{r_s} \\frac{\Delta r_s(x)}{\overline{r_s}} \delta \overline{T_s}+
    \gamma_{\epsilon} \Delta \epsilon(x) \delta \overline{T_s} \\\\
    &+ \gamma_{\delta r_s} \\frac{\delta \Delta r_s(x)}{\overline{r_s}} +
    \gamma_{\delta T_{FT}} \delta \Delta T_{FT}(x) + \gamma_{\delta \epsilon} \delta \Delta \epsilon(x)
    \\end{split}
    \\end{align}
    $$

    If `include_squared_terms=True`, then non-linear and squared terms are included. The LHS becomes
    $\\left[1+\mu(x)\\frac{\delta r_s(x)}{\overline{r_s}}\\right]\delta T_s(x)$
    where $\\mu(x) = \\frac{L_v \\alpha_s(x) q_s(x)}{\\beta_{s1}(x)}$, and extra terms e.g.
    $\gamma_{T_s^2}(\\frac{\Delta T_s}{\overline{T_s}})^2 \delta \overline{T_s}$ are included on the RHS.

    If `z_quant` is given (with other parameters the same), then the $z$ form of the theory will be computed
    (not possible with `include_squared_terms=True` because gets too complicated):

    $$
    \\begin{align}
    \\begin{split}
    \delta T_s(x) \\approx
    1 + \\frac{\overline{\\beta_{s1}}}{\overline{\\beta_{s1}} + \\beta_{A1}} \\bigg(
    &\gamma_{T_s}\\frac{\Delta T_s(x)}{\overline{T_s}} \delta \overline{T_s}+
    \gamma_{r_s} \\frac{\Delta r_s(x)}{\overline{r_s}} \delta \overline{T_s}+
    \gamma_{\epsilon} \Delta \epsilon(x) \delta \overline{T_s} \\\\
    &+ \gamma_{\delta r_s} \\frac{\delta \Delta r_s(x)}{\overline{r_s}} +
    \gamma_{\delta T_{FT}} \\frac{g}{R^{\dagger}} \delta \Delta z_{FT}(x) +
    \gamma_{\delta \epsilon} \delta \Delta \epsilon(x) \\bigg)
    \\end{split}
    \\end{align}
    $$

    Args:
        temp_surf_mean: `float [n_exp]`</br>
            Average near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        sphum_mean: `float [n_exp]`</br>
            Average near surface specific humidity of each simulation. Units: *kg/kg*.
        sphum_quant: `float [n_exp, n_quant]`</br>
            `sphum_quant[i, j]` is near-surface specific humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        temp_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        z_ft_mean: ONLY NEEDED FOR $z$ THEORY</br>`float [n_exp]`</br>
            Average geopotential height at `pressure_ft` in *m*.
        z_ft_quant: IF GIVEN, WILL RETURN $z$ THEORY</br>`float [n_exp, n_quant]`</br>
            `z_ft_quant[i, j]` is geopotential height at `pressure_ft`, averaged over all days with near-surface
            temperature corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *m*.
        z_form: If `True`, will return $z$ version of theory.
        epsilon_form: If `True`, will quantify deviation from convective equilibrium through moist static energy:
            $\\epsilon = h_s - h_{FT}^*$. Otherwise, will quantify it in temperature space through
            $T_{CE} = T_{FT} - T_A$.
        ignore_rh: If `True`, will set $\delta r_s(x) = \delta \overline{r_s} = 0$.
        include_squared_terms: If `True`, will include $\Delta T_s^2\delta \overline{T_s}$,
            $\Delta r_s^2\delta \overline{T_s}$, $\Delta T_s \Delta r_s \delta \overline{T_s}$,
            $\Delta T_s^2 \Delta r_s \delta \overline{T_s}$, $\Delta T_s \Delta r_s^2 \delta \overline{T_s}$,
            $\Delta T_s \delta \overline{r_s}$, $\Delta T_s \delta \Delta r_s$,
            $\delta \overline{r_s} \delta \overline{T_s}$ and $\delta \Delta T_A \delta \overline{T_s}$ terms in theory.
    Returns:
        `delta_temp_quant`: `float [n_quant]`</br>
            `delta_temp_quant[i]` refers to the theoretical temperature difference between experiments
            for percentile `quant_use[j]`.
        `info_coef`: Dictionary with 4 keys for each term in the simple version of the theory: `temp_s`, `humidity`,
            `r_change`, `temp_a_change`. The key refers to the variable that causes the variation with $x$.</br>
            This gives the prefactor for the term indicated such that `info_coef[var]` $\\times$ `info_change[var]`
            is the contribution for that term. Sum of all contributions equals $\delta T_s(x)-\delta \overline{T_s}$.
        `info_change`: Complementary dictionary to `info_coef` with same keys that gives the relavent change to a
            quantity i.e. the $\delta$ term. For both `temp_s` and `sphum`, this is $\delta \overline{T_s}$.
        `mu`: `float [n_quant]`</br>
            $\mu(x)$ factor. Will be all zeros if `include_squared_terms=False`.
            Otherwise, will be $\\mu(x) = \\frac{L_v \\alpha_s(x) q_s(x)}{\\beta_{s1}(x)}$.
    """
    # Compute adiabatic temperatures
    n_exp, n_quant = temp_surf_quant.shape
    temp_adiabat_mean = np.zeros_like(temp_surf_mean)
    for i in range(n_exp):
        temp_adiabat_mean[i] = get_temp_adiabat(temp_surf_mean[i], sphum_mean[i], pressure_surf, pressure_ft)

    # Compute relative humidities
    r_mean = sphum_mean / sphum_sat(temp_surf_mean, pressure_surf)
    r_quant = sphum_quant / sphum_sat(temp_surf_quant, pressure_surf)
    r_anom = r_quant - r_mean[:, np.newaxis]

    # Compute epsilon
    # Quantify deviation from convective equilibrium in MSE space
    epsilon_mean = (moist_static_energy(temp_surf_mean, sphum_mean, height=0) -
                    moist_static_energy(temp_ft_mean, sphum_sat(temp_ft_mean, pressure_ft), z_ft_mean))*1000
    epsilon_quant = (moist_static_energy(temp_surf_quant, sphum_quant, height=0) -
                     moist_static_energy(temp_ft_quant, sphum_sat(temp_ft_quant, pressure_ft), z_ft_quant))*1000
    epsilon_anom = epsilon_quant - epsilon_mean[:, np.newaxis]

    temp_ft_anom = temp_ft_quant - temp_ft_mean[:, np.newaxis]

    temp_surf_anom0 = (temp_surf_quant - temp_surf_mean[:, np.newaxis])[0]

    _, _, alpha_s_x, beta_s1_x, _, _ = get_theory_prefactor_terms(temp_surf_quant[0], pressure_surf, pressure_ft,
                                                                   sphum_quant[0])
    _, q_sat_s, alpha_s, beta_s1_mean, _, _ = get_theory_prefactor_terms(temp_surf_mean[0], pressure_surf, pressure_ft,
                                                                         sphum_mean[0])
    _, _, _, beta_ft1, beta_ft2, _ = get_theory_prefactor_terms(temp_ft_mean[0], pressure_surf, pressure_ft)
    mu_x = L_v * alpha_s_x * sphum_quant[0] / beta_s1_x * (r_quant[1] - r_quant[0]) / r_quant[0]
    mu_mean = L_v * alpha_s * sphum_mean[0] / beta_s1_mean * (r_mean[1] - r_mean[0]) / r_mean[0]

    if beta_approx is None:
        beta_approx = []
    beta_s1_use = [beta_s1_x] * 7
    for i in range(len(beta_s1_use)):
        if i in beta_approx:
            beta_s1_use[i] = beta_s1_mean

    term0 = beta_ft1 * (temp_ft_anom[1]-temp_ft_anom[0]) / beta_s1_use[0]
    mse_mod_mean_change = beta_s1_mean * (1+mu_mean)*(temp_surf_mean[1]-temp_surf_mean[0]) + \
                          L_v * q_sat_s * (r_mean[1]-r_mean[0]) - (epsilon_mean[1]-epsilon_mean[0])
    mse_mod_anom0 = (beta_s1_mean * temp_surf_anom0 + L_v * q_sat_s * r_anom[0] - epsilon_anom[0])
    term1 = mse_mod_mean_change/beta_s1_use[1]
    term2 = beta_ft2/beta_ft1**2/temp_ft_mean[0] * mse_mod_anom0 * mse_mod_mean_change / beta_s1_use[2]
    term3_mean = (epsilon_mean[1]-epsilon_mean[0])/beta_s1_use[3]
    term3_anom = (epsilon_anom[1]-epsilon_anom[0])/beta_s1_use[4]
    term4_mean = -L_v * sphum_quant[0]/r_quant[0] * (r_mean[1]-r_mean[0])/beta_s1_use[5]
    term4_anom = -L_v * sphum_quant[0] / r_quant[0] * (r_anom[1] - r_anom[0]) / beta_s1_use[6]

    final_answer = (term0+term1+term2+term3_mean+term3_anom+term4_mean+term4_anom)/(1+mu_x)
    return final_answer
