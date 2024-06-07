import numpy as np
import scipy.optimize
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v, R, g
from typing import Tuple, Union, Optional


def temp_adiabat_fit_func(temp_ft_adiabat: float, temp_surf: float, sphum_surf: float,
                          pressure_surf: float, pressure_ft: float) -> float:
    """
    Adiabatic Free Troposphere temperature, $T_{A,FT}$, is defined such that surface moist static energy, $h$
    is equal to the saturated moist static energy, $h^*$, evaluated at $T_A$ and free troposphere pressure,
    $p_{FT}$ i.e. $h(T_s, q_s, p_s) = h^*(T_{A}, p_{FT})$.

    Using the following approximate relationship between $z_A$ and $T_A$:

    $$z_A - z_s \\approx \\frac{R^{\\dagger}}{g}(T_s + T_A)$$

    where $R^{\dagger} = \\ln(p_s/p_{FT})/2$, we can obtain $T_A$
    by solving the following equation for modified MSE, $h^{\\dagger}$:

    $$h^{\\dagger} = (c_p - R^{\\dagger})T_s + L_v q_s \\approx
    (c_p + R^{\\dagger})T_A + L_vq^*(T_A, p_{FT})$$

    This function returns the LHS minus the RHS of this equation to then give to `scipy.optimize.fsolve` to find
    $T_A$.

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
                     guess_temp_adiabat: float = 273) -> float:
    """
    This returns the adiabatic temperature at `pressure_ft`, $T_{A, FT}$, such that surface moist static
    energy equals free troposphere saturated moist static energy: $h(T_s, q_s, p_{FT}) = h^*(T_{A}, p_{FT})$.

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
        temp_adiabat_anom: `float [n_exp, n_quant]`</br>
            Adiabatic temperature anomaly at `pressure_ft`, $\Delta T_A(x) = T_A(x) - \overline{T_A}$.
        temp_ce_mean: `float [n_exp]`</br>
            Deviation of mean temperature at `pressure_ft` from mean adiabatic temperature:
            $\overline{T_{CE}} = \overline{T_{FT}} - \overline{T_A}$.
        temp_ce_quant: `float [n_exp, n_quant]`</br>
            Deviation of temperature at `pressure_ft` from adiabatic temperature: $T_{CE}(x) = T_{FT}(x) - T_A(x)$.
            Conditioned on percentile of near-surface temperature.
        temp_ft_anom: `float [n_exp, n_quant]`</br>
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
        R_mod: Modified gas constant, $R^{\dagger} = R\\ln(p_s/p_{FT})/2$</br>
            Units: *J/kg/K*
        q_sat: `float` or `float [n_temp]`</br>
            Saturated specific humidity. $q^*(T, p_s)$ if `sphum` given, otherwise $q^*(T, p_{FT})$</br>
            Units: *kg/kg*
        alpha: `float` or `float [n_temp]`</br>
            Clausius clapeyron parameter. $\\alpha(T, p_s)$ if `sphum` given, otherwise $\\alpha(T, p_{FT})$</br>
            Units: K$^{-1}$
        beta_1: `float` or `float [n_temp]`</br>
            $\\frac{d(h^{\\dagger}+\epsilon)}{dT_s}(T, p_s)$ if `sphum` given, otherwise
            $\\frac{h^{\\dagger}}{dT_{FT}}(T, p_{FT})$.</br>
            Units: *J/kg/K*
        beta_2: `float` or `float [n_temp]`</br>
            $T_s\\frac{d^2(h^{\\dagger}+\epsilon)}{dT_s^2}(T, p_s)$ if `sphum` given, otherwise
            $T_{FT}\\frac{d^2h^{\\dagger}}{dT_{FT}^2}(T, p_{FT})$.</br>
            Units: *J/kg/K*
        beta_3: `float` or `float [n_temp]`</br>
            $T_s^2\\frac{d^3(h^{\\dagger}+\epsilon)}{dT_s^3}(T, p_s)$ if `sphum` given, otherwise
            $T_{FT}^2\\frac{d^3h^{\\dagger}}{dT_{FT}^3}(T, p_{FT})$.</br>
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


def get_gamma_factors(temp_surf: float, sphum: float, temp_ft: float, pressure_surf: float,
                      pressure_ft: float) -> dict:
    """
    Calculates the sensitivity $\gamma$ parameters such that the theoretical scaling factor is given by:

    $$
    \\begin{align}
    \\begin{split}
    &\\left(1 + \mu(x)\\frac{\delta r_s(x)}{r_s(x)}\\right)\\frac{\delta T_s(x)}{\overline{\delta T_s}} \\approx
    1 + \overline{\mu} \\frac{\delta \overline{r_s}}{\overline{r_s}} + \\\\
    &\\left[\gamma_{T}\\frac{\Delta T_s(x)}{\overline{T_s}} -
    \gamma_{Tr}\\frac{\Delta T_s(x)}{\overline{T_s}}\\frac{\Delta r_s(x)}{\overline{r_s}}
    - \gamma_{r} \\frac{\Delta r_s(x)}{\overline{r_s}}-
    \gamma_{\epsilon} \\frac{\Delta \epsilon(x)}{\overline{\\beta_{s1}}\overline{T_s}}\\right]
    \\left(1 + \overline{\mu} \\frac{\delta \overline{r_s}}{\overline{r_s}}\\right) +\\\\
    &\\left[-\gamma_{T\delta r}\\frac{\Delta T_s(x)}{\overline{T_s}} +
    \gamma_{Tr\delta r}\\frac{\Delta T_s(x)}{\overline{T_s}}\\frac{\Delta r_s(x)}{\overline{r_s}}
    + \gamma_{r\delta r} \\frac{\Delta r_s(x)}{\overline{r_s}}-
    \gamma_{\epsilon \delta r} \\frac{\Delta \epsilon(x)}{\overline{\\beta_{s1}}\overline{T_s}}\\right]
    \\frac{\delta \overline{r_s}}{\delta \overline{T_s}} +\\\\
    &\\left[-\gamma_{T\delta \epsilon}\\frac{\Delta T_s(x)}{\overline{T_s}} -
    \gamma_{Tr\delta \epsilon}\\frac{\Delta T_s(x)}{\overline{T_s}}\\frac{\Delta r_s(x)}{\overline{r_s}}
    - \gamma_{r\delta \epsilon} \\frac{\Delta r_s(x)}{\overline{r_s}}+
    \gamma_{\epsilon \delta \epsilon} \\frac{\Delta \epsilon(x)}{\overline{\\beta_{s1}}\overline{T_s}}\\right]
    \\frac{\delta \overline{\epsilon}}{\delta \overline{T_s}} +\\\\
    &- \gamma_{\delta \Delta r} \\frac{\delta \Delta r_s(x)}{\delta \overline{T_s}} -
    \gamma_{T \delta \Delta r} \\frac{\Delta T_s(x)}{\overline{T_s}}\\frac{\delta \Delta r_s(x)}{\delta \overline{T_s}}
    + \gamma_{FT} \\frac{\delta \Delta T_{FT}(x)}{\delta \overline{T_s}} + \
    \gamma_{\delta \Delta \epsilon} \\frac{\delta \Delta \epsilon(x)}{\delta \overline{T_s}}
    \\end{split}
    \\end{align}
    $$

    Args:
        temp_surf:
            Temperature at `pressure_surf` in *K*.
        sphum:
            Specific humidity at `pressure_surf` in *kg/kg*.
        temp_ft:
            Temperature at `pressure_ft` in *K*. Can also use adiabatic temperature here (computed from `temp_surf` and
            `sphum`), if want $\gamma$ factors to be independent of free tropospheric temperature
            (i.e. depend on two not three quantities).
        pressure_surf:
            Pressure at near-surface, $p_s$ in *Pa*.
        pressure_ft:
            Pressure at free troposphere level, $p_{FT}$ in *Pa*.

    Returns:
        gamma:
            The theory (ignoring $\mu$ factors) is a sum of 16 terms. The first four $\gamma$ factors multiply
            a climatological anomaly term ($\Delta$) but not a change
            ($\delta$) term, these are recorded in `gamma['temp_mean_change']` with the keys
            `t0`, `t0_r0`, `r0` and `e0`, They are all dimensionless.

            The next four preceed $\\frac{\delta \overline{r_s}}{\delta \overline{T_s}}$ as well as a $\Delta$ term.
            These are recorded in `gamma['r_mean_change']`. Again they have keys `t0`, `t0_r0`, `r0` and `e0`.
            These all have units of $K$.

            The next four preceed $\\frac{\delta \overline{\epsilon}}{\delta \overline{T_s}}$, as well as a $\Delta$
            term. These are recorded in `gamma['e_mean_change']`. Again they have keys `t0`, `t0_r0`, `r0` and `e0`.
            These all have units of $K kg J^{-1}$.

            The final four all preceed $\delta \Delta$ quantities and are recorded in `gamma['anomaly_change']`.
            They have keys `r`, `t0_r`, `ft` and `e`. They have units of $K$, $K$, dimensionless and  $K kg J^{-1}$
            respectively.
    """
    # Get parameters required for prefactors in the theory
    _, _, _, beta_ft1, beta_ft2, beta_ft3 = get_theory_prefactor_terms(temp_ft, pressure_surf, pressure_ft)
    _, q_sat_surf, alpha_s, beta_s1, beta_s2, beta_s3 = get_theory_prefactor_terms(temp_surf, pressure_surf,
                                                                                   pressure_ft, sphum)
    mu = L_v * sphum * alpha_s / beta_s1
    rh = sphum / q_sat_surf

    # Record coefficients of each term in equation for delta T_s(x)
    # label is anomaly that causes variation with x.

    gamma = {'t_mean_change': {}, 'r_mean_change': {}, 'e_mean_change': {}, 'anomaly_change': {}}
    # temp_s_mean_change terms
    key = 't_mean_change'
    gamma_e0 = beta_ft2 * beta_s1 / beta_ft1**2 * temp_surf/temp_ft
    gamma[key]['t0'] = gamma_e0 - beta_s2/beta_s1
    gamma[key]['t0_r0'] = beta_s2/beta_s1 - mu * gamma_e0
    gamma[key]['r0'] = mu * (1 - gamma_e0 / (alpha_s * temp_surf))
    gamma[key]['e0'] = gamma_e0

    # Anomaly change terms
    key = 'anomaly_change'
    gamma_r_change = mu / alpha_s / rh
    gamma_e_change = 1/beta_s1
    gamma[key]['r'] = gamma_r_change
    gamma[key]['t0_r'] = gamma_r_change * alpha_s * temp_surf
    gamma[key]['ft'] = beta_ft1 / beta_s1
    gamma[key]['e'] = gamma_e_change

    # r_s_mean change terms
    key = 'r_mean_change'
    gamma[key]['t0'] = gamma_r_change * gamma_e0 * (alpha_s * temp_surf / gamma_e0 - 1)
    gamma[key]['t0_r0'] = gamma_r_change * gamma_e0 * mu
    gamma[key]['r0'] = gamma_r_change * gamma_e0 * mu / (alpha_s * temp_surf)
    gamma[key]['e0'] = gamma_r_change * gamma_e0

    # epsilon mean change terms
    key = 'e_mean_change'
    gamma[key]['t0'] = gamma_e_change * gamma_e0
    gamma[key]['t0_r0'] = gamma_e_change * gamma_e0 * mu
    gamma[key]['r0'] = gamma_e_change * gamma_e0 * mu / (alpha_s * temp_surf)
    gamma[key]['e0'] = gamma_e_change * gamma_e0
    return gamma


def mse_mod_anom_change_ft_expansion(temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray,
                                     pressure_surf: float, pressure_ft: float,
                                     taylor_terms: str = 'linear', mse_mod_mean_change: Optional[float] = None,
                                     temp_ft_anom0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    This function returns an approximation in the change in modified MSE anomaly,
    $\delta \Delta h^{\dagger} = \delta (h^{\dagger}(x) - \overline{h^{\dagger}})$, with warming -
    the basis of a theory for $\delta T_s(x)$.

    Doing a second order taylor expansion of $h^{\dagger}$ in the base climate,
    about free tropospheric temperature $\overline{T_{FT}}$, we can get:

    $$\\Delta h^{\\dagger}(x) \\approx \\beta_{FT1} \\Delta T_{FT} + \\frac{1}{2\\overline{T_{FT}}}
    \\beta_{FT2} \\Delta T_{FT}^2$$

    Terms in equation:
        * $h^{\dagger} = h^*_{FT} - R^{\dagger}T_s - gz_s \\approx \\left(c_p + R^{\dagger}\\right) T_{FT} + L_v q^*_{FT}$
        where we used an approximate relation to replace $z_{FT}$ in $h^*_{FT}$.
        * $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\\Delta T_{FT} = T_{FT}(x) - \overline{T_{FT}}$
        * $\\beta_{FT1} = \\frac{d\\overline{h^{\\dagger}}}{d\overline{T_{FT}}} =
        c_p + R^{\dagger} + L_v \\alpha_{FT} q_{FT}^*$
        * $\\beta_{FT2} = \overline{T_{FT}} \\frac{d^2\\overline{h^{\\dagger}}}{d\overline{T_{FT}}^2} =
         \overline{T_{FT}}\\frac{d\\beta_{FT1}}{d\overline{T_{FT}}} =
         L_v \\alpha_{FT} q_{FT}^*(\\alpha_{FT} \\overline{T_{FT}} - 2)$
        * All terms on RHS are evaluated at the free tropospheric adiabatic temperature, $T_{FT}$. I.e.
        $q_{FT}^* = q^*(T_{FT}, p_{FT})$ where $p_{FT}$ is the free tropospheric pressure.

    Doing a second taylor expansion on this equation for a change with warming between simulations, $\delta$, we can
    decompose $\delta \\Delta h^{\\dagger}(x)$ into terms involving $\delta \Delta T_{FT}$ and
    $\delta \overline{T_{FT}}$

    We can then use a third taylor expansion to relate $\delta \overline{T_{FT}}$ to $\delta \overline{h^{\\dagger}}$:

    $$\\delta \\overline{T_{FT}} \\approx \\frac{\\delta \\overline{h^{\\dagger}}}{\\beta_{FT1}} -
    \\frac{1}{2} \\frac{\\beta_{FT2}}{\\beta_{FT1}^3 \\overline{T_{FT}}} (\\delta \\overline{h^{\\dagger}})^2$$

    Overall, we get $\delta \Delta h^{\dagger}$ as a function of $\delta \Delta T_{FT}$,
    $\delta \overline{h^{\\dagger}}$ and quantities evaluated at the base climate.
    The `taylor_terms` variable can be used to specify how many terms we want to keep.

    The simplest equation with `taylor_terms = 'linear'` is:

    $$\\delta \\Delta h^{\\dagger} \\approx \\beta_{FT1} \\delta \\Delta T_{FT} +
    \\frac{\\beta_{FT2}}{\\beta_{FT1}}\\frac{\\Delta T_{FT}}{\\overline{T_{FT}}} \\delta \\overline{h^{\\dagger}}$$

    Args:
        temp_ft_mean: `float [n_exp]`</br>
            Average free tropospheric temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
            Could also use adiabatic temperature, $\overline{T_A}$, instead if want to assume
            strict convective equilibrium.
        temp_ft_quant: `float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is free tropospheric temperature, averaged over all days with near-surface temperature
            corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
            Could also use adiabatic temperature, $T_A(x)$, instead if want to assume
            strict convective equilibrium.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        taylor_terms:
            The approximations in this equation arise from the three taylor series mentioned above, we can specify
            how many terms we want to keep, with one of the 3 options below:

            * `linear`: Only keep the two terms which are linear in all three taylor series i.e.
            $\\delta \\Delta h^{\\dagger} \\approx \\beta_{FT1} \\delta \\Delta T_{FT} +
            \\frac{\\beta_{FT2}}{\\beta_{FT1}}\\frac{\\Delta T_{FT}}{\\overline{T_{FT}}}
            \\delta \\overline{h^{\\dagger}}$
            * `non_linear`: Keep *LL*, *LLL* and *LNL* terms.
            * `squared_0`: Keep terms linear and squared in first expansion and then just linear terms:
            *LL*, *LLL*, *SL*, *SLL*.
            * `squared`: Keep four additional terms corresponding to *LLS*, *LSL*, *LNL* and *SNL* terms in the
            taylor series. SNL means second order in the first taylor series mentioned above, non-linear
            (i.e. $\\delta \\Delta T_{FT}\\delta \\overline{h^{\\dagger}}$ terms)  in the second
            and linear in the third. These 5 terms are the most significant non-linear terms.
            * `full`: In addition to the terms in `squared`, we keep the usually small *LSS* and *SLS* terms.
        mse_mod_mean_change: `float [n_exp]`</br>
            Can provide the $\\delta \\overline{h^{\\dagger}}$ in J/kg. Use this if you want to use a particular taylor
            approximation for this.
        temp_ft_anom0: `float [n_quant]`</br>
            Can specify the $\Delta T_{FT}$ term to use in the equation. May want to do this, if want to
            relate $\Delta T_{FT}$ to surface quantities assuming strict convective equilibrium.

    Returns:
        delta_mse_mod_anomaly: `float [n_quant]`</br>
            $\delta \Delta h^{\dagger}$ conditioned on each quantile of near-surface temperature. Units: *kJ/kg*.
        info_dict: Dictionary with 5 keys: `temp_ft_anom`, `mse_mod_mean`, `mse_mod_mean_squared`,
            `mse_mod_mean_cubed`, `non_linear`.</br>
            For each key, a list containing a prefactor computed in the base climate and a change between simulations is
            returned. I.e. for `info_dict[non_linear][1]` would be
            $\\delta \\Delta T_{FT}\\delta \\overline{h^{\\dagger}}$
            and the total contribution of non-linear terms to $\delta \Delta h^{\dagger}$ would be
            `info_dict[non_linear][0] * info_dict[non_linear][1]`. In the `linear` case this would be zero,
            and `info_dict[temp_ft_anom][0]`$=\\beta_{FT1}$ and `info_dict[mse_mod_mean][0]`$=
            \\frac{\\beta_{FT2}}{\\beta_{FT1}}\\frac{\\Delta T_A}{\\overline{T_{FT}}}$ would be the only non-zero
            prefactors.
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
        term_sl = beta_2 * temp_ft_anom0 / temp_ft_mean[0] * delta_temp_ft_anom
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
                                  q_sat_s_linear_term_use: Optional[Union[np.ndarray, float]] = None,
                                  beta_s1_use: Optional[Union[np.ndarray, float]] = None
                                  ) -> Tuple[Union[np.ndarray, float], dict]:
    """
    Does a taylor expansion of the change in modified moist static energy, $\delta h^{\dagger}$ in terms
    of surface quantities. This approximates the change of $h^{\dagger} = (c_p - R^{\\dagger})T_s + L_v q_s - \epsilon$
    between climates.

    $$\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha_s q_s)\\delta T_s + L_v q_s^* \\delta r_s
    - \delta \epsilon + L_v \\alpha_s q_s^* \delta T_s \\delta r_s  +
    0.5 L_v \\alpha_s q_s (\\alpha_s - 2 / T_s) \\delta T_s^2$$

    In terms of $\\beta$ parameters, this can be written as:

    $$\\delta h^{\\dagger} \\approx \\beta_{s1} \delta T_s + L_v q_s^* \\delta r - \delta \epsilon +
    L_v \\alpha_s q_s^* \delta T_s \\delta r_s + \\frac{1}{2 T_s} \\beta_{s2} \delta T_s^2$$

    The $\delta T_s^2$ term is only included if `taylor_terms == 'squared'`.
    The $\delta T_s \delta r_s$ term is only included if `taylor_terms` is `squared` or `non_linear`.

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

            * `linear`: $\\delta h^{\\dagger} \\approx (c_p - R^{\\dagger} + L_v \\alpha_s q_s)\\delta T_s +
                L_v \\alpha q_s^* \\delta r_s - \delta \epsilon$
            * `non_linear`: Includes the additional term $L_v \\alpha_s q_s^* \delta T_s \\delta r_s$
            * `squared`: Includes the additional term $0.5 L_v \\alpha_s q_s (\\alpha_s - 2 / T_s) \\delta T_s^2$
        q_sat_s_linear_term_use: `float` or `float [n_quant]` </br>
            Can specify the $q^*(T, p_s)$ value in the $L_v q^* \\delta r$ term. May want to do
            this if want to approximate $q^*(T(x), p_s) \\approx q^*(\overline{T}, p_s)$ for this term, so can combine
            $\\delta r(x)$ and $\\delta \overline{r}$ terms. If `None`, then will use
            `sphum_sat(temp_surf[0], pressure_surf)`.
        beta_s1_use: `float` or `float [n_quant]` </br>
            Can specify the $\\beta_{s1} = (c_p - R^{\\dagger} + L_v \\alpha q)$ factor preceeding the $\delta T_s$
            term. May want to do this if want to investigate the approximation
            $\\beta_{s1}(x) \\approx \overline{\\beta_{s1}}$.

    Returns:
        delta_mse_mod: `float` or `float [n_quant]` </br>
            Approximation of $\delta h^{\\dagger}$. Units are *kJ/kg*.
        info_dict: Dictionary containing 5 keys: `rh`, `temp`, `epsilon`, `non_linear`, `temp_squared`. </br>
            For each key, there is a list with first value being the prefactor and second the change in the expansion.
            The sum of all these prefactors multiplied by the changes equals the full theory.
            Units of prefactors are *kJ/kg* divided by units of the change term.
    """
    _, _, _, beta_s1, beta_s2, _ = get_theory_prefactor_terms(temp_surf[0], pressure_surf, pressure_ft, sphum_surf[0])

    delta_temp = temp_surf[1] - temp_surf[0]
    rh = sphum_surf / sphum_sat(temp_surf, pressure_surf)
    delta_rh = rh[1] - rh[0]
    delta_epsilon = epsilon[1] - epsilon[0]

    q_sat_s = sphum_sat(temp_surf[0], pressure_surf)
    alpha_s = clausius_clapeyron_factor(temp_surf[0], pressure_surf)

    if q_sat_s_linear_term_use is None:
        coef_rh = L_v * q_sat_s
    else:
        coef_rh = L_v * q_sat_s_linear_term_use
    coef_temp = beta_s1 if beta_s1_use is None else beta_s1_use

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


def get_scaling_factor_theory(temp_surf_mean: np.ndarray, temp_surf_quant: np.ndarray, sphum_mean: np.ndarray,
                              sphum_quant: np.ndarray, pressure_surf: float, pressure_ft: float,
                              temp_ft_mean: np.ndarray, temp_ft_quant: np.ndarray, z_ft_mean: np.ndarray,
                              z_ft_quant: np.ndarray, non_linear: bool = False,
                              z_form: bool = False, use_temp_adiabat: bool = False) -> Tuple[
    np.ndarray, dict, dict, dict, Union[float, np.ndarray]]:
    """
    Calculates the theoretical near-surface temperature change for percentile $x$, $\delta T_s(x)$, relative
    to the mean temperature change, $\overline{\delta T_s}$. In the most complicated case, with `non_linear = True`,
    this is:

    $$
    \\begin{align}
    \\begin{split}
    &\\left(1 + \mu(x)\\frac{\delta r_s(x)}{r_s(x)}\\right)\\frac{\delta T_s(x)}{\overline{\delta T_s}} \\approx
    1 + \overline{\mu} \\frac{\delta \overline{r_s}}{\overline{r_s}} + \\\\
    &\\left[\gamma_{T}\\frac{\Delta T_s(x)}{\overline{T_s}} -
    \gamma_{Tr}\\frac{\Delta T_s(x)}{\overline{T_s}}\\frac{\Delta r_s(x)}{\overline{r_s}}
    - \gamma_{r} \\frac{\Delta r_s(x)}{\overline{r_s}}-
    \gamma_{\epsilon} \\frac{\Delta \epsilon(x)}{\overline{\\beta_{s1}}\overline{T_s}}\\right]
    \\left(1 + \overline{\mu} \\frac{\delta \overline{r_s}}{\overline{r_s}}\\right) +\\\\
    &\\left[-\gamma_{T\delta r}\\frac{\Delta T_s(x)}{\overline{T_s}} +
    \gamma_{Tr\delta r}\\frac{\Delta T_s(x)}{\overline{T_s}}\\frac{\Delta r_s(x)}{\overline{r_s}}
    + \gamma_{r\delta r} \\frac{\Delta r_s(x)}{\overline{r_s}}-
    \gamma_{\epsilon \delta r} \\frac{\Delta \epsilon(x)}{\overline{\\beta_{s1}}\overline{T_s}}\\right]
    \\frac{\delta \overline{r_s}}{\delta \overline{T_s}} +\\\\
    &\\left[-\gamma_{T\delta \epsilon}\\frac{\Delta T_s(x)}{\overline{T_s}} -
    \gamma_{Tr\delta \epsilon}\\frac{\Delta T_s(x)}{\overline{T_s}}\\frac{\Delta r_s(x)}{\overline{r_s}}
    - \gamma_{r\delta \epsilon} \\frac{\Delta r_s(x)}{\overline{r_s}}+
    \gamma_{\epsilon \delta \epsilon} \\frac{\Delta \epsilon(x)}{\overline{\\beta_{s1}}\overline{T_s}}\\right]
    \\frac{\delta \overline{\epsilon}}{\delta \overline{T_s}} +\\\\
    &- \gamma_{\delta \Delta r} \\frac{\delta \Delta r_s(x)}{\delta \overline{T_s}} -
    \gamma_{T \delta \Delta r} \\frac{\Delta T_s(x)}{\overline{T_s}}\\frac{\delta \Delta r_s(x)}{\delta \overline{T_s}}
    + \gamma_{FT} \\frac{\delta \Delta T_{FT}(x)}{\delta \overline{T_s}} + \
    \gamma_{\delta \Delta \epsilon} \\frac{\delta \Delta \epsilon(x)}{\delta \overline{T_s}}
    \\end{split}
    \\end{align}
    $$

    If `z_form = True`, then all $\gamma$ and $\mu$ parameters are multiplied by
    $\\frac{\overline{\\beta_{s1}}}{\overline{\\beta_{s1}} + \\beta_{FT1}}$, and $\delta \Delta T_{FT}(x)$ is replaced
    with $\\frac{g}{R^{\dagger}}\delta \Delta z_{FT}(x)$.

    If `non_linear = False`, then $\overline{\mu}$ and $\mu(x)$ are set to zero.

    If `use_temp_adiabat = True`, then the mean adiabatic temperature, $\overline{T_A}$, is used in the computation of
    the $\gamma$ parameters (and $\\beta_{FT1}$ if `z_form=True`), rather than the mean free tropospheric temperature,
    $\overline{T_{FT}}$ if it is `False`.

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
        temp_ft_mean: `float [n_exp]`</br>
            Average temperature at `pressure_ft` in Kelvin.
        temp_ft_quant: `float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        z_ft_mean: `float [n_exp]`</br>
            Average geopotential height at `pressure_ft` in *m*.
        z_ft_quant: `float [n_exp, n_quant]`</br>
            `z_ft_quant[i, j]` is geopotential height at `pressure_ft`, averaged over all days with near-surface
            temperature corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *m*.
        non_linear: If `True`, will also include $\mu\delta r$ terms in theory.
        z_form: If `True`, will return $z$ version of theory.
        use_temp_adiabat: If `True`, then the mean adiabatic temperature, $\overline{T_A}$, is used in the computation
            of the $\gamma$ parameters, rather than the mean free tropospheric temperature, $\overline{T_{FT}}$
            if it is `False`.

    Returns:
        scaling_factor: `float [n_quant]`</br>
            `scaling_factor[i]` refers to the theoretical temperature difference between experiments
            for percentile `quant_use[i]`, relative to the mean temperature change, $\delta \overline{T_s}$.
        info_coef: The linear theory is a sum of 16 terms. The first four don't have any change ($\delta$)
            factor, these are recorded in `info_coef['temp_mean_change']`.
            The next four preceed $\\frac{\delta \overline{r_s}}{\delta \overline{T_s}}$, these are recorded in
            `info_coef['r_mean_change']`.
            The next four preceed $\\frac{\delta \overline{\epsilon}}{\delta \overline{T_s}}$, these are recorded in
            `info_coef['e_mean_change']`.
            The final four all preceed $\delta \Delta$ quantities and are recorded in `info_coef['anomaly_change']`.
            `info_coef` is independent of the value of `non_linear` used.
        info_change: Complementary dictionary to `info_coef` with same keys that gives the relavent change to a
            quantity i.e. `info_change['r_mean_change']` is $\\frac{\delta \overline{\epsilon}}{\delta \overline{T_s}}$.
            `info_change['anomaly_change']` is a dictionary of the four $\delta \Delta$ changes.
            If `non_linear = True`, each value in `info_change` is divided by
            $\\left(1 + \mu(x)\\frac{\delta r_s(x)}{r_s(x)}\\right)$.
        info_cont: Dictionary containing same keys as `info_coef`. Each term is `info_coef` $\times$ `info_change`
            so it gives the contribution in $K/K$ to the scaling factor for each of the 16 terms.
        mu_factor_ratio: `float` or `float [n_quant]`</br>
            The ratio $\\left(1 + \overline{\mu} \\frac{\delta \overline{r_s}}{\overline{r_s}}\\right)/
            \\left(1 + \mu(x)\\frac{\delta r_s(x)}{r_s(x)}\\right)$. Will be 1 if `non_linear = False`.
    """
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

    # Get factors needed for theory
    if use_temp_adiabat:
        # use adiabatic temperature for current climate to compute gamma params so only depends on surface quantities
        temp_mean_ft_beta_use = get_temp_adiabat(temp_surf_mean[0], sphum_mean[0], pressure_surf, pressure_ft)
    else:
        temp_mean_ft_beta_use = temp_ft_mean[0]
    gamma = get_gamma_factors(temp_surf_mean[0], sphum_mean[0], temp_mean_ft_beta_use, pressure_surf, pressure_ft)
    _, _, alpha_s, beta_s1, _, _ = get_theory_prefactor_terms(temp_surf_mean[0], pressure_surf, pressure_ft,
                                                                  sphum_mean[0])
    if non_linear:
        _, _, alpha_s_x, beta_s1_x, _, _ = get_theory_prefactor_terms(temp_surf_quant[0], pressure_surf, pressure_ft,
                                                                      sphum_quant[0])
        mu_x = L_v * alpha_s_x * sphum_quant[0] / beta_s1_x
        mu = L_v * alpha_s * sphum_mean[0] / beta_s1
    else:
        mu = 0
        mu_x = 0

    if z_form:
        R_mod, _, _, beta_ft1, _, _ = get_theory_prefactor_terms(temp_mean_ft_beta_use, pressure_surf, pressure_ft)
        temp_ft_anom_change = g/R_mod * np.diff(z_ft_quant - z_ft_mean[:, np.newaxis], axis=0)[0]
        # Need to multiply all mu and gamma factors by beta_s1/(beta_s1+beta_ft1) if z form of theory
        mu = mu * beta_s1 / (beta_s1+beta_ft1)
        mu_x = mu_x * beta_s1 / (beta_s1+beta_ft1)
        for key1 in gamma:
            for key2 in gamma[key1]:
                gamma[key1][key2] = gamma[key1][key2] * beta_s1 / (beta_s1+beta_ft1)
    else:
        temp_ft_anom_change = np.diff(temp_ft_quant - temp_ft_mean[:, np.newaxis], axis=0)[0]
    mu_factor = 1 + mu * (r_mean[1] - r_mean[0]) / r_mean[0]
    mu_factor_x = 1 + mu_x * (r_quant[1] - r_quant[0]) / r_quant[0]

    anom_norm0 = {'t0': (temp_surf_quant - temp_surf_mean[:, np.newaxis])[0] / temp_surf_mean[0],
                  'r0': r_anom[0] / r_mean[0], 'e0': epsilon_anom[0]/(beta_s1*temp_surf_mean[0])}
    anom_norm0['t0_r0'] = anom_norm0['t0'] * anom_norm0['r0']

    # Record the sign of each term, and also normalize relative humidity change terms by climatological mean
    # relative humidity
    coef_sign = {'t_mean_change': {key2: 1 if key2 == 't0' else -1 for key2 in gamma['t_mean_change']},
                 'r_mean_change': {key2: -1 if key2 in ['t0', 'e0'] else 1
                                   for key2 in gamma['r_mean_change']},
                 'e_mean_change': {key2: 1 if key2 == 'e0' else -1 for key2 in gamma['e_mean_change']},
                 'anomaly_change': {key2: -1 if 'r' in key2 else 1 for key2 in gamma['anomaly_change']}}

    # Each term is a coefficient evaluated in the current climate, multiplied by a change between climates.
    # Record the climatological coefficient in info_coef
    info_coef = {key: {} for key in gamma}
    for key1 in gamma:
        for key2 in gamma[key1]:
            if 'anomaly' in key1:
                if 't0' in key2:
                    info_coef[key1][key2] = coef_sign[key1][key2] * gamma[key1][key2] * anom_norm0['t0']
                else:
                    info_coef[key1][key2] = coef_sign[key1][key2] * gamma[key1][key2]
            else:
                info_coef[key1][key2] = coef_sign[key1][key2] * gamma[key1][key2] * anom_norm0[key2]

    # Record the change between climates in info_change
    info_change = {'t_mean_change': (temp_surf_mean[1] - temp_surf_mean[0]) * mu_factor,
                   'r_mean_change': r_mean[1] - r_mean[0],
                   'e_mean_change': epsilon_mean[1] - epsilon_mean[0],
                   'anomaly_change': {'r': r_anom[1]-r_anom[0],
                                      't0_r': r_anom[1]-r_anom[0],
                                      'ft': temp_ft_anom_change,
                                      'e': epsilon_anom[1] - epsilon_anom[0]}}

    # info_change - Normalise by mean temp change to get scale factor estimate
    # info_cont - Multiply coef by change to get overall contribution of each term
    info_cont = {key1: {} for key1 in gamma}
    for key1 in gamma:
        if 'mean' in key1:
            info_change[key1] = info_change[key1] / (temp_surf_mean[1] - temp_surf_mean[0]) / mu_factor_x
        for key2 in gamma[key1]:
            if 'mean' in key1:
                info_cont[key1][key2] = info_coef[key1][key2] * info_change[key1]
            else:
                info_change[key1][key2] = info_change[key1][key2] / (temp_surf_mean[1] - temp_surf_mean[0]
                                                                     ) / mu_factor_x
                info_cont[key1][key2] = info_coef[key1][key2] * info_change[key1][key2]

    final_answer = mu_factor/mu_factor_x + sum([sum([info_cont[key1][key2] for key2 in info_coef[key1]])
                                                for key1 in info_coef])

    return final_answer, info_coef, info_change, info_cont, mu_factor/mu_factor_x
