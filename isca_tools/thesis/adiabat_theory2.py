import numpy as np
import scipy.optimize
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v, R, g
from .adiabat_theory import get_theory_prefactor_terms, get_temp_adiabat, get_p_x
from typing import Tuple, Union, Optional


def get_sensitivity_factors(temp_surf_ref: Union[np.ndarray, float], r_ref: Union[np.ndarray, float],
                            pressure_surf: float, pressure_ft: float,
                            epsilon_ref: Optional[Union[np.ndarray, float]] = None,
                            z_approx_ref: Optional[Union[np.ndarray, float]] = None) -> dict:
    """
    Calculates the dimensionless sensitivity $\gamma$ parameters such that the theoretical scaling factor is given by:

    $$
    \\begin{align}
    \\frac{\delta \hat{T}_s(x)}{\delta \\tilde{T}_s} &= \gamma_{\delta T_{FT}}\\frac{\delta T_{FT}[x]}{\delta \\tilde{T}_s}
    - \gamma_{\delta r}\\frac{\\tilde{T}_s}{\\tilde{r}_s} \\frac{\delta r_s[x]}{\delta \\tilde{T}_s}
    + \gamma_{\delta \epsilon} \\frac{\delta \epsilon[x]}{c_p \delta \\tilde{T}_s} \\\\
    &+ \gamma_{\Delta T_s} \\frac{\Delta T_s(x)}{\\tilde{T}_s}
    - \gamma_{\Delta r} \\frac{\Delta r[x]}{\\tilde{r}_s}
    - \gamma_{\Delta \epsilon} \\frac{\Delta \epsilon[x]}{c_p \\tilde{T}_s}
    - \gamma_{\delta \\tilde{r}}\\frac{\delta \\tilde{r}_s}{\\tilde{r}_s}
    \\end{align}
    $$

    These $\gamma$ parameters quantify the significance of different physical mechanisms in causing a change
    in the near-surface temperature distribution.

    Terms in equation:
        * $h^{\dagger} = h^*_{FT} - R^{\dagger}T_s - gz_s = (c_p - R^{\dagger})T_s + L_v q_s - \epsilon =
            \\left(c_p + R^{\dagger}\\right) T_{FT} + L_v q^*_{FT} + A_z$
            where we used an approximate relation to replace $z_{FT}$ in $h^*_{FT}$.
        * $\epsilon = h_s - h^*_{FT}$, where $h_s$ is near-surface MSE (at $p_s$) and
            $h^*_{FT}$ is free tropospheric saturated MSE (at $p_{FT}$).
        * $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\\Delta \chi[x] = \chi[x] - \\tilde{\chi}$
        * $\chi[x]$ is the value of $\chi$ averaged over all days
            where near-surface temperature, $T_s$, is between percentile $x-0.5$ and $x+0.5$.
        * $\\tilde{\chi}$ is the reference value of $\chi$, which is free to be chosen.
        * $\\beta_{FT1} = \\frac{\partial h^{\\dagger}}{\partial T_{FT}} = c_p + R^{\dagger} + L_v \\alpha_{FT} q_{FT}^*$
        * $\\beta_{FT2} = T_{FT} \\frac{\partial^2h^{\\dagger}}{\partial T_{FT}^2} =
            T_{FT}\\frac{d\\beta_{FT1}}{d T_{FT}} = L_v \\alpha_{FT} q_{FT}^*(\\alpha_{FT} T_{FT} - 2)$
        * $\\beta_{s1} = \\frac{\partial h^{\dagger}}{\partial T_s} = c_p - R^{\dagger} + L_v \\alpha_s q_s$
        * $\\beta_{s2} = T_s \\frac{\partial^2 h^{\dagger}}{\partial T_s^2} =
            T_s\\frac{\partial \\beta_{s1}}{\partial T_s} = L_v \\alpha_s q_s(\\alpha_s T_s - 2)$
        * $\mu=\\frac{L_v \\alpha_s q_s}{\\beta_{s1}}$
        * $q = rq^*$ where $q$ is the specific humidity, $r$ is relative humidity and $q^*(T, p)$
            is saturation specific humidity which is a function of temperature and pressure.
        * $\\alpha(T, p)$ is the clausius clapeyron parameter which is a function of temperature and pressure,
            such that $\partial q^*/\partial T = \\alpha q^*$.

    Args:
        temp_surf_ref: `float [n_exp]` $\\tilde{T}_s$</br>
            Reference near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. It is assumed that `n_exp=2`.</br>
            If provide just one value, will assume `r_ref` for both experiments is the same, and thus
            the value of `temp_surf_ref` in the second (warmer) experiment does not make a difference.
        r_ref: `float [n_exp]` $\\tilde{r}_s$</br>
            Reference near surface relative humidity of each simulation. Units: dimensionless (from 0 to 1).</br>
            If provide just one value, will assume it is the same for both experiments.
        pressure_surf:
            Pressure at near-surface, $p_s$, in *Pa*.
        pressure_ft:
            Pressure at free troposphere level, $p_{FT}$, in *Pa*.
        epsilon_ref: `float [n_exp]` $\\tilde{\epsilon}_s$</br>
            Reference value of $\epsilon = h_s - h^*_{FT}$, where $h_s$ is near-surface MSE and
            $h^*_{FT}$ is saturated MSE at `pressure_ft`. If not given, weill set to 0. Units: *kJ/kg*.</br>
            If provide just one value, will assume it is the same for both experiments.
        z_approx_ref: `float [n_exp]` $\\tilde{A}_z$</br>
            The exact equation for modified MSE is given by: $h^{\dagger} = (c_p - R^{\dagger})T_s + L_v q_s
            - \epsilon = (c_p + R^{\dagger})T_{FT} + L_vq^*(T_{FT}, p_{FT}) + A_z$
            where $R^{\dagger} = R\\ln(p_s/p_{FT})/2$ and $A_z$ quantifies the error due to
            approximation of geopotential height, as relating to temperature.</br>
            Here you have the option of specifying the reference $A_z$ for each simulation. If not provided,
            will set to 0. Units: *kJ/kg*.</br>
            If provide just one value, will assume it is the same for both experiments.

    Returns:
        gamma: Dictionary containing sensitivity parameters. All are a single dimensionless `float`.
            Below, I give the equation for each parameter if
            $\\delta \\tilde{r}_s = \delta \\tilde{\epsilon} = 0$.

            * `temp_ft_change`: $\gamma_{\delta T_{FT}} = \\frac{\\tilde{\\beta}_{FT1}}{\\tilde{\\beta}_{s1}}$
            * `r_change`: $\gamma_{\delta r} = \\frac{L_v\\tilde{q}_s}{\\tilde{\\beta}_{s1} \\tilde{T}_s}$
            * `epsilon_change`: $\gamma_{\delta \epsilon} = \\frac{c_p}{\\tilde{\\beta}_{s1}}$
            * `temp_anom`: $\gamma_{\Delta T_s} = \\frac{\\tilde{\\beta}_{FT2}}{\\tilde{\\beta}_{FT1}}
                \\frac{\\tilde{\\beta}_{s1} \\tilde{T}_s}{\\tilde{\\beta}_{FT1}\\tilde{T}_{FT}} -
                \\frac{\\tilde{\\beta}_{s2}}{\\tilde{\\beta}_{s1}}$
            * `r_anom`: $\gamma_{\Delta r} = \\tilde{\mu} - \\frac{\\tilde{\\beta}_{FT2}}{\\tilde{\\beta}_{FT1}}
                \\frac{L_v \\tilde{q}_s}{\\tilde{\\beta}_{FT1}\\tilde{T}_{FT}}$
            * `epsilon_anom`: $\gamma_{\Delta \epsilon} = \\frac{\\tilde{\\beta}_{FT2}}{\\tilde{\\beta}_{FT1}}
                \\frac{c_p \\tilde{T}_s}{\\tilde{\\beta}_{FT1}\\tilde{T}_{FT}}$
            * `r_ref_change`: $\gamma_{\delta \\tilde{r}} = \\tilde{\mu}$
    """
    if isinstance(temp_surf_ref, (float, int)) and isinstance(r_ref, (float, int)):
        # If give numbers, then set r_ref change to be zero, and
        # Cannot set temp_surf_ref_change to 0, as divide by zero in gamma['epsilon_anom'] equation
        temp_surf_ref_change = 1        # arbitrarily have temp_diff=1K, could be anything, as does not contribute if r_ref_change=0
        temp_surf_ref = np.asarray([temp_surf_ref, temp_surf_ref+temp_surf_ref_change])
        r_ref_change = 0
        r_ref = np.asarray([r_ref, r_ref+r_ref_change])
        if isinstance(epsilon_ref, (list, np.ndarray)):
            # If epsilon_ref_change non-zero, then must have non-zero temp_surf_ref_change, and its value does matter.
            if epsilon_ref[1] != epsilon_ref[0]:
                raise ValueError('Cannot have epsilon_ref different for each experiment if only one temp_surf_ref provided')
    elif not isinstance(temp_surf_ref, (list, np.ndarray)) and isinstance(r_ref, (list, np.ndarray)):
        raise ValueError('`temp_surf_ref` and `r_ref` must be of same type: either both float or both number')
    n_exp = temp_surf_ref.size

    if z_approx_ref is None:
        z_approx_ref = np.zeros(n_exp)
    elif isinstance(z_approx_ref, (float, int)):
        # If give float, set same for both experiments (doesn't actually influence anything as temp_ft_ref[1] not used)
        z_approx_ref = np.full(n_exp, z_approx_ref)

    if epsilon_ref is None:
        epsilon_ref = np.zeros(n_exp)
    elif isinstance(epsilon_ref, (float, int)):
        # If give float, set same for both experiments (use epsilon_ref_change in mse_mod_ref_change0).
        epsilon_ref = np.full(n_exp, epsilon_ref)

    sphum_ref = r_ref * sphum_sat(temp_surf_ref, pressure_surf)
    temp_ft_ref = np.zeros(n_exp)
    for i in range(n_exp):
        temp_ft_ref[i] = get_temp_adiabat(temp_surf_ref[i], sphum_ref[i], pressure_surf, pressure_ft,
                                          epsilon=epsilon_ref[i] + z_approx_ref[i])

    # Get parameters required for prefactors in the theory
    _, _, _, beta_ft1, beta_ft2, _, _ = get_theory_prefactor_terms(temp_ft_ref, pressure_surf, pressure_ft)
    _, _, _, beta_s1, beta_s2, _, mu = get_theory_prefactor_terms(temp_surf_ref, pressure_surf, pressure_ft,
                                                                  sphum_ref)
    # Change in mse_mod, taking linear taylor expansion
    temp_surf_ref_change = np.diff(temp_surf_ref, axis=0).squeeze()
    r_ref_change = np.diff(r_ref, axis=0).squeeze()
    mse_mod_ref_change0 = beta_s1[0] * (1 + mu[0] * (r_ref_change/r_ref[0])
                                        ) * temp_surf_ref_change \
                          + L_v * sphum_ref[0] * (r_ref_change/r_ref[0]) \
                          - np.diff(epsilon_ref*1000, axis=0).squeeze()


    gamma = {}
    gamma['temp_ft_change'] = beta_ft1[0]/beta_s1[0]
    gamma['r_change'] = L_v * sphum_ref[0]/(beta_s1[0] * temp_surf_ref[0])
    gamma['epsilon_change'] = c_p / beta_s1[0]
    # gamma['epsilon_anom'] just becomes (beta_ft2/beta_ft1) ((c_p*T_s) / (beta_ft1*T_ft))
    # if r_ref_change = epsilon_ref_change = 0
    gamma['epsilon_anom'] = beta_ft2[0] / beta_ft1[0] ** 2 / temp_ft_ref[0] \
                            * temp_surf_ref[0] * mse_mod_ref_change0 / temp_surf_ref_change
    gamma['temp_anom'] = gamma['epsilon_anom'] - beta_s2[0]/beta_s1[0] * (1 + r_ref_change/r_ref[0]) \
                         - mu[0] * temp_surf_ref[0]/r_ref[0] * r_ref_change/temp_surf_ref_change
    gamma['r_anom'] = mu[0] * (1 + r_ref_change/r_ref[0]) \
                      - gamma['epsilon_anom'] * L_v * sphum_ref[0]/(beta_s1[0] * temp_surf_ref[0])
    gamma['r_ref_change'] = mu[0]
    # account for new non-dimensional form of gamma so divide epsilon by c_p not beta_s1 in sf equation
    gamma['epsilon_anom'] *= c_p / beta_s1[0]
    return gamma


def get_scale_factor_theory(temp_surf_ref: np.ndarray, temp_surf_quant: np.ndarray, r_ref: np.ndarray,
                            r_quant: np.ndarray, temp_ft_quant: np.ndarray,
                            epsilon_quant: np.ndarray,
                            pressure_surf: float, pressure_ft: float,
                            epsilon_ref: Optional[np.ndarray] = None,
                            z_approx_ref: Optional[np.ndarray] = None,
                            include_non_linear: bool = False) -> Tuple[np.ndarray, dict, dict, dict]:
    """
    Calculates the theoretical near-surface temperature change for percentile $x$, $\delta \hat{T}_s(x)$, relative
    to the reference temperature change, $\delta \\tilde{T}_s$. The theoretical scale factor is given by:

    $$
    \\begin{align}
    \\frac{\delta \hat{T}_s(x)}{\delta \\tilde{T}_s} &= \gamma_{\delta T_{FT}}\\frac{\delta T_{FT}[x]}{\delta \\tilde{T}_s}
    - \gamma_{\delta r}\\frac{\\tilde{T}_s}{\\tilde{r}_s} \\frac{\delta r_s[x]}{\delta \\tilde{T}_s}
    + \gamma_{\delta \epsilon} \\frac{\delta \epsilon[x]}{c_p \delta \\tilde{T}_s} \\\\
    &+ \gamma_{\Delta T_s} \\frac{\Delta T_s(x)}{\\tilde{T}_s}
    - \gamma_{\Delta r} \\frac{\Delta r[x]}{\\tilde{r}_s}
    - \gamma_{\Delta \epsilon} \\frac{\Delta \epsilon[x]}{c_p \\tilde{T}_s}
    - \gamma_{\delta \\tilde{r}}\\frac{\delta \\tilde{r}_s}{\\tilde{r}_s}
    \\end{align}
    $$

    where the dimensionless $\gamma$ parameters quantify the significance of different physical mechanisms in causing
    a change in the near-surface temperature distribution. These are given by the `get_sensitivity_factors` function.

    The approximations which cause $\\frac{\delta \hat{T}_s(x)}{\delta \\tilde{T}_s}$ to differ from the exact
    scale factor are given in `get_approx_terms`.

    Reference Quantities:
        The reference quantities, $\\tilde{\chi}$ are free to be chosen by the user. For ease of interpretation,
        I propose the following, where $\overline{\chi}$ is the mean value of $\chi$ across all days:

        * $\\tilde{T}_s = \overline{T_s}; \delta \\tilde{T}_s = \delta \overline{T_s}$
        * $\\tilde{r}_s = \overline{r_s}; \delta \\tilde{r}_s = 0$
        * $\\tilde{\epsilon} = 0; \delta \\tilde{\epsilon} = 0$
        * $\\tilde{A}_z = \overline{A}_z; \delta \\tilde{A}_z = 0$

        Given the choice of these four reference variables and their changes with warming, the reference free
        troposphere temperature, $\\tilde{T}_{FT}$, can be computed according to the definition of $\\tilde{h}^{\dagger}$:

        $\\tilde{h}^{\dagger} = (c_p - R^{\dagger})\\tilde{T}_s + L_v \\tilde{q}_s - \\tilde{\epsilon} =
            (c_p + R^{\dagger}) \\tilde{T}_{FT} + L_v q^*(\\tilde{T}_{FT}, p_{FT}) + \\tilde{A}_z$

        Poor choice of reference quantities may cause the theoretical scale factor to be a bad approximation. If this
        is the case, `get_approx_terms` can be used to investigate what is causing the theory to break down.

    Terms in equation:
        * $h^{\dagger} = h^*_{FT} - R^{\dagger}T_s - gz_s = (c_p - R^{\dagger})T_s + L_v q_s - \epsilon =
            (c_p + R^{\dagger}) T_{FT} + L_v q^*_{FT} + A_z$
            where we used an approximate relation to replace $z_{FT}$ in $h^*_{FT}$.
        * $\epsilon = h_s - h^*_{FT}$, where $h_s$ is near-surface MSE (at $p_s$) and
            $h^*_{FT}$ is free tropospheric saturated MSE (at $p_{FT}$).
        * $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\\Delta \chi[x] = \chi[x] - \\tilde{\chi}$
        * $\chi[x]$ is the value of $\chi$ averaged over all days
            where near-surface temperature, $T_s$, is between percentile $x-0.5$ and $x+0.5$.
        * $\\tilde{\chi}$ is the reference value of $\chi$, which is free to be chosen.
        * $\\beta_{FT1} = \\frac{\partial h^{\\dagger}}{\partial T_{FT}} = c_p + R^{\dagger} + L_v \\alpha_{FT} q_{FT}^*$
        * $\\beta_{FT2} = T_{FT} \\frac{\partial^2h^{\\dagger}}{\partial T_{FT}^2} =
            T_{FT}\\frac{d\\beta_{FT1}}{d T_{FT}} = L_v \\alpha_{FT} q_{FT}^*(\\alpha_{FT} T_{FT} - 2)$
        * $\\beta_{s1} = \\frac{\partial h^{\dagger}}{\partial T_s} = c_p - R^{\dagger} + L_v \\alpha_s q_s$
        * $\\beta_{s2} = T_s \\frac{\partial^2 h^{\dagger}}{\partial T_s^2} =
            T_s\\frac{\partial \\beta_{s1}}{\partial T_s} = L_v \\alpha_s q_s(\\alpha_s T_s - 2)$
        * $\mu=\\frac{L_v \\alpha_s q_s}{\\beta_{s1}}$
        * $q = rq^*$ where $q$ is the specific humidity, $r$ is relative humidity and $q^*(T, p)$
            is saturation specific humidity which is a function of temperature and pressure.
        * $\\alpha(T, p)$ is the clausius clapeyron parameter which is a function of temperature and pressure,
            such that $\partial q^*/\partial T = \\alpha q^*$.

    Args:
        temp_surf_ref: `float [n_exp]` $\\tilde{T}_s$</br>
            Reference near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]` $T_s(x)$ </br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        r_ref: `float [n_exp]` $\\tilde{r}_s$</br>
            Reference near surface relative humidity of each simulation. Units: dimensionless (from 0 to 1).
        r_quant: `float [n_exp, n_quant]` $r_s[x]$</br>
            `r_quant[i, j]` is near-surface relative humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: dimensionless.
        temp_ft_quant: `float [n_exp, n_quant]` $T_{FT}[x]$</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        epsilon_quant: `float [n_exp, n_quant]` $\epsilon[x]$</br>
            `epsilon_quant[i, j]` is $\epsilon = h_s - h^*_{FT}$, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kJ/kg*.
        pressure_surf:
            Pressure at near-surface, $p_s$, in *Pa*.
        pressure_ft:
            Pressure at free troposphere level, $p_{FT}$, in *Pa*.
        epsilon_ref: `float [n_exp]` $\\tilde{\epsilon}_s$</br>
            Reference value of $\epsilon = h_s - h^*_{FT}$, where $h_s$ is near-surface MSE and
            $h^*_{FT}$ is saturated MSE at `pressure_ft`. If not given, weill set to 0. Units: *kJ/kg*.
        z_approx_ref: `float [n_exp]` $\\tilde{A}_z$</br>
            The exact equation for modified MSE is given by: $h^{\dagger} = (c_p - R^{\dagger})T_s + L_v q_s
            - \epsilon = (c_p + R^{\dagger})T_{FT} + L_vq^*(T_{FT}, p_{FT}) + A_z$
            where $R^{\dagger} = R\\ln(p_s/p_{FT})/2$ and $A_z$ quantifies the error due to
            approximation of geopotential height, as relating to temperature.</br>
            Here you have the option of specifying the reference $A_z$ for each simulation. If not provided,
            will set to 0. Units: *kJ/kg*.
        include_non_linear: If `True`, will include the approximate values of
            $A_{\delta \Delta T_{FT}}$, $A_{\delta r}[x]$ and $A_{\Delta T_s \Delta r}[x]$ in `info_cont` with the
            names `nl_temp_ft_anom_change`, `nl_r_change` and `nl_anom_temp_s_r` respectively.
            These are obtained using `get_approx_terms` with `simple=True`.
            These will also be included in the `scale_factor` theory.

    Returns:
        scale_factor: `float [n_quant]`</br>
            `scale_factor[i]` refers to the theoretical temperature difference between experiments
            for percentile `quant_use[i]`, relative to the reference temperature change, $\delta \\tilde{T_s}$.
        gamma: This is the dictionary output by `get_sensitivity_factors`
        info_var: For each `key` in `gamma`, this dictionary has an entry for the same `key` which equals the dimensionless
            variable which multiplies `gamma[key]` in the equation for $\\frac{\delta \hat{T}_s(x)}{\delta \\tilde{T}_s}$:

            * `temp_ft_change`: $\\frac{\delta T_{FT}[x]}{\delta \\tilde{T}_s}$
            * `r_change`: $\\frac{\\tilde{T}_s}{\\tilde{r}_s} \\frac{\delta r_s[x]}{\delta \\tilde{T}_s}$
            * `epsilon_change`: $\\frac{\delta \epsilon[x]}{\\tilde{\\beta}_{s1} \delta \\tilde{T}_s}$
            * `temp_anom`: $\\frac{\Delta T_s(x)}{\\tilde{T}_s}$
            * `r_anom`: $\\frac{\Delta r[x]}{\\tilde{r}_s}$
            * `epsilon_anom`: $\\frac{\Delta \epsilon[x]}{c_p \\tilde{T}_s}$
            * `r_ref_change`: $\\frac{\delta \\tilde{r}_s}{\\tilde{r}_s}$

            All are arrays of size `float [n_quant]`, except `r_ref_change` which is just a single `float`.
        info_cont: Dictionary containing `gamma[key] x info_var[key]` for each `key` in `gamma`. This gives
            the contribution from each physical mechanism to the overall scale factor.</br>
            If `include_non_linear=True`, will also include `nl_temp_ft_anom_change`, `nl_r_change`
            and `nl_anom_temp_s_r` which arise from including the most significant approximations.

    """
    n_exp = temp_surf_ref.size
    if epsilon_ref is None:
        epsilon_ref = np.zeros(n_exp)
    gamma = get_sensitivity_factors(temp_surf_ref, r_ref, pressure_surf, pressure_ft, epsilon_ref, z_approx_ref)
    sphum_ref = r_ref * sphum_sat(temp_surf_ref, pressure_surf)
    _, _, _, beta_s1, beta_s2, _, mu = get_theory_prefactor_terms(temp_surf_ref, pressure_surf, pressure_ft,
                                                                  sphum_ref)
    temp_surf_ref_change = np.diff(temp_surf_ref, axis=0).squeeze()
    # Get non-dimensional variables which multiply gamma
    # Multiply epsilon by 1000 to get in correct units of J/kg
    info_var = {'r_ref_change': np.diff(r_ref, axis=0).squeeze()/r_ref[0],
                'temp_ft_change': np.diff(temp_ft_quant, axis=0).squeeze()/temp_surf_ref_change,
                'r_change': np.diff(r_quant, axis=0).squeeze()/r_ref[0, np.newaxis] * temp_surf_ref[0]/temp_surf_ref_change,
                'epsilon_change': np.diff(epsilon_quant, axis=0).squeeze()*1000/c_p/temp_surf_ref_change,
                'temp_anom': (temp_surf_quant[0]-temp_surf_ref[0]) / temp_surf_ref[0],
                'r_anom': (r_quant[0]-r_ref[0]) / r_ref[0],
                'epsilon_anom': (epsilon_quant[0]-epsilon_ref[0])*1000 / c_p / temp_surf_ref[0]}
    # All gamma are positive, so sign below is to multiply gamma in equation
    coef_sign = {'r_ref_change': -1, 'temp_ft_change': 1, 'r_change': -1, 'epsilon_change': 1,
                 'temp_anom': 1, 'r_anom': -1, 'epsilon_anom': -1}

    # Get contribution from each term
    info_cont = {}
    for key in gamma:
        info_cont[key] = coef_sign[key] * gamma[key] * info_var[key]

    if include_non_linear:
        # Add non-linear terms
        approx_var = get_approx_terms(temp_surf_ref, temp_surf_quant, r_ref, r_quant, temp_ft_quant,
                                      epsilon_quant, pressure_surf, pressure_ft, epsilon_ref,
                                      z_approx_ref, simple=True)[0]
        for key in ['temp_ft_anom_change', 'r_change', 'anom_temp_s_r']:
            info_cont['nl_'+key] = approx_var[key]

    final_answer = np.asarray(sum([info_cont[key] for key in info_cont]))
    return final_answer, gamma, info_var, info_cont


def get_approx_terms(temp_surf_ref: np.ndarray, temp_surf_quant: np.ndarray, r_ref: np.ndarray,
                     r_quant: np.ndarray, temp_ft_quant: np.ndarray,
                     epsilon_quant: np.ndarray,
                     pressure_surf: float, pressure_ft: float,
                     epsilon_ref: Optional[np.ndarray] = None,
                     z_approx_ref: Optional[np.ndarray] = None,
                     simple: bool = False) -> Tuple[dict, dict]:
    """
    Function which returns terms quantifying the errors associated with various approximations, grouped together in
    $A$ variables that go into the derivation of the theoretical scaling factor,
    $\delta \hat{T}_s(x)/\delta \\tilde{T}_s$, returned by `get_scale_factor_theory`.

    The exact scaling factor is given by:

    $$
    \\begin{align}
    \\frac{\delta T_s(x)}{\delta \\tilde{T}_s} &= \\frac{\delta \hat{T}_s(x)}{\delta \\tilde{T}_s} + A_{\delta \Delta T_{FT}}
    + A_{\delta \Delta T_s} + A_{\delta r}[x] + A_{\delta \Delta T_s \delta r}[x] \\\\
    &+ A_{\Delta T_s \Delta r}[x] + A_{\Delta}[x] + \\tilde{A}_{\delta} + A_{NL}[x]
    \\end{align}
    $$

    For more details on the approximations, there is a
    [Jupyter notebook](https://github.com/jduffield65/Isca/blob/main/jobs/tau_sweep/land/meridional_band/publish_figures/theory_approximations2.ipynb)
    that goes through each step of the derivation.

    Terms in equation:
        * $h^{\dagger} = h^*_{FT} - R^{\dagger}T_s - gz_s = (c_p - R^{\dagger})T_s + L_v q_s - \epsilon =
            \\left(c_p + R^{\dagger}\\right) T_{FT} + L_v q^*_{FT} + A_z$
            where we used an approximate relation to replace $z_{FT}$ in $h^*_{FT}$.
        * $\epsilon = h_s - h^*_{FT}$, where $h_s$ is near-surface MSE (at $p_s$) and
            $h^*_{FT}$ is free tropospheric saturated MSE (at $p_{FT}$).
        * $R^{\dagger} = R\\ln(p_s/p_{FT})/2$
        * $\\Delta \chi[x] = \chi[x] - \\tilde{\chi}$
        * $\chi[x]$ is the value of $\chi$ averaged over all days
            where near-surface temperature, $T_s$, is between percentile $x-0.5$ and $x+0.5$.
        * $\\tilde{\chi}$ is the reference value of $\chi$, which is free to be chosen.
        * $\\beta_{FT1} = \\frac{\partial h^{\\dagger}}{\partial T_{FT}} = c_p + R^{\dagger} + L_v \\alpha_{FT} q_{FT}^*$
        * $\\beta_{FT2} = T_{FT} \\frac{\partial^2h^{\\dagger}}{\partial T_{FT}^2} =
            T_{FT}\\frac{d\\beta_{FT1}}{d T_{FT}} = L_v \\alpha_{FT} q_{FT}^*(\\alpha_{FT} T_{FT} - 2)$
        * $\\beta_{s1} = \\frac{\partial h^{\dagger}}{\partial T_s} = c_p - R^{\dagger} + L_v \\alpha_s q_s$
        * $\\beta_{s2} = T_s \\frac{\partial^2 h^{\dagger}}{\partial T_s^2} =
            T_s\\frac{\partial \\beta_{s1}}{\partial T_s} = L_v \\alpha_s q_s(\\alpha_s T_s - 2)$
        * $\mu=\\frac{L_v \\alpha_s q_s}{\\beta_{s1}}$
        * $q = rq^*$ where $q$ is the specific humidity, $r$ is relative humidity and $q^*(T, p)$
            is saturation specific humidity which is a function of temperature and pressure.
        * $\\alpha(T, p)$ is the clausius clapeyron parameter which is a function of temperature and pressure,
            such that $\partial q^*/\partial T = \\alpha q^*$.
        * $\Delta h^{\dagger}_0$ is referred to as `mse_mod_anom0` in the code, and is defined through:</br>
            $\Delta h^{\dagger}[x] = \\tilde{\\beta}_{s1}\\left(1+\\tilde{\mu}\\frac{\Delta r_s[x]}{\\tilde{r}_s}\\right)
            \Delta T_s[x] + L_v \\tilde{q}_s\\frac{\Delta r_s[x]}{\\tilde{r}_s} - \Delta \epsilon[x] + A_{s\Delta}[x]=
            \Delta h^{\dagger}_0[x] + A_{s\Delta}[x]$
        * $\delta \\tilde{h}^{\dagger}_0$ is referred to as `mse_mod_ref_change0` in the code, and is defined through:</br>
            $\delta \\tilde{h}^{\dagger} =
            \\tilde{\\beta}_{s1}\\left(1+\\tilde{\mu}\\frac{\delta \\tilde{r}_s}{\\tilde{r}_s}\\right)\delta \\tilde{T}_s+
            L_v \\tilde{q}_s\\frac{\delta \\tilde{r}_s}{\\tilde{r}_s} - \delta \\tilde{\epsilon} + \\tilde{A}_{s\delta}=
            \delta \\tilde{h}^{\dagger}_0 + \\tilde{A}_{s\delta}$
        *  $\delta \Delta T_{FT}'[x]$ is referred to as `temp_ft_anom_change_mod` in the code, and is defined through:</br>
            $\\tilde{\\beta}_{FT1}\delta \Delta T_{FT}'[x] = \\tilde{\\beta}_{FT1} \delta T_{FT}[x] - \delta \\tilde{h}^{\dagger}_0$</br>
            The idea being that $\delta \\tilde{T}_{FT} \\approx \delta \\tilde{h}^{\dagger}_0 / \\tilde{\\beta}_{FT1}$.

    Args:
        temp_surf_ref: `float [n_exp]` $\\tilde{T}_s$</br>
            Reference near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]` $T_s(x)$ </br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        r_ref: `float [n_exp]` $\\tilde{r}_s$</br>
            Reference near surface relative humidity of each simulation. Units: dimensionless (from 0 to 1).
        r_quant: `float [n_exp, n_quant]` $r_s[x]$</br>
            `r_quant[i, j]` is near-surface relative humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: dimensionless.
        temp_ft_quant: `float [n_exp, n_quant]` $T_{FT}[x]$</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        epsilon_quant: `float [n_exp, n_quant]` $\epsilon[x]$</br>
            `epsilon_quant[i, j]` is $\epsilon = h_s - h^*_{FT}$, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kJ/kg*.
        pressure_surf:
            Pressure at near-surface, $p_s$, in *Pa*.
        pressure_ft:
            Pressure at free troposphere level, $p_{FT}$, in *Pa*.
        epsilon_ref: `float [n_exp]` $\\tilde{\epsilon}_s$</br>
            Reference value of $\epsilon = h_s - h^*_{FT}$, where $h_s$ is near-surface MSE and
            $h^*_{FT}$ is saturated MSE at `pressure_ft`. If not given, weill set to 0. Units: *kJ/kg*.
        z_approx_ref: `float [n_exp]` $\\tilde{A}_z$</br>
            The exact equation for modified MSE is given by: $h^{\dagger} = (c_p - R^{\dagger})T_s + L_v q_s
            - \epsilon = (c_p + R^{\dagger})T_{FT} + L_vq^*(T_{FT}, p_{FT}) + A_z$
            where $R^{\dagger} = R\\ln(p_s/p_{FT})/2$ and $A_z$ quantifies the error due to
            approximation of geopotential height, as relating to temperature.</br>
            Here you have the option of specifying the reference $A_z$ for each simulation. If not provided,
            will set to 0. Units: *kJ/kg*.
        simple: If `True`, will return approximate values of $A_{\delta \Delta T_{FT}}$, $A_{\delta r}[x]$ and
            $A_{\Delta T_s \Delta r}[x]$. Idea being that these tend to be the most significant, so these approximate
            values can then be incorporated into theory for scale factor.
            The approximate terms correspond to non-linear combinations of different physical mechanisms, e.g.
            combined effect of relative humidity anomaly in current climate and free tropospheric change with warming:
            $\Delta r_s[x] \delta \Delta T_{FT}[x]$.

    Returns:
        approx_terms: Dictionary containing approximations associated with final scaling factor,
            $\delta T_s(x)/\delta \\tilde{T}_s$ so units are *K/K*. Terms have been named based on what causes the
            variation in $x$. Each value in dictionary is a `float [n_quant]` array, except
            `ref_change` which is a `float`.

            * `temp_ft_anom_change`: $A_{\delta \Delta T_{FT}}$</br>
                Involves contribution from $\delta \Delta T_{FT}[x]$.
                If `simple=True`, will return approximate form:</br>
                $A_{\delta \Delta T_{FT}} \\tilde{\\beta}_{s1}\delta \\tilde{T}_s \\approx
                \\frac{\\tilde{\\beta}_{FT2}}{2\\tilde{\\beta}_{FT1}} \\left(\\frac{\delta T_{FT}[x]}{\\tilde{T}_{FT}} -
                \\frac{\delta \\tilde{h}^{\dagger}_0}{\\tilde{\\beta}_{FT1}\\tilde{T}_{FT}}\\right)
                (\delta \\tilde{h}^{\dagger}_0 + 2\Delta \\tilde{h}^{\dagger}_0[x] + \\tilde{\\beta}_{FT1}\delta T_{FT}[x])$</br>
                Note that $\Delta \\tilde{h}^{\dagger}_0[x]$ can be decomposed to give relative contributions
                of different anomalies in current climate: $\Delta T_s[x], \Delta r_s[x], \Delta \epsilon[x]$.
            * `temp_s_anom_change`: $A_{\delta \Delta T_s}$</br>
                Involves contribution from $\delta \Delta T_s(x)$.
            * `r_change`: $A_{\delta r}[x]$</br>
                Involves contribution from $\delta r_s[x]$ and $\delta \\tilde{r}_s$.
                If `simple=True`, will return approximate form:</br>
                $A_{\delta r}[x] \\tilde{\\beta}_{s1}\delta \\tilde{T}_s \\approx -
                \\tilde{\mu}\\tilde{\\beta}_{s1}\\left(\delta \\tilde{T}_s + \Delta T_s[x]\\right)
                \\frac{\delta r_s[x]}{\\tilde{r}_s}$
            * `temp_s_anom_r_change`: $A_{\delta \Delta T_s \delta r}[x]$</br>
                Involves contribution from $\delta \Delta T_s(x) \delta (r_s[x]/\\tilde{r}_s)$.
                If `simple=True`, will return approximate form:</br>
                $A_{\Delta T_s \Delta r} \\tilde{\\beta}_{s1}\delta \\tilde{T}_s \\approx
                \\left[\\frac{\\tilde{\\beta}_{FT2}}{\\tilde{\\beta}_{FT1}}
                \\frac{\delta \\tilde{h}^{\dagger}_0}{\\tilde{\\beta}_{FT1}\\tilde{T}_{FT}}
                \\tilde{\mu} \\tilde{\\beta}_{s1} \\tilde{T}_s -
                \\tilde{\\beta}_{s2}\delta \\tilde{T}_s\\right]\\frac{\Delta r_s[x]}{\\tilde{r}_s}
                \\frac{\Delta T_s[x]}{\\tilde{T}_s}$
            * `anom_temp_s_r`: $A_{\Delta T_s \Delta r}[x]$</br>
                Involves contribution from $\Delta T_s(x) \Delta r_s[x]$ in the current climate.
            * `anom`: $A_{\Delta}[x]$</br> Groups together errors due to approximation of anomaly in current climate.
                Excludes those in $A_{\Delta T_s \Delta r}$.
            * `ref_change`: $\\tilde{A}_{\delta}$</br>
                Groups together errors due to change with warming of the reference day quantities.
            * `z_anom_change`: $\delta \Delta A_z[x]$</br>
                Quantifies how error due to approximation of geopotential height changes with warming.
            * `nl`: $A_{NL}[x]$</br>
                Residual non-linear errors that don't directly correspond to any of the above.
        approx: Approximation, $A$, terms which arise through the derivation of the theory.
            All have units of *J/kg*, except `ft_beta` and `s_beta` which are both dimensionless:

            * `z_ref`: `float [n_exp]`</br>Same as `z_approx_ref` input, $\\tilde{A}_z$ (set to 0 if not provided).
            * `z_quant`: `float [n_exp, n_quant]`</br>Error associated with geopotential height approx on quantile day,
                $A_z[x]$, computed from provided variables according to
                $h^{\dagger}[x] = (c_p - R^{\dagger})T_s[x] + L_v q_s[x] - \epsilon = (c_p + R^{\dagger})T_{FT}[x] + L_vq^*(T_{FT}[x], p_{FT}) + A_z[x]$
            * `z_anom`: `float [n_exp, n_quant]`</br>$\Delta A_z[x] = A_z[x] - \\tilde{A}_z$.
            * `ft_anom`: `float [n_exp, n_quant]`</br> Approximation associated with anomaly of $h^{\dagger}$ at FT level:
                $A_{FT\Delta}[x] = \sum_{n=2}^{\infty}\\frac{1}{n!}\\frac{\partial^n h^{\dagger}}{\partial T_{FT}^n}(\Delta T_{FT})^n$
            * `ft_beta`: `float`</br> $\\tilde{A}_{FT\\beta}$ such that
                $\delta \\tilde{\\beta}_{FT1} = \\tilde{\\beta}_{FT2}(1 + \\tilde{A}_{FT\\beta})\\frac{\delta \\tilde{T}_{FT}}{\\tilde{T}_{FT}}$
                where $\\beta_{FT1} = \\frac{\partial h^{\dagger}}{\partial T_{FT}}$ and
                $\\beta_{FT2} = T_{FT} \\frac{\partial^2 h^{\dagger}}{\partial T_{FT}^2}$.
            * `ft_ref_change`: `float`</br> Approximation associated with change of $h^{\dagger}$ with warming at FT level:
                $\\tilde{A}_{FT\delta} = \sum_{n=2}^{\infty}\\frac{1}{n!}\\frac{\partial^n h^{\dagger}}{\partial T_{FT}^n}(\delta \\tilde{T}_{FT})^n$
            * `s_anom`: `float [n_exp, n_quant]`</br> Approximation associated with anomaly of $h^{\dagger}$ at surface:
                $A_{s\Delta}[x] = (1 + \\frac{\Delta r_s[x]}{\\tilde{r}_s})A_{s\Delta T}[x]$ where
                $A_{s\Delta T}[x] = \sum_{n=2}^{\infty}\\frac{1}{n!}\\frac{\partial^n h^{\dagger}}{\partial T_{s}^n}(\Delta T_{s})^n$
            * `s_anom_temp_cont`: `float [n_exp, n_quant]`</br> Temperature only contribution to `s_anom`:
                $A_{s\Delta T}[x] = \sum_{n=2}^{\infty}\\frac{1}{n!}\\frac{\partial^n h^{\dagger}}{\partial T_{s}^n}(\Delta T_{s})^n$.
            * `s_anom_temp_r_cont`: `float [n_exp, n_quant]`</br> Temperature and relative humidity non-linear contribution
                to `s_anom`: $A_{s\Delta T \Delta r}[x] = \\frac{\Delta r_s[x]}{\\tilde{r}_s}A_{s\Delta T}[x]$.
            * `s_anom_change_temp_cont`: `float [n_quant]`</br> Temperature only contribution due to change in `s_anom` with warming:
                $A_{s\delta \Delta T}[x] = (1 + \\frac{\Delta r_s[x]}{\\tilde{r}_s})\delta A_{s\Delta T}[x]$
            * `s_anom_change_r_cont`: `float [n_quant]`</br> Relative humidity only contribution due to change in `s_anom` with warming:
                $A_{s\delta \Delta r}[x] = \delta (\\frac{\Delta r_s[x]}{\\tilde{r}_s}) A_{s\Delta T}[x]$
            * `s_anom_change_temp_r_cont`: `float [n_quant]`</br> Temperature and relative humidity non-linear contribution
                due to change in `s_anom` with warming:
                $A_{s\delta \Delta T \delta \Delta r}[x] = \delta (\\frac{\Delta r_s[x]}{\\tilde{r}_s}) \delta A_{s\Delta T}[x]$
            * `s_beta`: `float`</br> $\\tilde{A}_{s\\beta}$ such that
                $\delta \\tilde{\\beta}_{s1} = \\tilde{\\beta}_{s2}(1 + \\tilde{A}_{FT\\beta})\\left(1 + \\frac{\delta \\tilde{r}_s}{\\tilde{r}_s}\\right)\\frac{\delta \\tilde{T}_s}{\\tilde{T}_s} + \\tilde{\mu}\\tilde{\\beta}_{s1} \\frac{\delta \\tilde{r}_s}{\\tilde{r}_s}$
            * `s_ref_change`: `float`</br> Approximation associated with change of $h^{\dagger}$ with warming at surface:
                $\\tilde{A}_{s\delta}[x] = (1 + \\frac{\delta \\tilde{r}_s[x]}{\\tilde{r}_s})\sum_{n=2}^{\infty}\\frac{1}{n!}\\frac{\partial^n h^{\dagger}}{\partial T_{s}^n}(\delta \\tilde{T}_{s})^n$

    """
    n_exp, n_quant = temp_surf_quant.shape
    if z_approx_ref is None:
        z_approx_ref = np.zeros(n_exp)
    if epsilon_ref is None:
        epsilon_ref = np.zeros(n_exp)

    sphum_ref = r_ref * sphum_sat(temp_surf_ref, pressure_surf)
    sphum_quant = r_quant * sphum_sat(temp_surf_quant, pressure_surf)
    temp_ft_ref = np.zeros(n_exp)
    for i in range(n_exp):
        temp_ft_ref[i] = get_temp_adiabat(temp_surf_ref[i], sphum_ref[i], pressure_surf, pressure_ft,
                                          epsilon=epsilon_ref[i] + z_approx_ref[i])

    R_mod, _, _, beta_ft1, beta_ft2, _, _ = get_theory_prefactor_terms(temp_ft_ref, pressure_surf, pressure_ft)
    _, _, _, beta_s1, beta_s2, _, mu = get_theory_prefactor_terms(temp_surf_ref, pressure_surf, pressure_ft, sphum_ref)

    # For everything below, deal in units of J/kg
    mse_mod_quant = (moist_static_energy(temp_surf_quant, sphum_quant, height=0,
                                         c_p_const=c_p - R_mod) - epsilon_quant) * 1000
    mse_mod_ref = (moist_static_energy(temp_surf_ref, sphum_ref, height=0, c_p_const=c_p - R_mod) - epsilon_ref) * 1000
    mse_mod_anom = mse_mod_quant - mse_mod_ref[:, np.newaxis]
    temp_surf_anom = temp_surf_quant - temp_surf_ref[:, np.newaxis]
    temp_ft_anom = temp_ft_quant - temp_ft_ref[:, np.newaxis]
    r_anom = r_quant - r_ref[:, np.newaxis]
    epsilon_anom = (epsilon_quant - epsilon_ref[:, np.newaxis]) * 1000

    approx = {}
    # Z error - The starting equation for mse_mod approximates the geopotential height. We quantify that here.
    approx['z_quant'] = mse_mod_quant - moist_static_energy(temp_ft_quant, sphum_sat(temp_ft_quant, pressure_ft),
                                                            height=0, c_p_const=c_p+R_mod)*1000
    approx['z_ref'] = z_approx_ref * 1000
    approx['z_anom'] = approx['z_quant'] - approx['z_ref'][:, np.newaxis]

    # Expansion of mse_mod about ref at FT. I.e. in approximating mse_mod_anom for each simulation.
    approx['ft_anom'] = mse_mod_anom - approx['z_anom'] - beta_ft1[:, np.newaxis] * temp_ft_anom

    # Change with warming of mse_mod_anom at FT level has contribution from change in beta_ft1 - dimensionless
    approx['ft_beta'] = np.diff(beta_ft1, axis=0).squeeze() * temp_ft_ref[0] \
                        / np.diff(temp_ft_ref, axis=0).squeeze() / beta_ft2[0] - 1

    # Change in mse_mod_ref with warming at FT level
    approx['ft_ref_change'] = np.diff(mse_mod_ref, axis=0).squeeze() - beta_ft1[0] * \
                              np.diff(temp_ft_ref, axis=0).squeeze() - np.diff(approx['z_ref'], axis=0).squeeze()

    # Expansion of mse_mod about ref at Surface. I.e. in approximating mse_mod_anom for each simulation.
    approx['s_anom'] = mse_mod_anom - beta_s1[:, np.newaxis] * (
            1 + mu[:, np.newaxis] * (r_anom / r_ref[:, np.newaxis])) * temp_surf_anom - \
                       L_v * sphum_ref[:, np.newaxis] * (r_anom / r_ref[:, np.newaxis]) + epsilon_anom
    approx['s_anom_temp_cont'] = approx['s_anom'] / (1 + (r_anom / r_ref[:, np.newaxis]))
    # Decompose change with warming into contributions from temp, RH and NL
    approx['s_anom_change_temp_cont'] = (1 + (r_anom / r_ref[:, np.newaxis])[0]) * np.diff(
        approx['s_anom_temp_cont'], axis=0).squeeze()
    approx['s_anom_change_r_cont'] = np.diff((r_anom / r_ref[:, np.newaxis]), axis=0).squeeze() * \
                                     approx['s_anom_temp_cont'][0]
    approx['s_anom_change_temp_r_cont'] = np.diff((r_anom / r_ref[:, np.newaxis]), axis=0).squeeze() * np.diff(
        approx['s_anom_temp_cont'], axis=0).squeeze()

    # Change with warming of mse_mod_anom at Surface has contribution from change in beta_s1 - dimensionless
    approx['s_beta'] = (np.diff(beta_s1, axis=0).squeeze() - mu[0] * beta_s1[0] * np.diff(r_ref, axis=0).squeeze() /
                        r_ref[0]) * temp_surf_ref[0] / np.diff(temp_surf_ref, axis=0).squeeze() / (
                                   1 + np.diff(r_ref, axis=0).squeeze() / r_ref[0]) / beta_s2[0] - 1

    # Change in mse_mod_ref with warming at Surface
    # Remember to convert epsilon to J/kg in this calculation
    approx['s_ref_change'] = np.diff(mse_mod_ref, axis=0).squeeze() - beta_s1[0] * (
            1 + mu[0] * (np.diff(r_ref, axis=0).squeeze() / r_ref[0])
    ) * np.diff(temp_surf_ref, axis=0).squeeze() - L_v * sphum_ref[0] * (np.diff(r_ref, axis=0).squeeze() / r_ref[0]) \
                             + np.diff(epsilon_ref*1000, axis=0).squeeze()

    # Combine approximations into terms which contribute to final scaling factor
    prefactor_mse_ft = beta_ft2[0] / beta_ft1[0] ** 2 / temp_ft_ref[0]
    mse_mod_ref_change0 = np.diff(mse_mod_ref, axis=0).squeeze() - approx['s_ref_change']
    mse_mod_anom0 = mse_mod_anom[0] - approx['s_anom'][0]
    temp_ft_anom_change_mod = np.diff(temp_ft_quant, axis=0).squeeze() - (mse_mod_ref_change0 / beta_ft1[0])
    var_ref_change = approx['s_ref_change'] - approx['ft_ref_change'] - np.diff(approx['z_ref'], axis=0).squeeze()
    var_anom0 = approx['s_anom'][0] - approx['ft_anom'][0] - approx['z_anom'][0]

    # For details of different terms, see theory_approximations2 notebook
    approx_terms = {}
    # From FT derivation
    if simple:
        approx_terms['temp_ft_anom_change'] = prefactor_mse_ft * beta_ft1[0] * (
                mse_mod_ref_change0 + mse_mod_anom0 + 0.5*beta_ft1[0]*temp_ft_anom_change_mod) * temp_ft_anom_change_mod
    else:
        approx_terms['temp_ft_anom_change'] = np.diff(approx['ft_anom'], axis=0).squeeze() + \
                                              prefactor_mse_ft * beta_ft1[0] * (1 + approx['ft_beta']) * (
                                                      mse_mod_ref_change0 + var_ref_change) * temp_ft_anom_change_mod
    approx_terms['z_anom_change'] = np.diff(approx['z_anom'], axis=0).squeeze()
    approx_terms['ref_change'] =  -var_ref_change - prefactor_mse_ft * (1+approx['ft_beta']) * var_ref_change * (
        mse_mod_ref_change0 + var_ref_change)
    approx_terms['nl'] = prefactor_mse_ft * ((approx['ft_beta'] * mse_mod_ref_change0) +
                                              (1+approx['ft_beta']) * var_ref_change) * var_anom0
    approx_terms['anom'] = prefactor_mse_ft * mse_mod_ref_change0 * (var_anom0 + approx['ft_beta'] * mse_mod_anom0) \
                           + (prefactor_mse_ft * (1+approx['ft_beta']) * var_ref_change) * mse_mod_anom0
    approx_terms['anom_temp_s_r'] = prefactor_mse_ft * mse_mod_ref_change0 * beta_s1[0] * mu[0] * \
                                    r_anom[0]/r_ref[0] * temp_surf_anom[0]

    # From Surface derivation
    approx_terms['anom'] -= (approx['s_beta'] * beta_s2[0] * (1 + np.diff(r_ref, axis=0).squeeze()/r_ref[0]) *
                                               np.diff(temp_surf_ref, axis=0).squeeze()/temp_surf_ref[0]
                                               ) * temp_surf_anom[0] + (approx['s_ref_change']/r_ref[0]) * r_anom[0]
    if simple:
        approx_terms['anom_temp_s_r'] -= beta_s2[0] * np.diff(temp_surf_ref, axis=0).squeeze() * r_anom[0]/r_ref[0] * \
                                         temp_surf_anom[0]/temp_surf_ref[0]
    else:
        approx_terms['anom_temp_s_r'] -= np.diff(beta_s1, axis=0).squeeze() * \
                                       r_anom[0]/r_ref[0] * temp_surf_anom[0]
    approx_terms['temp_s_anom_change'] = - approx['s_anom_change_temp_cont'] - (
            (mu*beta_s1/r_ref)[0]*r_anom[0] + (1+r_anom[0]/r_ref[0])*np.diff(beta_s1, axis=0).squeeze()
    )* np.diff(temp_surf_anom, axis=0).squeeze()

    if simple:
        approx_terms['r_change'] = -mu[0]*beta_s1[0] * (np.diff(temp_surf_ref, axis=0).squeeze()+temp_surf_anom[0]) * \
                                   np.diff(r_anom/r_ref[:, np.newaxis], axis=0).squeeze()
    else:
        # r_change1 term
        approx_terms['r_change'] = -approx['s_anom_change_r_cont'] - (L_v * np.diff(sphum_ref, axis=0).squeeze() +
                                          (mu[0]*beta_s1[0] + np.diff(beta_s1, axis=0).squeeze()) * temp_surf_anom[0]
                                          ) * np.diff(r_anom/r_ref[:, np.newaxis], axis=0).squeeze()
        # Add r_change2 term
        approx_terms['r_change'] -= L_v * sphum_ref[0] * (
                np.diff(r_anom / r_ref[:, np.newaxis], axis=0).squeeze() + r_anom[0] *
                (np.diff(r_ref, axis=0).squeeze() / r_ref[0] ** 2)
                - np.diff(r_anom, axis=0).squeeze() / r_ref[0])

    approx_terms['temp_s_anom_r_change'] = -approx['s_anom_change_temp_r_cont'] - (
            mu[0]*beta_s1[0] + np.diff(beta_s1, axis=0).squeeze()
    ) * np.diff(r_anom/r_ref[:, np.newaxis], axis=0).squeeze() * np.diff(temp_surf_anom, axis=0).squeeze()
    # Convert into scaling factor K/K units
    for key in approx_terms:
        approx_terms[key] = approx_terms[key] / (beta_s1[0] * np.diff(temp_surf_ref, axis=0).squeeze())
    return approx_terms, approx


def decompose_var_x_change(var_x: np.ndarray, var_p: np.ndarray,
                           quant_p: np.ndarray = np.arange(100, dtype=int), simple: bool = True
                           ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    We can decompose the change in variable $\chi$, conditioned on near-surface temperature
    percentile $x$ into the change in the corresponding percentile of $\chi$: $p_x$, but accounting
    for how $p_x$ changes with warming:

    $\delta \chi[x] = \delta \chi(p_x) + [\chi(p_x+\delta p_x) - \chi(p_x)]
    + [\delta \chi(p_x+\delta p_x) - \delta \chi(p_x)]$

    where:

    * $p_x$ is defined such that $\chi[x] = \chi(p_x)$.
    * $\delta \chi(p_x) = \chi^{hot}(p^{cold}_x) - \chi^{cold}(p^{cold}_x)$ i.e. keep $p_x$ constant, at its value
        in the colder simulation.
    * $\delta \chi(p_x)$ is the contribution due to change in the distribution of $\chi$ with warming, neglecting
        change in percentile.
    * $\chi(p_x+\delta p_x) - \chi(p_x)$ is the contribution due to change in percentile, neglecting change in
        distribution of $\chi$.
    * $\delta \chi(p_x+\delta p_x) - \delta \chi(p_x)$ is the non-linear contribution, influenced by both changes
        in the distribution and percentile of $\chi$.

    Keeping only the first two linear terms on the RHS, provides a good approximation.
    This is achieved by setting `simple=True`.

    Args:
        var_x: `float [n_exp, n_quant_x]`</br>
            Variable $\chi$ conditioned on near-surface temperature percentile, $x$, for each
            experiment: $\chi[x]$. $x$ can differ from `quant_px`, but likely to be the same: `np.arange(1, 100)`.
        var_p: `float [n_exp, n_quant_p]`</br>
            `var_p[i, j]` is the $p=$`quant_p[j]`$^{th}$ percentile of variable $\chi$ for
            experiment `i`: $\chi(p)$.
        quant_p: `float [n_quant_p]`</br>
            Corresponding quantiles to `var_p`.
        simple: If `True`, `var_x_change_theory` will be
            $\delta \chi(p_x) + [\chi(p_x+\delta p_x) - \chi(p_x)]$. If `False`, will also include
            $\delta \chi(p_x+\delta p_x) - \delta \chi(p_x)$, and will exactly match `var_x_change`.

    Returns:
        var_x_change: `float [n_quant_x]`</br>
            Simulated $\delta \chi[x]$
        var_x_change_theory: `float [n_quant_x]`</br>
            Theoretical $\delta \chi[x]$
        var_x_change_cont: Dictionary recording the five terms in the theory for $\delta \chi[x]$.
            The sum of all these terms should match the simulated `var_x_change`.

            * `dist`: $\delta \chi(p_x)$
            * `p_x`: $\chi(p_x+\delta p_x) - \chi(p_x)$
            * `nl`: $\delta \chi(p_x+\delta p_x) - \delta \chi(p_x)$

    """
    n_exp, n_quant_x = var_x.shape
    # Get FT percentile corresponding to each FT temperature conditioned on near-surface percentile
    p_x = np.zeros((n_exp, n_quant_x))
    for i in range(n_exp):
        p_x[i] = get_p_x(var_x[i], var_p[i], quant_p)[0]

    # Interpolation so can use p_x which are not integers
    var_p_interp_func = [scipy.interpolate.interp1d(quant_p, var_p[i], fill_value='extrapolate') for i in range(n_exp)]
    # Sanity check that interpolation function works
    for i in range(n_exp):
        if not np.allclose(var_p_interp_func[i](p_x[i]), var_x[i]):
            raise ValueError(f'Error in interpolation for experiment {i}')


    # Isolate x dependence into different terms
    # Use var_x sometimes and var_p sometimes, so sum of these contributions exactly equal var_x_change
    # I.e. exploit that var_x[0] = var_p_interp_func[0](p_x[0]) and var_x[1] = var_p_interp_func[1](p_x[1])
    var_x_change = var_x[1] - var_x[0]
    var_x_change_cont = {'dist': var_p_interp_func[1](p_x[0]) - var_x[0],
                         'p_x': var_p_interp_func[0](p_x[1]) - var_x[0],
                         'nl': var_x[1] - var_p_interp_func[0](p_x[1]) -
                               (var_p_interp_func[1](p_x[0]) - var_x[0])}

    # Theory is sum of terms
    var_x_change_theory = var_x_change_cont['dist'] + var_x_change_cont['p_x']
    if not simple:
        var_x_change_theory = var_x_change_theory + \
                                var_x_change_cont['nl']

    return var_x_change, var_x_change_theory, var_x_change_cont


def decompose_var_x_change_integrate(var_av: np.ndarray, var_x: np.ndarray, var_p: np.ndarray,
                                     quant_p: np.ndarray = np.arange(100, dtype=int), simple: bool = True
                                     ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    We can decompose the change in variable $\chi$, conditioned on near-surface temperature
    percentile $x$ into the change in the corresponding percentile of $\chi$: $p_x$, but accounting
    for how $p_x$ changes with warming:

    $\delta \chi[x] \\approx \delta \chi(p_x) + \overline{\eta}\delta p_x +
    \Delta \eta(p_x)\delta p_x + \delta \eta(p_x) \delta p_x$

    where:

    * $p_x$ is defined such that $\chi[x] = \chi(p_x)$ and $\overline{p}$ such that
    $\overline{\chi} = \chi[\overline{p}]$.
    * $\eta(p_x) = \\frac{\\partial \chi}{\\partial p}\\bigg|_{p_x}$;
        $\overline{\eta} = \\frac{\\partial \chi}{\\partial p}\\bigg|_{\overline{p}}$ and
        $\Delta \eta(p_x) = \eta(p_x) - \overline{\eta}$.
    * $\delta \chi(p_x) = \chi^{hot}(p^{cold}_x) - \chi^{cold}(p^{cold}_x)$ i.e. keep $p_x$ constant, at its value
        in the colder simulation.

    The only approximation in the above is saying that $\eta(p) + \delta \eta(p)$ is constant between
    $p=p_x$ and $p=p_x+\delta p_x$.
    Keeping only the first two terms on the RHS, also provides a good approximation.
    This is achieved by setting `simple=True`.

    Args:
        var_av: `float [n_exp]`</br>
            Average of variable $\chi$ for each experiment, ikely to be mean or median.
        var_x: `float [n_exp, n_quant_x]`</br>
            Variable $\chi$ conditioned on near-surface temperature percentile, $x$, for each
            experiment: $\chi[x]$. $x$ can differ from `quant_px`, but likely to be the same: `np.arange(1, 100)`.
        var_p: `float [n_exp, n_quant_p]`</br>
            `var_p[i, j]` is the $p=$`quant_p[j]`$^{th}$ percentile of variable $\chi$ for
            experiment `i`: $\chi(p)$.
        quant_p: `float [n_quant_p]`</br>
            Corresponding quantiles to `var_p`.
        simple: If `True`, `temp_ft_change_theory` will be
            $\delta \chi(p_x) + \overline{\eta}\delta p_x$. If `False`, will also include
            $\Delta \eta(p_x) \delta p_x + \delta \eta(p_x) \delta p_x$.

    Returns:
        var_x_change: `float [n_quant_x]`</br>
            Simulated $\delta \chi[x]$
        var_x_change_theory: `float [n_quant_x]`</br>
            Theoretical $\delta \chi[x]$
        var_x_change_cont: Dictionary recording the five terms in the theory for $\delta \Delta \chi(x)$.
            The sum of all these terms should match the simulated `var_x_change`.

            * `var_p`: $\delta \chi(p_x)$
            * `p_x`: $\overline{\eta}\delta p_x$
            * `nl_eta0`: $\Delta \eta(p_x) \delta p_x$
            * `nl_change`: $\delta \eta(p_x) \delta p_x$
            * `approx_integral`: Accounts for approximation made during the integral:
                $\int_{p_x}^{p_x + \delta p_x} \eta(p) + \delta \eta(p) dp -
                (\eta(p_x) + \delta \eta(p_x))\delta p_x$

    """
    n_exp, n_quant_x = var_x.shape

    # Get FT percentile corresponding to each FT temperature conditioned on near-surface percentile
    p_av_ind = np.zeros(n_exp, dtype=int)
    p_x = np.zeros((n_exp, n_quant_x))
    p_x_ind = np.zeros((n_exp, n_quant_x), dtype=int)
    for i in range(n_exp):
        p_av_ind[i] = get_p_x(var_av[i], var_p[i], quant_p)[1]
        p_x[i], p_x_ind[i] = get_p_x(var_x[i], var_p[i], quant_p)

    # Get eta on corresponding to p_x in the reference (coldest) simulation
    eta_av0 = np.zeros(n_exp)
    eta_px0 = np.zeros((n_exp, n_quant_x))
    for i in range(n_exp):
        eta_use = np.gradient(var_p[i], quant_p)
        eta_av0[i] = eta_use[p_av_ind[0]]
        for j in range(n_quant_x):
            eta_px0[i, j] = eta_use[p_x_ind[0, j]]

    # Isolate x dependence into different terms
    eta_px0_anom = eta_px0 - eta_av0[:, np.newaxis]
    var_x_change = var_x[1] - var_x[0]
    var_x_change_cont = {'var_p': var_p[1, p_x_ind[0]] - var_p[0, p_x_ind[0]],
                         'p_x': eta_av0[0] * (p_x[1] - p_x[0]),
                         'nl_eta0': eta_px0_anom[0] * (p_x[1] - p_x[0]),
                         'nl_change': (eta_px0[1] - eta_px0[0]) * (p_x[1] - p_x[0])}

    # This residual term is due to error in integral approximation where assume integrand constant
    var_x_change_cont['approx_integral'] = var_x_change - sum(var_x_change_cont.values())

    # Theory is sum of terms
    var_x_change_theory = var_x_change_cont['var_p'] + var_x_change_cont['p_x']
    if not simple:
        var_x_change_theory = var_x_change_theory + \
                                var_x_change_cont['nl_eta0'] + var_x_change_cont['nl_change']

    return var_x_change, var_x_change_theory, var_x_change_cont