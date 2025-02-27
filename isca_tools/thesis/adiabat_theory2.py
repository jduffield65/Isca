import numpy as np
import scipy.optimize
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat, moist_static_energy
from ..utils.constants import c_p, L_v, R, g
from .adiabat_theory import get_theory_prefactor_terms, get_temp_adiabat
from typing import Tuple, Union, Optional

def get_approx_terms(temp_surf_ref: np.ndarray, temp_surf_quant: np.ndarray, r_ref: np.ndarray,
                     r_quant: np.ndarray, temp_ft_quant: np.ndarray,
                     epsilon_ref: np.ndarray, epsilon_quant: np.ndarray,
                     pressure_surf: float, pressure_ft: float,
                     z_approx_ref: Optional[np.ndarray] = None) -> Tuple[dict, dict]:
    """
    Function which returns terms quantifying the errors associated with various approximations that go
    into the derivation of the theoretical scaling factor, $\delta T_s(x)/\delta \\tilde{T}_s$,
    given by `get_scaling_factor_theory`.

    For more details on the approximations, there is a
    [Jupyter notebook](https://github.com/jduffield65/Isca/blob/main/jobs/tau_sweep/land/meridional_band/publish_figures/theory_approximations2.ipynb)
    that goes through each step of the derivation.

    Args:
        temp_surf_ref: `float [n_exp]`</br>
            Reference near surface temperature of each simulation, corresponding to a different
            optical depth, $\kappa$. Units: *K*. We assume `n_exp=2`.
        temp_surf_quant: `float [n_exp, n_quant]`</br>
            `temp_surf_quant[i, j]` is the percentile `quant_use[j]` of near surface temperature of
            experiment `i`. Units: *K*.</br>
            Note that `quant_use` is not provided as not needed by this function, but is likely to be
            `np.arange(1, 100)` - leave out `x=0` as doesn't really make sense to consider $0^{th}$ percentile
            of a quantity.
        r_ref: `float [n_exp]`</br>
            Reference near surface relative humidity of each simulation. Units: dimensionless (from 0 to 1).
        r_quant: `float [n_exp, n_quant]`</br>
            `r_quant[i, j]` is near-surface relative humidity, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: dimensionless.
        temp_ft_quant: `float [n_exp, n_quant]`</br>
            `temp_ft_quant[i, j]` is temperature at `pressure_ft`, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kg/kg*.
        epsilon_ref: `float [n_exp]`</br>
            Reference value of $\epsilon = h_s - h^*_{FT}$, where $h_s$ is near-surface MSE and
            $h^*_{FT}$ is saturated MSE at `pressure_ft`. Units: *kJ/kg*.
        epsilon_quant: `float [n_exp, n_quant]`</br>
            `epsilon_quant[i, j]` is $\epsilon = h_s - h^*_{FT}$, averaged over all days with near-surface temperature
             corresponding to the quantile `quant_use[j]`, for experiment `i`. Units: *kJ/kg*.
        pressure_surf:
            Pressure at near-surface in *Pa*.
        pressure_ft:
            Pressure at free troposphere level in *Pa*.
        z_approx_ref: `float [n_exp]`</br>
            The exact equation for modified MSE is given by: $h^{\dagger} = (c_p - R^{\dagger})T_s + L_v q_s
            - \epsilon = (c_p + R^{\dagger})T_{FT} + L_vq^*(T_{FT}, p_{FT}) + A_z$
            where $R^{\dagger} = R\\ln(p_s/p_{FT})/2$ and $A_z$ quantifies the error due to
            approximation of geopotential height, as relating to temperature.</br>
            Here you have the option of specifying the reference $A_z$ for each simulation. If not provided,
            will set to 0. Units: *kJ/kg*.

    Returns:
        approx_terms: Dictionary containing approximations associated with final scaling factor,
            $\delta T_s(x)/\delta \\tilde{T}_s$ so units are *K/K*. Terms have been named based on what causes the
            variation in $x$. Each value in dictionary is a `float [n_quant]` array:

            * `temp_ft_anom_change`: Involves contribution from $\delta \Delta T_{FT}[x]$.
            * `temp_s_anom_change`: Involves contribution from $\delta \Delta T_s(x)$.
            * `r_anom_change`: Involves contribution from $\delta \Delta r_s[x]$.
            * `temp_s_r_anom_change`: Involves contribution from $\delta \Delta T_s(x) \Delta r_s[x]$.
            * `anom_temp_s_r`: Involves contribution from $\Delta T_s(x) \Delta r_s[x]$.
            * `anom`: Groups together erros due to approximation of anomaly in current climate. Excludes
                those in `anom_temp_s_r`.
            * `ref_change`: Groups together erros due to change with warming of the reference day quantities.
            * `z_anom_change`: $\delta \Delta A_z[x]$ where $A_z$ quantifies the error due to
                approximation of geopotential height.
            * `nl`: Residual non-linear errors that don't directly correspond to any of the above.
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
                where $\mu=\\frac{L_v \\alpha_s q_s}{\\beta_{s1}}$; $\\beta_{s1} = \\frac{\partial h^{\dagger}}{\partial T_s}$ and
                $\\beta_{s2} = T_s \\frac{\partial^2 h^{\dagger}}{\partial T_s^2}$.
            * `s_ref_change`: `float`</br> Approximation associated with change of $h^{\dagger}$ with warming at surface:
                $\\tilde{A}_{s\delta}[x] = (1 + \\frac{\delta \\tilde{r}_s[x]}{\\tilde{r}_s})\sum_{n=2}^{\infty}\\frac{1}{n!}\\frac{\partial^n h^{\dagger}}{\partial T_{s}^n}(\delta \\tilde{T}_{s})^n$

    """
    n_exp, n_quant = temp_surf_quant.shape
    if z_approx_ref is None:
        z_approx_ref = np.zeros(n_exp)

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
    approx['s_ref_change'] = np.diff(mse_mod_ref, axis=0).squeeze() - beta_s1[0] * (
            1 + mu[0] * (np.diff(r_ref, axis=0).squeeze() / r_ref[0])
    ) * np.diff(temp_surf_ref, axis=0).squeeze() - L_v * sphum_ref[0] * (np.diff(r_ref, axis=0).squeeze() / r_ref[0]) \
                             + np.diff(epsilon_ref, axis=0).squeeze()

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
    approx_terms['anom_temp_s_r'] = prefactor_mse_ft * mse_mod_ref_change0 * beta_s1[0] * (
        mu[0] * (r_anom[0]/r_ref[0])) * temp_surf_anom[0]

    # From Surface derivation
    approx_terms['anom'] -= (approx['s_beta'] * beta_s2[0] * (1 + np.diff(r_ref, axis=0).squeeze()/r_ref[0]) *
                                               np.diff(temp_surf_ref, axis=0).squeeze()/temp_surf_ref[0]
                                               ) * temp_surf_anom[0] + (approx['s_ref_change']/r_ref[0]) * r_anom[0]
    approx_terms['anom_temp_s_r'] -= np.diff(beta_s1, axis=0).squeeze() * \
                                   r_anom[0]/r_ref[0] * temp_surf_anom[0]
    approx_terms['temp_s_anom_change'] = - approx['s_anom_change_temp_cont'] - (
            (mu*beta_s1/r_ref)[0]*r_anom[0] + (1+r_anom[0]/r_ref[0])*np.diff(beta_s1, axis=0).squeeze()
    )* np.diff(temp_surf_anom, axis=0).squeeze()
    approx_terms['r_anom_change'] = -approx['s_anom_change_r_cont'] - (L_v * np.diff(sphum_ref, axis=0).squeeze() +
                                      (mu[0]*beta_s1[0] + np.diff(beta_s1, axis=0).squeeze()) * temp_surf_anom[0]
                                      ) * np.diff(r_anom/r_ref[:, np.newaxis], axis=0).squeeze()
    approx_terms['temp_s_r_anom_change'] = -approx['s_anom_change_temp_r_cont'] - (
            mu[0]*beta_s1[0] + np.diff(beta_s1, axis=0).squeeze()
    ) * np.diff(r_anom/r_ref[:, np.newaxis], axis=0).squeeze() * np.diff(temp_surf_anom, axis=0).squeeze()
    # Convert into scaling factor K/K units
    for key in approx_terms:
        approx_terms[key] = approx_terms[key] / (beta_s1[0] * np.diff(temp_surf_ref, axis=0).squeeze())
    return approx_terms, approx
