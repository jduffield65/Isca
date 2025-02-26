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
                     z_approx_ref: Optional[np.ndarray] = None):
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
    approx_terms['temp_s_anom_change'] = - np.diff(approx['s_anom'], axis=0).squeeze() - (
            (mu*beta_s1/r_ref)[0]*r_anom[0] + (1+r_anom[0]/r_ref[0])*np.diff(beta_s1, axis=0).squeeze()
    )* np.diff(temp_surf_anom, axis=0).squeeze()
    approx_terms['r_anom_change'] = -(L_v * np.diff(sphum_ref, axis=0).squeeze() +
                                      (mu[0]*beta_s1[0] + np.diff(beta_s1, axis=0).squeeze()) * temp_surf_anom[0]
                                      ) * np.diff(r_anom/r_ref[:, np.newaxis], axis=0).squeeze()
    approx_terms['temp_s_r_anom_change'] = -(mu[0]*beta_s1[0] + np.diff(beta_s1, axis=0).squeeze()
                                             ) * np.diff(r_anom/r_ref[:, np.newaxis], axis=0).squeeze() * \
                                           np.diff(temp_surf_anom, axis=0).squeeze()
    # Convert into scaling factor K/K units
    for key in approx_terms:
        approx_terms[key] = approx_terms[key] / (beta_s1[0] * np.diff(temp_surf_ref, axis=0).squeeze())
    return approx_terms, approx

    #
    # approx_terms['anom1'] = (prefactor_mse_ft * mse_mod_ref_change0) * (approx['s_anom'] - approx['ft_anom'])[0]
    # # Due to beta approx and var_ref_change squared
    # approx_terms['ref_change1'] = prefactor_mse_ft * (
    #         mse_mod_ref_change0 * approx['ft_beta'] + (1 + approx['ft_beta']) * var_ref_change) * (
    #         mse_mod_anom0 + beta_ft1[0] * temp_ft_anom_change_mod - var_ref_change)
    # # Due to just var_ref_change
    # approx_terms['ref_change1'] -= var_ref_change + (prefactor_mse_ft * mse_mod_ref_change0 * var_ref_change)
    #
    # # Combine all z errors into single value as z_ref error usually zero
    # approx_terms['z'] = np.diff(approx['z_anom'], axis=0).squeeze()  # change in z_anom
    # approx_terms['z'] -= prefactor_mse_ft * mse_mod_ref_change0 * approx['z_anom'][0]  # z_anom in current climate
    # # non-linear: anomaly in current climate and change in ref climate
    # approx_terms['z'] += prefactor_mse_ft * np.diff(approx['z_ref'], axis=0).squeeze() * approx['z_anom'][0]
    # # Change in ref climate
    # approx_terms['z'] += np.diff(approx['z_ref'], axis=0).squeeze()[:, np.newaxis] \
    #                      - prefactor_mse_ft * np.diff(approx['z_ref'], axis=0).squeeze() * (
    #                              mse_mod_anom0 + beta_ft1[0] * temp_ft_anom_change_mod +
    #                              np.diff(approx['z_ref'], axis=0).squeeze() - mse_mod_ref_change0)
    #
    # # Due to combinations of approx in z, anom and ref_change
    # approx_terms['nl1'] = prefactor_mse_ft * mse_mod_ref_change0 * approx['ft_beta'] * (
    #         approx['s_anom'] - approx['ft_anom'] + np.diff(approx['z_ref'], axis=0).squeeze() - approx['z_anom'])[0]
    # approx_terms['nl1'] -= prefactor_mse_ft * approx['ft_beta'] * np.diff(approx['z_ref'], axis=0).squeeze() * (
    #         mse_mod_anom0 + beta_ft1 * temp_ft_anom_change_mod - var_ref_change)
    # approx_terms['nl1'] += prefactor_mse_ft * np.diff(approx['z_ref'], axis=0).squeeze() * var_ref_change
    # approx_terms['nl1'] += prefactor_mse_ft * (1 + approx['ft_beta']) * (
    #         approx['s_change'] - approx['ft_change'] - np.diff(approx['z_ref'], axis=0).squeeze()) * (
    #         approx['s_anom'][0] - approx['ft_anom'][0] + np.diff(approx['z_ref'], axis=0).squeeze() -
    #         approx['z_anom'][0])
    # approx_terms['nl1'] += prefactor_mse_ft * np.diff(approx['z_ref'], axis=0).squeeze() * (
    #         np.diff(approx['z_ref'], axis=0).squeeze()[:, np.newaxis] - approx['z_anom'][0])
    #
    # # Due to multiple