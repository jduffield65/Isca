import numpy as np
import xarray as xr
import inspect
from typing import Union, Tuple

from jobs.thesis_season.thesis_figs.utils import day_seconds
from .surface_flux_taylor_2layer import get_sensitivity_sh, get_sensitivity_lh, \
    get_sensitivity_lw_surf, get_sensitivity_lw_atm
from isca_tools.utils.constants import c_p, g, L_v
from ..utils.moist_physics import clausius_clapeyron_factor, sphum_sat


def get_feedback_params(temp_surf: Union[float, np.ndarray, xr.DataArray],
                        temp_atm: Union[float, np.ndarray, xr.DataArray],
                        temp_rad_surf: Union[float, np.ndarray, xr.DataArray],
                        temp_rad_atm: Union[float, np.ndarray, xr.DataArray],
                        rh_atm: Union[float, np.ndarray, xr.DataArray],
                        w_atm: Union[float, np.ndarray, xr.DataArray],
                        drag_coef: Union[float, np.ndarray, xr.DataArray],
                        p_surf: Union[float, np.ndarray, xr.DataArray],
                        odp_surf: Union[float, np.ndarray, xr.DataArray],
                        sigma_atm: float,
                        evap_prefactor: float = 1,
                        temp_rad_surf_coef_amp: Union[float, np.ndarray, xr.DataArray] = 0,
                        temp_rad_atm_coef_amp: Union[float, np.ndarray, xr.DataArray] = 0
                        ) -> Tuple[Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray], Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray]]:
    local_vars = locals()
    get_sensitivity = {'lh': get_sensitivity_lh, 'sh': get_sensitivity_sh, 'lw_surf': get_sensitivity_lw_surf,
                       'lw_atm': get_sensitivity_lw_atm}
    gamma = {}
    for key in get_sensitivity:
        arg_names = list(inspect.signature(get_sensitivity[key]).parameters.keys())
        args_use = {name: local_vars[name] for name in arg_names if name in local_vars}
        gamma[key] = get_sensitivity[key](**args_use)

    # Construct two layer feedback parameters from individual flux sensitivity factors
    lambda_s1 = gamma['sh']['temp_surf'] + gamma['lh']['temp_surf'] + gamma['lw_surf']['temp_surf']
    lambda_s2 = gamma['sh']['temp_surf'] + gamma['lh']['temp_surf'] + gamma['lw_atm']['temp_surf']
    lambda_a1 = -(gamma['sh']['temp_atm'] + gamma['lh']['temp_atm'] +
                  gamma['lw_surf']['temp_rad_surf'] * (1-temp_rad_surf_coef_amp))
    lambda_a2 = -gamma['lw_atm']['temp_rad_atm'] * (1-temp_rad_atm_coef_amp)
    return lambda_s1, lambda_s2, lambda_a1, lambda_a2


def get_heat_cap_lambda_eff(temp_surf: Union[float, np.ndarray, xr.DataArray],
                            temp_atm: Union[float, np.ndarray, xr.DataArray],
                            temp_rad_surf: Union[float, np.ndarray, xr.DataArray],
                            temp_rad_atm: Union[float, np.ndarray, xr.DataArray],
                            rh_atm: Union[float, np.ndarray, xr.DataArray],
                            w_atm: Union[float, np.ndarray, xr.DataArray],
                            drag_coef: Union[float, np.ndarray, xr.DataArray],
                            p_surf: Union[float, np.ndarray, xr.DataArray],
                            odp_surf: Union[float, np.ndarray, xr.DataArray],
                            sigma_atm: float,
                            heat_cap_surf: Union[float, np.ndarray, xr.DataArray],
                            temp_col_sphum: Union[float, np.ndarray, xr.DataArray],
                            p_col_sphum: float,
                            rh_col: Union[float, np.ndarray, xr.DataArray],
                            pressure_heat_cap_atmos_calc: float,
                            evap_prefactor: float = 1,
                            temp_rad_surf_coef_amp: Union[float, np.ndarray, xr.DataArray] = 0,
                            temp_rad_atm_coef_amp: Union[float, np.ndarray, xr.DataArray] = 0,
                            temp_rad_atm_coef_phase: Union[float, np.ndarray, xr.DataArray] = 0,
                            temp_col_coef_amp: Union[float, np.ndarray, xr.DataArray] = 0,
                            temp_col_coef_phase: Union[float, np.ndarray, xr.DataArray] = 0,
                            temp_col_sphum_coef_amp: Union[float, np.ndarray, xr.DataArray] = 0,
                            n_year_days: int = 360,
                            day_seconds: int = 86400,
                            assume_small_temp_col_coef_phase: bool = True,
                            assume_small_temp_rad_atm_phase: bool = True,
                            assume_small_heat_cap_atmos: bool = False
                            ) -> Tuple[Union[float, np.ndarray, xr.DataArray],
Union[float, np.ndarray, xr.DataArray], Union[float, np.ndarray, xr.DataArray]]:
    local_vars = locals()
    arg_names = list(inspect.signature(get_feedback_params).parameters.keys())
    args_use = {name: local_vars[name] for name in arg_names if name in local_vars}
    lambda_s1, lambda_s2, lambda_a1, lambda_a2 = get_feedback_params(**args_use)

    heat_cap_atmos = c_p * pressure_heat_cap_atmos_calc / g
    alpha_col_sphum = clausius_clapeyron_factor(temp_col_sphum, p_col_sphum)
    q_sat_col_sphum = sphum_sat(temp_col_sphum, p_col_sphum)
    mu = L_v/c_p * alpha_col_sphum * rh_col * q_sat_col_sphum * (1 - temp_col_sphum_coef_amp)
    f = 1/(n_year_days*day_seconds)
    omega = 2 * np.pi * f

    if assume_small_temp_col_coef_phase:
        # Taylor expansion of cos and sin to 1st order
        temp_col_amp_param = 1 - temp_col_coef_amp
        temp_col_phase_param = (1 - temp_col_coef_amp) * temp_col_coef_phase
    else:
        temp_col_amp_param = (1 - temp_col_coef_amp) * np.cos(temp_col_coef_phase)
        temp_col_phase_param = (1 - temp_col_coef_amp) * np.sin(temp_col_coef_phase)

    lambda_a2_mod = lambda_a1-omega*temp_col_phase_param*heat_cap_atmos
    heat_cap_atmos_mod = heat_cap_atmos*(temp_col_amp_param+mu)

    # Add lambda2 contribution, where phase can be important
    if assume_small_temp_rad_atm_phase:
        lambda_a2_mod += lambda_a2
        heat_cap_atmos_mod += lambda_a2 * temp_rad_atm_coef_phase/omega
    else:
        lambda_a2_mod += lambda_a2*np.cos(temp_rad_atm_coef_phase)
        heat_cap_atmos_mod += lambda_a2 * np.sin(temp_rad_atm_coef_phase) / omega

    if assume_small_heat_cap_atmos:
        lambda_s1_scaling = 1-lambda_s2*lambda_a1/lambda_s1/lambda_a2_mod
        heat_cap_scaling = 1 + lambda_a1*lambda_s2/lambda_a2_mod**2*heat_cap_atmos_mod/heat_cap_surf
    else:
        lambda_s1_scaling = 1 - lambda_s2*lambda_a2_mod/(lambda_a2_mod**2+omega**2*heat_cap_atmos_mod**2
                                                         )*lambda_a1/lambda_s1
        heat_cap_scaling = 1 + lambda_a1*lambda_s2/(lambda_a2_mod**2+omega**2*heat_cap_atmos_mod**2
                                                         )*heat_cap_atmos_mod/heat_cap_surf
    return lambda_s1, lambda_s1_scaling, heat_cap_scaling
