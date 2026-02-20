import numpy as np
import xarray as xr
from typing import Union, Tuple, Optional
from ..utils.moist_physics import get_density, sphum_sat, clausius_clapeyron_factor
from ..utils.constants import L_v


def get_latent_heat(
        temp_surf: Union[float, np.ndarray, xr.DataArray],
        temp_diseqb: Union[float, np.ndarray, xr.DataArray],
        rh_atm: Union[float, np.ndarray, xr.DataArray],
        w_atm: Union[float, np.ndarray, xr.DataArray],
        drag_coef: Union[float, np.ndarray, xr.DataArray],
        p_surf: Union[float, np.ndarray, xr.DataArray],
        p_atm: Union[float, np.ndarray, xr.DataArray],
        evap_prefactor: float = 1,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Compute the surface latent heat flux using a bulk aerodynamic formula.
    This function uses the bulk exchange estimate:

    $$
    LH = \\beta\\, L_v\\, C_E\\, \\rho\\, U\\, \\left(q_s^* - q_a\\right).
    $$

    Here, the near-surface atmospheric state is diagnosed as: $T_a = T_s - T_{dq}$
    and the near-surface atmospheric specific humidity is computed from relative
    humidity, $r_a$: $q_a = r_a q^*(T_a, p_a)$, while the surface saturation specific humidity is $q_s^* = q^*(T_s, p_s)$.

    Args:
        temp_surf: Surface temperature, $T_s$ (K)
        temp_diseqb: Surface–air temperature disequilibrium, $T_{dq}$ (K), used
            in $T_a = T_s - T_{diseqb}$
        rh_atm: Near-surface relative humidity, $r_a$ (unitless, 0–1)
        w_atm: Near-surface wind speed, $U$ (m s$^{-1}$)
        drag_coef: Bulk exchange coefficient, $C_E$ (unitless)
        p_surf: Surface pressure, $p_s$ (Pa)
        p_atm: Near-surface atmospheric pressure, $p_a$ (Pa)
        evap_prefactor: Evaporation prefactor, $\\beta$ (unitless)

    Returns:
        flux_lh: Latent heat flux, $LH$ (W m$^{-2}$)

    """
    temp_atm = temp_surf - temp_diseqb
    rho_atm = get_density(temp_atm, p_atm)
    q_atm = rh_atm * sphum_sat(temp_atm, p_atm)
    q_surf = sphum_sat(temp_surf, p_surf)  # sat specific humidity at surface
    return evap_prefactor * L_v * drag_coef * rho_atm * w_atm * (q_surf - q_atm)


def get_sensitivity_lh(temp_surf: Union[float, np.ndarray, xr.DataArray],
                       temp_diseqb: Union[float, np.ndarray, xr.DataArray],
                       rh_atm: Union[float, np.ndarray, xr.DataArray],
                       w_atm: Union[float, np.ndarray, xr.DataArray],
                       drag_coef: Union[float, np.ndarray, xr.DataArray],
                       p_surf: Union[float, np.ndarray, xr.DataArray],
                       sigma_atm: float,
                       evap_prefactor: float = 1) -> dict:
    p_atm = p_surf * sigma_atm
    temp_atm = temp_surf - temp_diseqb
    alpha_surf = clausius_clapeyron_factor(temp_surf, p_surf)
    alpha_atm = clausius_clapeyron_factor(temp_atm, p_atm)
    q_atm_sat = sphum_sat(temp_atm, p_atm)
    q_atm = rh_atm * q_atm_sat
    rho_atm = get_density(temp_atm, p_atm)
    lh = get_latent_heat(temp_surf, temp_diseqb, rh_atm, w_atm, drag_coef, p_surf, p_atm, evap_prefactor)
    lh_prefactor = evap_prefactor * L_v * drag_coef * w_atm * rho_atm

    # Differential of lh wrt each param
    out_dict = {'evap_prefactor': lh / evap_prefactor,
                'drag_coef': lh / drag_coef,
                'w_atm': lh / w_atm,
                'p_surf': 0,
                'rh_atm': -lh_prefactor * q_atm_sat,
                'temp_diseqb': lh / temp_atm + lh_prefactor * q_atm * alpha_atm,
                'temp_surf': (alpha_surf - 1 / temp_atm) * lh + lh_prefactor * q_atm * (alpha_surf - alpha_atm),
                }

    # Nonlinear contributions
    out_dict['nl_temp_surf_square'] = \
        (alpha_surf - 1 / temp_atm) * out_dict['temp_surf'] + \
        lh * (1 / temp_atm ** 2 - 2 * alpha_surf / temp_surf) + \
        lh_prefactor * q_atm * (2 * alpha_atm / temp_atm -
                                2 * alpha_surf / temp_surf + alpha_atm * (alpha_surf - alpha_atm))
    out_dict['nl_temp_surf_square'] = out_dict['nl_temp_surf_square'] * 0.5     # to match the taylor series coef

    # Mechanism combinations
    for key in ['evap_prefactor', 'drag_coef', 'w_atm', 'p_surf']:
        out_dict[f'nl_temp_surf_{key}'] = out_dict[key] * out_dict['temp_surf'] / lh
    out_dict['nl_temp_surf_rh_atm'] = -lh_prefactor*alpha_atm*q_atm_sat
    out_dict['nl_temp_surf_temp_diseqb'] = -lh/temp_atm**2 + out_dict['temp_surf']/temp_atm + \
        lh_prefactor * q_atm * alpha_atm * (alpha_atm-2/temp_atm)
    return out_dict
