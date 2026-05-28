import numpy as np
import xarray as xr
import inspect
from typing import Union
from .surface_flux_taylor_2layer import get_sensitivity_sh, get_sensitivity_lh, \
    get_sensitivity_lw_surf, get_sensitivity_lw_atm


def get_feedback_params(temp_surf: Union[float, np.ndarray, xr.DataArray],
                        temp_atm: Union[float, np.ndarray, xr.DataArray],
                        temp_diseqb_surf: Union[float, np.ndarray, xr.DataArray],
                        temp_diseqb_atm: Union[float, np.ndarray, xr.DataArray],
                        rh_atm: Union[float, np.ndarray, xr.DataArray],
                        w_atm: Union[float, np.ndarray, xr.DataArray],
                        drag_coef: Union[float, np.ndarray, xr.DataArray],
                        p_surf: Union[float, np.ndarray, xr.DataArray],
                        odp_surf: Union[float, np.ndarray, xr.DataArray],
                        sigma_atm: float,
                        evap_prefactor: float = 1,
                        temp_diseqb_surf_coef: Union[float, np.ndarray, xr.DataArray] = 0,
                        temp_diseqb_atm_coef: Union[float, np.ndarray, xr.DataArray] = 0) -> dict:
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
    lambda_a1 = -(gamma['sh']['temp_atm'] + gamma['lh']['temp_atm'] + gamma['lw_surf']['temp_atm'] +
                  gamma['lw_surf']['temp_diseqb_surf']*temp_diseqb_surf_coef)
    lambda_a2 = -(gamma['sh']['temp_atm'] + gamma['lh']['temp_atm'] + gamma['lw_atm']['temp_atm'] +
                  gamma['lw_atm']['temp_diseqb_atm']*temp_diseqb_atm_coef +
                  gamma['lw_atm']['temp_diseqb_surf']*temp_diseqb_surf_coef)
    return lambda_s1, lambda_s2, lambda_a1, lambda_a2
