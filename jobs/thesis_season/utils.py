import xarray as xr
import numpy as np
import os
from typing import List, Optional, Union

from isca_tools.utils.base import mass_weighted_vertical_integral
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.radiation import get_heat_capacity, opd_lw_gray
from isca_tools import load_dataset, load_namelist
from isca_tools.utils.constants import c_p_ocean, rho_ocean
from jobs.thesis_season.column.utils import get_fit_coef_complex_xr, lat_min, lat_max

var_keep = ['temp', 'ps', 'sphum', 'olr', 'swdn_toa', 'swdn_sfc', 'lwdn_sfc', 'lwup_sfc', 'flux_t',
            'flux_lhe', 't_surf']  # just the fluxes, no variables


def load_ds(exp_name: str, exp_dir: str, var_keep: List = var_keep,
            lat_min: float = lat_min, lat_max: float = lat_max,
            first_month_file: Optional[int] = None) -> xr.Dataset:
    # Load all info required for empirical approximation of 2 layer energy budget
    # Will need to update later to add that required analytically such as
    # temp_col_sphum, temp_rad_surf, temp_rad_atm
    exp_path = os.path.join(exp_dir, exp_name)

    ds = load_dataset(exp_path, first_month_file=first_month_file).sel(lat=slice(lat_min, lat_max))[var_keep]
    ds = ds.load()

    # Load info from namelist and add to attributes
    namelist = load_namelist(exp_path)
    sigma_levels_half = np.asarray(namelist['vert_coordinate_nml']['bk'])
    sigma_levels_full = np.convolve(sigma_levels_half, np.ones(2) / 2, 'valid')
    ds['lev_sigma'] = (ds.pfull*0+sigma_levels_full).squeeze()
    ds.attrs['albedo'] = namelist['mixed_layer_nml']['albedo_value']
    ds.attrs['depth'] = namelist['mixed_layer_nml']['depth']
    ds.attrs['heat_cap_surf'] = get_heat_capacity(c_p_ocean, rho_ocean, ds.attrs['depth'])

    # Get longwave optical depth at surface - is a function of latitude
    odp_info = {'ir_tau_eq': 6, 'ir_tau_pole': 1.5, 'linear_tau': 0.1, 'wv_exponent': 4,
                'odp': 1, 'atm_abs': 0}  # default vals
    for key in odp_info:  # If provided, update
        if key in namelist['two_stream_gray_rad_nml']:
            odp_info[key] = namelist['two_stream_gray_rad_nml'][key]
    ds.attrs['odp'] = odp_info['odp']
    ds.attrs['atm_abs'] = odp_info['atm_abs']
    ds['odp_surf'] = opd_lw_gray(ds.lat, kappa=ds.odp, tau_eq=odp_info['ir_tau_eq'],
                                 tau_pole=odp_info['ir_tau_pole'], frac_linear=odp_info['linear_tau'],
                                 k_exponent=odp_info['wv_exponent'])  # optical depth as function of latitude

    # Get column quantities - very important to use simpson integral method
    p_lev = ds.ps * ds.lev_sigma
    ds['temp_col'] = mass_weighted_vertical_integral(ds.temp, p_lev, 'pfull', simpson_method=True)
    ds['sphum_col'] = mass_weighted_vertical_integral(ds.sphum, p_lev, 'pfull', simpson_method=True)
    ds['rh_col'] = ds['sphum_col'] / mass_weighted_vertical_integral(sphum_sat(ds.temp, p_lev),
                                                                     p_lev, 'pfull', simpson_method=True)

    # Only keep the lowest model level
    ds = ds.sel(pfull = np.inf, method='nearest')
    ds['p_integ_calc'] = ds.p_surf * (sigma_levels_full[-1] - sigma_levels_full[0])  # keep track of p range for integration
    
    return ds


