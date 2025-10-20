import numpy as np
import os
import xarray as xr
import re
import logging
import sys
from typing import Optional
from isca_tools import cesm
from isca_tools.convection import potential_temp, dry_profile_temp, lcl_metpy
from isca_tools.thesis.lapse_theory import get_var_at_plev, get_ds_in_pressure_range
from isca_tools.thesis.profile_fitting import get_mse_env, get_lnb_lev_ind, get_mse_prof_rms
from isca_tools.utils.constants import g
from isca_tools.utils.base import weighted_RMS, dp_from_pressure, print_log
from isca_tools.utils.xarray import convert_ds_dtypes
from isca_tools.utils.moist_physics import moist_static_energy, sphum_sat

def get_co2_multiplier(name):
    match = re.match(r'co2_([\d_]+)x', name)
    if match:
        # Replace underscore with decimal point and convert to float
        return float(match.group(1).replace('_', '.'))
    elif name == 'pre_industrial':
        return 1  # for pre_industrial or other defaults
    else:
        raise ValueError(f'Not valid name = {name}')

small_ds = False
save_processed = True
ds_path = '/Users/joshduffield/Desktop/ds_lcl.nc'
load_processed = os.path.exists(ds_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()  # for printing to console time info

# Load in Data
if load_processed:
    ds_lcl = xr.load_dataset(ds_path)
    print_log(f'Dataset loaded from {ds_path}', logger)
else:
    print_log('Start', logger)
    data_dir = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/3_hour/hottest_quant'
    quant_type = 'REFHT_quant99'
    exp_name = 'pre_industrial'
    var_keep = ['T', 'Q', 'Z3', 'PS', 'P0', 'hyam', 'hybm']
    ds = xr.open_dataset(os.path.join(data_dir, exp_name, quant_type, 'output.nc'))
    if small_ds:
        lat_min = 0
        lat_max = 50
        lon_min = 0
        lon_max = 20
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max), sample=slice(0, 2))
        print_log(f'Only using small subset | Lat {lat_min} to {lat_max} | Lon {lon_min} to {lon_max} | Sample 1 and 2', logger)
    ds = ds[var_keep].load()
    ds.attrs['co2'] = get_co2_multiplier(exp_name)
    p0 = float(ds.P0)
    ds = ds.drop_vars('P0')
    ds.attrs['P0'] = p0
    print_log('Loaded in Data', logger)

    # Add variables to compute LCL
    ds['P'] = cesm.get_pressure(ds.PS, ds.P0, ds.hyam, ds.hybm)
    ds['P_diff'] = dp_from_pressure(ds.P)
    ds['TREFHT'] = ds.T.isel(lev=-1)
    ds['QREFHT'] = ds.Q.isel(lev=-1)
    ds['ZREFHT'] = ds.Z3.isel(lev=-1)
    ds['PREFHT'] = ds.P.isel(lev=-1)
    ds['lnb_ind'] = get_lnb_lev_ind(ds.T, ds.Z3, ds.P)
    ds = ds.drop_vars('Q')
    print_log('Added variables for LCL computation', logger)

    # Compute empirical estimate of LCL
    print_log('Empirical LCL | Start', logger)
    small = 1
    use_lev = ds.P >= ds.P.isel(lev=ds.lnb_ind) - small
    error_rms = get_mse_prof_rms(ds.T.where(use_lev), ds.P.where(use_lev), ds.Z3.where(use_lev), ds.P_diff.where(use_lev))
    lcl_lev_ind = error_rms.argmin(dim='lev')
    print_log('Empirical LCL | Computed best model level', logger)
    pressure_min = ds.P.isel(lev=lcl_lev_ind-1)
    pressure_max = ds.P.isel(lev=lcl_lev_ind+1)
    n_split = 20
    ds_split = get_ds_in_pressure_range(ds[['T', 'Z3', 'P']], pressure_min, pressure_max, n_split,
                                        pressure_dim_name_out='lev_fine_ind')
    ds_split = ds_split.drop_vars('lev')
    print_log('Empirical LCL | Computed fine grid about model level', logger)
    error_rms = get_mse_prof_rms(ds.T.where(use_lev), ds.P.where(use_lev), ds.Z3.where(use_lev),
                          ds.P_diff.where(use_lev), ds_split.T, ds_split.P, ds_split.Z3, split_dim='lev_fine_ind')
    p_lcl_emp = ds_split.P.isel(lev_fine_ind=error_rms.argmin(dim='lev_fine_ind'))
    error_rms_emp = error_rms.min(dim='lev_fine_ind')
    print_log('Empirical LCL | End', logger)

    # Compute RMS error from physical LCL
    p_lcl = lcl_metpy(ds.TREFHT, ds.QREFHT, ds.PREFHT)[0]
    ds_lcl_phys = get_ds_in_pressure_range(ds[['T', 'Z3', 'P']], p_lcl, p_lcl+1, n_pressure=1,
                                           pressure_dim_name_out='lev_fine_ind')
    error_rms = get_mse_prof_rms(ds.T.where(use_lev), ds.P.where(use_lev), ds.Z3.where(use_lev),
                          ds.P_diff.where(use_lev), ds_lcl_phys.T, ds_lcl_phys.P, ds_lcl_phys.Z3, split_dim='lev_fine_ind')
    error_rms = error_rms.min(dim='lev_fine_ind')
    print_log('Computed Physical LCL', logger)

    # Correct empirical LCL to be physical if the error is less
    p_lcl_emp = p_lcl_emp.where(error_rms_emp < error_rms, p_lcl)
    error_rms_emp = error_rms_emp.where(error_rms_emp < error_rms, error_rms)

    # Save data
    ds_lcl = xr.Dataset({'p_lcl': p_lcl, 'error_rms': error_rms, 'p_lcl_emp': p_lcl_emp, 'error_rms_emp': error_rms_emp})
    ds_lcl = ds_lcl.drop_vars('lev_fine_ind')
    ds_lcl = convert_ds_dtypes(ds_lcl)
    comp_level = 4
    if (not os.path.exists(ds_path)) and save_processed:
        ds_lcl.to_netcdf(ds_path, format="NETCDF4",
                     encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_lcl.data_vars})
    print_log('End', logger)

hi = 5

