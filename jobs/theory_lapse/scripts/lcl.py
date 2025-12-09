# Script to compute empirical LCL which minimizes error in MSE environmental compared to convective profile
import numpy as np
import os
import xarray as xr
import re
import logging
import sys
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

def find_lcl_empirical2(temp_env, p_env, z_env, temp_start=None, p_start=None, temp_pot_thresh=2,
                        temp_pot_thresh_lapse=0.5, lapse_thresh=8.5):
    # Find the lowest model layer with lapse rate less than lapse_thresh
    # LCL is the level at which the pot temp drops by 0.5K within this layer
    if temp_start is None:
        temp_start = temp_env.isel(lev=-1)
    if p_start is None:
        p_start = p_env.isel(lev=-1)
    temp_pot_env = potential_temp(temp_env, p_env)

    # First mask is pot temp close to surface pot temperature
    temp_pot_start = potential_temp(temp_start, p_start)
    mask_temp = np.abs(temp_pot_env - temp_pot_start) <= temp_pot_thresh

    # Second mask is lapse rate close to dry adiabat
    # lower is so append high value at surface
    lapse = -temp_env.diff(dim='lev', label='lower') / z_env.diff(dim='lev', label='lower') * 1000
    lapse = lapse.reindex_like(temp_env)  # make same shape
    lapse = lapse.fillna(lapse_thresh + 5)  # ensure final value satisfies lapse criteria
    mask_lapse = lapse > lapse_thresh
    mask = (mask_temp & mask_lapse)
    lcl_ind = ((mask.where(mask, other=np.nan) * np.arange(lapse.lev.size)).min(dim='lev')).astype(int)

    p_low = np.log10(p_env.isel(lev=lcl_ind))  # use log as better for interpolation - gradient is approx constant
    p_high = np.log10(p_env.isel(lev=lcl_ind - 1))  # further from surface

    # print(p_low)
    # print(p_high)
    # print(np.log10(p_env))

    temp_pot_low = temp_pot_env.isel(lev=lcl_ind)
    temp_pot_high = temp_pot_env.isel(lev=lcl_ind - 1)
    gradient = (temp_pot_high - temp_pot_low) / (p_high - p_low)
    temp_pot_target = temp_pot_low + temp_pot_thresh_lapse
    p_target = p_low + (temp_pot_target - temp_pot_low) / gradient
    p_target = p_target.clip(min=p_high)
    p_lcl = 10 ** p_target
    # print(dry_profile_temp(temp_start, p_start, p_lcl))
    # print(lapse.isel(lev=lcl_ind-1))
    return p_lcl, dry_profile_temp(temp_start, p_start, p_lcl)

def load_ds_quant(exp_name: list = ['pre_industrial', 'co2_2x'], quant_type: str = 'REFHT_quant99',
                  data_dir: str = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/3_hour/hottest_quant',
                  var_keep: list = ['T', 'Q', 'Z3', 'PS', 'P0', 'hyam', 'hybm', 'CAPE', 'FREQZM'],
                  compute_p_diff: bool = False,
                  lat_ind = None, lon_ind = None, sample_ind = None):
    co2_vals = [get_co2_multiplier(i) for i in exp_name]
    ds = [xr.open_dataset(os.path.join(data_dir, exp_name[i], quant_type, 'output.nc')) for i in range(len(exp_name))]
    if len(exp_name) == 1:
        ds = ds[0]
    else:
        ds = xr.concat(ds, dim=xr.DataArray(co2_vals, dims="co2", coords={"exp_name": ("co2", exp_name)}))
    if lat_ind is not None:
        ds = ds.isel(lat=lat_ind)
    if lon_ind is not None:
        ds = ds.isel(lon=lon_ind)
    if sample_ind is not None:
        ds = ds.isel(sample=sample_ind)
    ds = ds[var_keep].load()
    if ('hyam' in var_keep) & (len(exp_name) != 1):
        ds['hyam'] = ds.hyam.isel(co2=0)
    if ('hybm' in var_keep) & (len(exp_name) != 1):
        ds['hybm'] = ds.hybm.isel(co2=0)
    if 'P0' in var_keep:
        if len(exp_name) == 1:
            p0 = float(ds.P0)
        else:
            p0 = float(ds.P0.isel(co2=0))
        ds = ds.drop_vars('P0')
        ds.attrs['P0'] = p0

    # Add variables to compute LCL
    if 'PS' in var_keep:
        ds['P'] = cesm.get_pressure(ds.PS, ds.P0, ds.hyam, ds.hybm)
    if compute_p_diff:
        ds['P_diff'] = dp_from_pressure(ds.P)
    if 'T' in var_keep:
        ds['TREFHT'] = ds.T.isel(lev=-1)
    if 'Q' in var_keep:
        ds['QREFHT'] = ds.Q.isel(lev=-1)
        ds = ds.drop_vars('Q')
    if 'Z3' in var_keep:
        ds['ZREFHT'] = ds.Z3.isel(lev=-1)
    if 'PS' in var_keep:
        ds['PREFHT'] = ds.P.isel(lev=-1)
    if ('PS' in var_keep) & ('T' in var_keep) & ('Z3' in var_keep):
        ds['lnb_ind'] = get_lnb_lev_ind(ds.T, ds.Z3, ds.P)
    return ds


small_ds = False                 # use a small subset of data for testing
save_processed = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)
logger = logging.getLogger()  # for printing to console time info



# Always load in sample data, so can easily import into jupyter notebook
# File location Info
print_log('Start', logger)
data_dir = '/Users/joshduffield/Documents/StAndrews/Isca/jobs/cesm/3_hour/hottest_quant'
quant_type = 'REFHT_quant99'
exp_name = ['pre_industrial', 'co2_2x']
processed_dir = [os.path.join(data_dir, exp_name[i], quant_type, 'lcl_calc') for i in range(len(exp_name))]
processed_file_name = 'ds_lcl.nc'           # combined file from all samples
load_processed = [os.path.exists(os.path.join(processed_dir[i], processed_file_name)) for i in range(len(exp_name))]
var_keep = ['T', 'Q', 'Z3', 'PS', 'P0', 'hyam', 'hybm', 'CAPE', 'FREQZM']
co2_vals = [get_co2_multiplier(i) for i in exp_name]

# Load in LCL calc data if exists
if all(load_processed):
    ds_lcl = [xr.open_dataset(os.path.join(processed_dir[i], processed_file_name)) for i in range(len(exp_name))]
    ds_lcl = xr.concat(ds_lcl, dim=xr.DataArray(co2_vals, dims="co2", coords={"exp_name": ("co2", exp_name)}))
    ds_lcl = ds_lcl.load()
    print_log(f"LCL Info loaded from {processed_file_name} files", logger)
else:
    ds_lcl = None

# If run script, compute the empirical LCL - takes a while so do one sample at a time
if (__name__ == '__main__') and not all(load_processed):
    if small_ds:
        lat_ind = slice(10, 30)
        lon_ind = slice(0, 10)
        sample_ind = slice(0, 3)
        print_log(f'Only using small subset', logger)
    else:
        lat_ind = None
        lon_ind = None
        sample_ind = None
    ds = load_ds_quant(exp_name, quant_type, data_dir, var_keep, True, lat_ind, lon_ind, sample_ind)
    print_log('Loaded in Data', logger)
    # Compute empirical estimate of LCL
    n_files = ds.co2.size * ds.sample.size      # One file for each co2 conc and sample due to speed
    print_log(f'Empirical and Physical LCL for {n_files} Files | Start', logger)
    small = 1               # units of Pa just to ensure use pressure below LNB
    comp_level = 4
    n_split = 20            # number of pressure values to use in the fine grid
    for i in range(ds.co2.size):
        if load_processed[i]:
            print_log(f'Files already exist for {exp_name[i]}', logger)
            continue
        path_use = [os.path.join(processed_dir[i], f'sample{int(ds.sample[j])}.nc') for j in range(ds.sample.size)]
        for j in range(ds.sample.size):
            if os.path.exists(path_use[j]):
                print_log(f'File {i * ds.sample.size + j + 1}/{n_files} Exists Already', logger)
                continue
            print_log(f'File {i * ds.sample.size + j + 1}/{n_files} | Start', logger)

            # Compute RMS error of MSE profile with LCL at each of the model levels
            ds_use = ds.isel(co2=i, sample=j)
            use_lev = ds_use.P >= ds_use.P.isel(lev=ds_use.lnb_ind) - small
            error_rms = get_mse_prof_rms(ds_use.T.where(use_lev), ds_use.P.where(use_lev), ds_use.Z3.where(use_lev), ds_use.P_diff.where(use_lev))
            lcl_lev_ind = error_rms.argmin(dim='lev')
            print_log(f'File {i*ds.sample.size+j+1}/{n_files} | Computed best model level', logger)

            # Build a fine grid of n_split pressure levels around best LCL model level
            pressure_min = ds_use.P.isel(lev=np.clip(lcl_lev_ind-1, 0, ds.lev.size-1))
            pressure_max = ds_use.P.isel(lev=np.clip(lcl_lev_ind+1, 0, ds.lev.size-1))
            ds_split = get_ds_in_pressure_range(ds_use[['T', 'Z3', 'P']], pressure_min, pressure_max, n_split,
                                                pressure_dim_name_out='lev_fine_ind')
            ds_split = ds_split.drop_vars('lev')
            print_log(f'File {i*ds.sample.size+j+1}/{n_files} | Computed fine grid about model level', logger)

            # On this fine grid, repeat RMS error of MSE profile calculation - select min error as the LCL
            error_rms = get_mse_prof_rms(ds_use.T.where(use_lev), ds_use.P.where(use_lev), ds_use.Z3.where(use_lev),
                                  ds_use.P_diff.where(use_lev), ds_split.T, ds_split.P, ds_split.Z3, split_dim='lev_fine_ind')
            p_lcl_emp = ds_split.P.isel(lev_fine_ind=error_rms.argmin(dim='lev_fine_ind'))
            error_rms_emp = error_rms.min(dim='lev_fine_ind')
            print_log(f'File {i*ds.sample.size+j+1}/{n_files} | Computed Empirical LCL', logger)

            # Compute RMS error from physical LCL
            p_lcl = lcl_metpy(ds_use.TREFHT, ds_use.QREFHT, ds_use.PREFHT)[0]
            ds_lcl_phys = get_ds_in_pressure_range(ds_use[['T', 'Z3', 'P']], p_lcl, p_lcl+1, n_pressure=1,
                                                   pressure_dim_name_out='lev_fine_ind')
            error_rms = get_mse_prof_rms(ds_use.T.where(use_lev), ds_use.P.where(use_lev), ds_use.Z3.where(use_lev),
                                  ds_use.P_diff.where(use_lev), ds_lcl_phys.T, ds_lcl_phys.P, ds_lcl_phys.Z3, split_dim='lev_fine_ind')
            error_rms = error_rms.min(dim='lev_fine_ind')
            print_log(f'File {i*ds.sample.size+j+1}/{n_files} | Computed Physical LCL', logger)

            # Correct empirical LCL to be physical if the error is less
            p_lcl_emp = p_lcl_emp.where(error_rms_emp < error_rms, p_lcl)
            error_rms_emp = error_rms_emp.where(error_rms_emp < error_rms, error_rms)

            # Save data
            ds_lcl = xr.Dataset({'p_lcl': p_lcl, 'error_rms': error_rms, 'p_lcl_emp': p_lcl_emp, 'error_rms_emp': error_rms_emp})
            ds_lcl = ds_lcl.drop_vars('lev_fine_ind')
            ds_lcl = convert_ds_dtypes(ds_lcl)
            if (not os.path.exists(path_use[j])) and save_processed:
                ds_lcl.to_netcdf(path_use[j], format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_lcl.data_vars})
            print_log(f'File {i*ds.sample.size+j+1}/{n_files} | Saved', logger)

        if (not os.path.exists(os.path.join(processed_dir[i], processed_file_name))) and save_processed:
            # Combine all sample files into a single file for each experiment
            ds_lcl = xr.concat([xr.load_dataset(path_use[j]) for j in range(ds.sample.size)], dim=ds.sample)
            ds_lcl.to_netcdf(os.path.join(processed_dir[i], processed_file_name), format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_lcl.data_vars})
            print_log(f'{exp_name[i]} | Combined samples into one {processed_file_name} File', logger)

print_log('End', logger)
hi = 5

