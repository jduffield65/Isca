# Uses physical LCL to compute the mod_parcel1 and const1 lapse rate fitting
import os
import numpy as np
import xarray as xr
import logging
import sys
sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')

from isca_tools.utils import get_memory_usage
from isca_tools.utils.base import print_log
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.xarray import print_ds_var_list, convert_ds_dtypes
from isca_tools.convection.base import lcl_metpy
from isca_tools.thesis.lapse_theory import get_var_at_plev
from geocat.comp.interpolation import interp_hybrid_to_pressure
from isca_tools.thesis.lapse_integral_simple import fitting_2_layer_xr
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

# File location Info
from jobs.theory_lapse.scripts.lcl import load_ds_quant, data_dir, exp_name, var_keep
small_ds = False
quant_type = 'REFHT_quant50'
processed_dir = [os.path.join(data_dir, exp_name[i], quant_type, 'lapse_fitting') for i in range(len(exp_name))]
processed_file_name = 'ds_lapse_simple.nc'  # combined file from all samples
var_keep = ['T', 'Q', 'Z3', 'PS', 'P0', 'hyam', 'hybm', 'CAPE', 'FREQZM']#, 'T_zonal_av']
p_ft = 400 * 100            # Free tropopsheric level
comp_level = 4
n_lev_above_integral = 3
temp_surf_lcl_calc = 300       # Compute LCL from RH using 300K as surf temp

def get_ds_quant_lat(co2_ind, lat_ind, temp_surf_lcl_calc=temp_surf_lcl_calc,
                     n_lev_above_integral=n_lev_above_integral, p_ft=p_ft):
    ds = load_ds_quant([exp_name[co2_ind]], quant_type, data_dir, var_keep, compute_p_diff=False,
                       lat_ind=lat_ind)
    # Compute rh
    ds['rh_REFHT'] = ds.QREFHT / sphum_sat(ds.TREFHT, ds.PREFHT)

    # Interpolate data onto FT level
    ds['T_ft_env'] = interp_hybrid_to_pressure(ds.T, ds.PS, ds.hyam, ds.hybm, ds.P0, np.atleast_1d(p_ft),
                                               lev_dim='lev')
    ds['T_ft_env'].load()
    ds['Z_ft_env'] = interp_hybrid_to_pressure(ds.Z3, ds.PS, ds.hyam, ds.hybm, ds.P0, np.atleast_1d(p_ft),
                                               lev_dim='lev')
    ds['Z_ft_env'].load()
    ds = ds.isel(plev=0)                    # remove plev as a dimension
    ds = ds.rename_vars({"plev": "p_ft"})   # change to p_ft
    ds.attrs['temp_surf_lcl_calc'] = temp_surf_lcl_calc
    ds.attrs['n_lev_above_integral'] = n_lev_above_integral
    return ds


if __name__ == '__main__':
    logger = logging.getLogger()  # for printing to console time info

    # ds = load_ds_quant(exp_name, quant_type, data_dir, var_keep, small_ds=small_ds, compute_p_diff=False,
    #                    load_fully=False)
    print_log(f'Quantile data loaded | Memory used {get_memory_usage() / 1000:.1f}GB',
              logger)

    load_processed = [os.path.exists(os.path.join(processed_dir[i], processed_file_name)) for i in range(len(exp_name))]

    # Compute empirical estimate of LCL
    lat_vals = load_ds_quant([exp_name[0]], quant_type, data_dir, ['TREFHT'], compute_p_diff=False,
                          sample_ind=0, lon_ind=0).lat
    n_lat = lat_vals.size
    n_files = len(exp_name) * n_lat      # One file for each co2 conc and sample due to speed
    print_log(f'Empirical lapse fitting for {n_files} Files | Start', logger)
    # ds.attrs['p_lcl_log_mod'] = np.asarray([-10, -5, -1, 0, 1, 5, 10])   # check if work with rediculous lcl
    # ds.attrs['n_lev_above_integral'] = n_lev_above_integral

    var_names = ['lapse', 'integral', 'error']
    n_digit = len(str(n_lat))
    # One file for each latitude and co2
    for i in range(len(exp_name)):
        if load_processed[i]:
            print_log(f'Files already exist for {exp_name[i]}', logger)
            continue
        path_use = [os.path.join(processed_dir[i], f'lat{j:0{n_digit}d}.nc') for j in range(n_lat)]
        for j in range(n_lat):
            if os.path.exists(path_use[j]):
                print_log(f'File {i * n_lat + j + 1}/{n_files} Exists Already', logger)
                continue
            print_log(f'File {i * n_lat + j + 1}/{n_files} | Start | Memory used {get_memory_usage() / 1000:.1f}GB', logger)
            ds_use = get_ds_quant_lat(i, j)
            print_log(f'File {i * n_lat + j + 1}/{n_files} | Data loaded | Memory used {get_memory_usage() / 1000:.1f}GB',
                      logger)
            for key in ['const', 'mod_parcel']:
                var = fitting_2_layer_xr(ds_use.T, ds_use.P, ds_use.TREFHT, ds_use.PREFHT, ds_use.rh_REFHT,
                                         ds_use.T_ft_env, float(ds_use.p_ft),
                                         n_lev_above_upper2_integral=ds_use.n_lev_above_integral,
                                         temp_surf_lcl_calc=ds_use.temp_surf_lcl_calc, method_layer2=key)
                for k, key2 in enumerate(var_names):
                    ds_use[f'{key}1_{key2}'] = var[k]
                print_log(f'File {i * n_lat + j + 1}/{n_files} | {key}1 Lapse Fitting Complete | Memory used {get_memory_usage() / 1000:.1f}GB',
                          logger)

            # Save data
            ds_use['layer'] = xr.DataArray(['below lcl', 'above lcl'], name='layer', dims='layer')
            ds_use = convert_ds_dtypes(ds_use)
            if not os.path.exists(path_use[j]):
                ds_use.to_netcdf(path_use[j], format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_use.data_vars})
            print_log(f'File {i*n_lat+j+1}/{n_files} | Saved | Memory used {get_memory_usage() / 1000:.1f}GB', logger)

        if not os.path.exists(os.path.join(processed_dir[i], processed_file_name)):
            # Combine all sample files into a single file for each experiment
            ds_lapse = xr.concat([xr.load_dataset(path_use[j]) for j in range(n_lat)], dim=lat_vals)
            ds_lapse.to_netcdf(os.path.join(processed_dir[i], processed_file_name), format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_lapse.data_vars})
            print_log(f'{exp_name[i]} | Combined samples into one {processed_file_name} File', logger)

    hi = 5