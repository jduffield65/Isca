# Uses physical LCL to compute the mod_parcel1 and const1 lapse rate fitting
import os
import numpy as np
import xarray as xr
import logging
import sys
from isca_tools.utils.base import print_log
from isca_tools.utils.moist_physics import sphum_sat
from isca_tools.utils.xarray import print_ds_var_list, convert_ds_dtypes
from isca_tools.convection.base import lcl_metpy
from isca_tools.thesis.lapse_theory import get_var_at_plev
from geocat.comp.interpolation import interp_hybrid_to_pressure
from isca_tools.thesis.lapse_integral_simple import fitting_2_layer_xr
from jobs.theory_lapse.scripts.lcl import load_ds_quant, data_dir, exp_name, var_keep
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

# File location Info
small_ds = True
quant_type = 'REFHT_quant99'
processed_dir = [os.path.join(data_dir, exp_name[i], quant_type, 'lapse_fitting') for i in range(len(exp_name))]
processed_file_name = 'ds_lapse_simple.nc'  # combined file from all samples
var_keep = ['T', 'Q', 'Z3', 'PS', 'P0', 'hyam', 'hybm', 'CAPE', 'FREQZM']#, 'T_zonal_av']
p_ft = 400 * 100            # Free tropopsheric level
comp_level = 4
n_lev_above_integral = 3
temp_surf_lcl_calc = 'median'       # Median to compute from the data


if __name__ == '__main__':
    logger = logging.getLogger()  # for printing to console time info

    ds = load_ds_quant(exp_name, quant_type, data_dir, var_keep, small_ds=small_ds, compute_p_diff=False)

    # Compute RH
    ds['rh_REFHT'] = ds.QREFHT / sphum_sat(ds.TREFHT, ds.PREFHT)

    if temp_surf_lcl_calc == 'median':
        temp_surf_lcl_calc = float(np.ceil(ds.TREFHT.median()))
    ds.attrs['temp_surf_lcl_calc'] = temp_surf_lcl_calc

    # Interpolate data onto FT level
    ds['T_ft_env'] = interp_hybrid_to_pressure(ds.T, ds.PS, ds.hyam, ds.hybm, ds.P0, np.atleast_1d(p_ft),
                                               lev_dim='lev')
    ds['T_ft_env'].load()
    ds['Z_ft_env'] = interp_hybrid_to_pressure(ds.Z3, ds.PS, ds.hyam, ds.hybm, ds.P0, np.atleast_1d(p_ft),
                                               lev_dim='lev')
    ds['Z_ft_env'].load()
    ds = ds.isel(plev=0)                    # remove plev as a dimension
    ds = ds.rename_vars({"plev": "p_ft"})   # change to p_ft
    print_log('FT Temp and Z Computed', logger)

    load_processed = [os.path.exists(os.path.join(processed_dir[i], processed_file_name)) for i in range(len(exp_name))]

    # Compute empirical estimate of LCL
    n_files = ds.co2.size * ds.lat.size      # One file for each co2 conc and sample due to speed
    print_log(f'Empirical lapse fitting for {n_files} Files | Start', logger)
    # ds.attrs['p_lcl_log_mod'] = np.asarray([-10, -5, -1, 0, 1, 5, 10])   # check if work with rediculous lcl
    ds.attrs['n_lev_above_integral'] = n_lev_above_integral

    var_names = ['lapse', 'integral', 'error']
    n_digit = len(str(ds.lat.size))
    # One file for each latitude and co2
    for i in range(ds.co2.size):
        if load_processed[i]:
            print_log(f'Files already exist for {exp_name[i]}', logger)
            continue
        path_use = [os.path.join(processed_dir[i], f'lat{j:0{n_digit}d}.nc') for j in range(ds.lat.size)]
        for j in range(ds.lat.size):
            if os.path.exists(path_use[j]):
                print_log(f'File {i * ds.lat.size + j + 1}/{n_files} Exists Already', logger)
                continue
            print_log(f'File {i * ds.lat.size + j + 1}/{n_files} | Start', logger)
            ds_use = ds.isel(co2=i, lat=j)
            for key in ['const', 'mod_parcel']:
                var = fitting_2_layer_xr(ds_use.T, ds_use.P, ds_use.TREFHT, ds_use.PREFHT, ds_use.rh_REFHT,
                                         ds_use.T_ft_env, float(ds.p_ft),
                                         n_lev_above_upper2_integral=ds.n_lev_above_integral,
                                         temp_surf_lcl_calc=ds.temp_surf_lcl_calc, method_layer2=key)
                # Must include fillna as inf to deal with all nan slice.
                for k, key2 in enumerate(var_names):
                    ds_use[f'{key}1_{key2}'] = var[k]

            # Save data
            ds_use['layer'] = xr.DataArray(['below lcl', 'above lcl'], name='layer', dims='layer')
            ds_use = convert_ds_dtypes(ds_use)
            if not os.path.exists(path_use[j]):
                ds_use.to_netcdf(path_use[j], format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_use.data_vars})
            print_log(f'File {i*ds.lat.size+j+1}/{n_files} | Saved', logger)

        if not os.path.exists(os.path.join(processed_dir[i], processed_file_name)):
            # Combine all sample files into a single file for each experiment
            ds_lapse = xr.concat([xr.load_dataset(path_use[j]) for j in range(ds.lat.size)], dim=ds.lat)
            ds_lapse.to_netcdf(os.path.join(processed_dir[i], processed_file_name), format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_lapse.data_vars})
            print_log(f'{exp_name[i]} | Combined samples into one {processed_file_name} File', logger)

    hi = 5