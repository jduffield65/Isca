import xarray as xr
import numpy as np
import os
import logging
import sys
from tqdm import tqdm

sys.path.append('/Users/joshduffield/Documents/StAndrews/Isca')
from isca_tools.utils.constants import L_v, c_p, g, R, kappa, lapse_dry
from isca_tools.utils.base import print_log
import jobs.theory_lapse.cesm.thesis_figs.scripts.utils as utils
from isca_tools.utils.xarray import convert_ds_dtypes

test_mode = False
comp_level = 4

if __name__ == '__main__':
    try:
        # ideally get quant_type from terminal
        surf = sys.argv[1]  # e.g. 'ocean'
    except IndexError:
        # Default values
        surf = 'land'
    try:
        lat_ind_start = int(sys.argv[2])
    except (IndexError, ValueError):
        lat_ind_start = 1  # starts at 1
    try:
        lat_ind_end = int(sys.argv[3])
    except (IndexError, ValueError):
        lat_ind_end = None
    out_path_cape = os.path.join(utils.out_dir, f'ds_tropics_{surf}_cape.nc')
    if os.path.exists(out_path_cape):
        sys.exit(f'File {out_path_cape} already exists')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
    logger = logging.getLogger()  # for printing to console time info

    path_input = os.path.join(utils.out_dir, f"ds_tropics_{surf}.nc")
    print_log(f'Loading data from {path_input}', logger)
    p_ft_use = 400 * 100
    ds = xr.load_dataset(path_input).sel(p_ft=p_ft_use)
    if test_mode:
        # So will run quickly
        ds = ds.isel(lon_sample=slice(0, 10), quant=slice(0, 20))
    print_log('Loaded data', logger)

    # Chose best rh_mod
    ds = utils.sel_best_rh_mod(ds)
    # Record actual rh_mod value
    ds['mod_parcel_rh_mod'] = ds.rh_mod[ds.mod_parcel_rh_mod_ind]

    ds['lapse_Dz'] = ds.mod_parcel_lapse.isel(layer=0) / 1000 - lapse_dry
    ds['lapse_Mz'] = ds.mod_parcel_lapse.isel(layer=1) / 1000
    ds['lapse_D'] = R / g * ds.TREFHT * ds['lapse_Dz']
    ds['lapse_M'] = R / g * ds.T_ft_env * ds['lapse_Mz']

    lat_weights = utils.lat_weights.reindex_like(ds.lat)
    temp_surf_lcl_calc = ds.temp_surf_lcl_calc
    p_ft = float(ds.p_ft)

    n_lat = ds.lat.size
    if lat_ind_end is None:
        lat_ind_end = n_lat
    n_digit = len(str(n_lat))
    path_use = [os.path.join(utils.out_dir, 'cape_tropics', surf, f'lat{j:0{n_digit}d}.nc') for j in range(n_lat)]
    for i in range(n_lat):
        if (i + 1 < lat_ind_start) or (i + 1 > lat_ind_end):
            # Don't get data for these latitudes as outside range selected
            print_log(f'File {i + 1}/{n_lat} | '
                      f'Skipped - outside lat ind range of {lat_ind_start}-{lat_ind_end}', logger)
            continue
        if os.path.exists(path_use[i]):
            print_log(f'File {i + 1}/{n_lat} Exists Already', logger)
            continue
        print_log(f'File {i + 1}/{n_lat} | Start', logger)
        ds_cape = utils.get_ds_cape(ds.isel(quant=i),
                                    ds['mod_parcel_rh_mod'].isel(quant=i),
                                    p_ft, temp_surf_lcl_calc)
        print_log(f'File {i + 1}/{n_lat} | Obtained CAPE info', logger)
        ds_cape = convert_ds_dtypes(ds_cape)
        if not os.path.exists(path_use[i]):
            ds_cape.to_netcdf(path_use[i], format="NETCDF4",
                             encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_cape.data_vars})
        print_log(f'File {i+1}/{n_lat} | Saved', logger)
    # ds_cape = xr.concat(ds_cape, dim=ds.quant)
    # comp_level = 4
    # ds_cape = convert_ds_dtypes(ds_cape)
    # if not os.path.exists(out_path_cape):
    #     ds_cape.to_netcdf(out_path_cape, format="NETCDF4",
    #                       encoding={var: {"zlib": True, "complevel": comp_level} for var in ds_cape.data_vars})
    #     print('Saved ds_cape to {}'.format(out_path_cape))
