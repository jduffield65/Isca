# Script to save REFHT and 500hPa level info for all days corresponding to a given quantile of TREFHT.
# Idea is that this gives useful variables to decompose vertical coupling between REFHT and 500hPa on hot days.
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from save_info import load_raw_data
from isca_tools import cesm
from isca_tools.papers.byrne_2021 import get_quant_ind
from isca_tools.utils.moist_physics import moist_static_energy, sphum_sat
from isca_tools.utils.ds_slicing import lat_lon_range_slice
from isca_tools.utils import get_memory_usage, len_safe
from isca_tools.utils.base import parse_int_list, print_log
from isca_tools.utils.xarray import set_attrs
from isca_tools.utils.constants import g
import sys
import os
import numpy as np
import f90nml
import warnings
import fnmatch
import logging
import time
import inspect
import xarray as xr
from typing import Optional, Union, List

# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

def main(input_file_path: str):
    logger = logging.getLogger()  # for printing to console time info
    logger.info("Start")
    input_info = f90nml.read(input_file_path)
    script_info = input_info['script_info']
    if script_info['out_dir'] is None:
        script_info['out_dir'] = os.path.dirname(input_file_path)
    if script_info['use_in_calc_name'] is None:
        script_info['use_in_calc_name'] = script_info['out_name']
    out_file = os.path.join(script_info['out_dir'], script_info['out_name'])
    if os.path.exists(out_file):
        if script_info['overwrite']:
            warnings.warn(f'Output file already exists at:\n{out_file}\nWill be overwriten')
        else:
            # Raise error if output data already exists
            raise ValueError('Output file already exists at:\n{}'.format(out_file))

    # Load daily average data
    # Arguments required for `load_raw_data` are all in `script_info` with same name, hence inspect.signature stuff
    func_arg_names = inspect.signature(load_raw_data).parameters
    func_args = {k: v for k, v in script_info.items() if k in func_arg_names}
    ds = load_raw_data(**func_args, logger=logger)
    logger.info(f"Finished lazy-loading datasets | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Breakdown T_ft into zonal average and anomaly from this
    ds['T_zonal_av'] = ds.T.mean(dim='lon')     # zonal average is computed over all surfaces
    ds['T_anom'] = ds['T'] - ds['T_zonal_av']
    if 'dayofyear' in script_info['var_save']:
        ds['dayofyear'] = ds.time.dt.dayofyear
    ds = ds[script_info['var_save']]
    first_var = script_info['var_save'][0]

    quant_mask = xr.open_dataset(os.path.join(script_info['use_in_calc_dir'], script_info['use_in_calc_name'])
                                 ).sel(quant=script_info['quant']).use_in_calc
    quant_mask = quant_mask.load() > 0
    if quant_mask.time.size != ds.time.size:
        raise ValueError(f'Dataset size mismatch in time: quant_mask has {quant_mask.time.size} values, but dataset has {ds.time.size} values\n'
                         f'May need to re-run dataset to find quant_mask with all years')
    # Reindex quant_mask so has same lat and lon coordinates as input dataset
    quant_mask = quant_mask.reindex_like(ds[first_var], method="nearest", tolerance=0.01)
    mask_load = quant_mask > 0

    ds = ds.isel(plev=0).where(mask_load)
    logger.info(f"Finished masking data | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Fully load data if chose to do that
    if script_info['load_all_at_start']:
        ds.load()
        logger.info(f"Fully loaded data | Memory used {get_memory_usage() / 1000:.1f}GB")

    n_days_notnan = (~np.isnan(ds[first_var])).sum(dim='time')
    n_days_save = int(n_days_notnan.max())
    # Sanity check that each coordinate has same amount of non-nan days
    n_days_save_range = float(n_days_notnan.max() - n_days_notnan.min())
    logger.info(f"Number of days for each coordinate is {n_days_save}")
    if n_days_save_range > 0:
        logger.info(f"Range in number of non-nan days across lat and lon is non-zero: {n_days_save_range}")

    output_info = {key: np.full((ds.lat.size, ds.lon.size, n_days_save), np.nan) for key in ds}
    for i in range(ds.lat.size):
        for j in range(ds.lon.size):
            ind_time_use = np.where(~np.isnan(ds[first_var].isel(lat=i, lon=j)))[0]
            for key in ds:
                # Account for fact that number of days to save can be different at different locations
                output_info[key][i, j, :len(ind_time_use)] = ds[key].isel(lat=i, lon=j, time=ind_time_use)
        logger.info(f"Finished latitude {i+1}/{ds.lat.size} | Memory used {get_memory_usage() / 1000:.1f}GB")
    # Info for converting numpy arrays to
    output_dims = ['lat', 'lon', 'sample']
    coords = {'lat': ds.lat, 'lon': ds.lon, 'sample': np.arange(n_days_save)}
    output_long_name = {}
    output_units = {}
    for var in script_info['var_save']:
        output_long_name[var] = ds[var].long_name if 'long_name' in ds[var].attrs else ''
        output_units[var] = ds[var].units if 'units' in ds[var].attrs else ''

    # Convert output dict into xarray dataset
    # Convert individual arrays
    for var in script_info['var_save']:
        output_info[var] = xr.DataArray(output_info[var], dims=output_dims,
                                        coords={key: coords[key] for key in coords})
        output_info[var] = set_attrs(output_info[var], long_name=output_long_name[var],
                                     units=output_units[var])

    # Add pressure level info to variables at single level
    for var in ['mse_sat_ft', 'mse_lapse', 'T', 'Z3', 'T_zonal_av', 'T_anom']:
        if var not in output_info:
            continue
        # Need np.atleast1d because was getting error when plev is just a number not an array
        output_info[var] = output_info[var].expand_dims(plev=np.atleast_1d(ds.coords['plev']))
    # Convert dict to dataset
    ds_out = xr.Dataset(output_info)
    ds_out = ds_out.expand_dims(quant=[script_info['quant']])           # add quant as a coordinate
    ds_out = ds_out.astype('float32')           # save memory
    if 'dayofyear' in ds_out:
        # dayofyear is integer so save as such. Can't have nan as integer so replace with negative
        ds_out['dayofyear'] = ds_out['dayofyear'].fillna(-1).astype('int16')
    # Save output to nd2 file with compression - reduces size of file by factor of 10
    # Compression makes saving step slower
    ds_out.to_netcdf(os.path.join(script_info['out_dir'], script_info['out_name']), format="NETCDF4",
                     encoding={var: {"zlib": True, "complevel": script_info['complevel']} for var in ds_out.data_vars})
    logger.info("End")


if __name__ == "__main__":
    main(sys.argv[1])
