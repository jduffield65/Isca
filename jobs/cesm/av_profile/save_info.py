import copy
import os
from isca_tools import cesm
from isca_tools.papers.byrne_2021 import get_quant_ind
from isca_tools.utils.moist_physics import moist_static_energy, sphum_sat
from isca_tools.utils.ds_slicing import lat_lon_range_slice
from isca_tools.utils import get_memory_usage, len_safe
from isca_tools.utils.base import parse_int_list
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


def load_raw_data(exp_name: str, archive_dir: str,
                  var: Union[str, List],
                  year_files: Optional[Union[int, List, str]] = None,
                  lon_min: Optional[float] = None, lon_max: Optional[float] = None,
                  lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                  chunks_time: Optional[int] = None, chunks_lat: Optional[int] = None, chunks_lon: Optional[int] = None,
                  load_parallel: bool = False, logger: Optional[logging.Logger] = None):
    if isinstance(var, str):
        var = [var]
    chunks = {"time": chunks_time, "lat": chunks_lat, "lon": chunks_lon}

    def preprocess_atm(ds):
        # Preprocessing so don't load in entire dataset
        ds = lat_lon_range_slice(ds, lat_min, lat_max, lon_min, lon_max)
        return ds[var]

    ds = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=1, comp='atm',
                           year_files=year_files, chunks=chunks, parallel=load_parallel,
                           preprocess=preprocess_atm, logger=logger)
    logger.info('Loaded data')
    return ds


def main(input_file_path: str):
    logger = logging.getLogger()  # for printing to console time info
    logger.info("Start")
    input_info = f90nml.read(input_file_path)
    script_info = input_info['script_info']
    if script_info['out_dir'] is None:
        script_info['out_dir'] = os.path.dirname(input_file_path)
    out_file = os.path.join(script_info['out_dir'], script_info['out_name'])
    if os.path.exists(out_file):
        if script_info['overwrite']:
            warnings.warn(f'Output file already exists at:\n{out_file}\nWill be overwriten')
        else:
            # Raise error if output data already exists
            raise ValueError('Output file already exists at:\n{}'.format(out_file))

    if isinstance(script_info['var'], str):
        script_info['var'] = [script_info['var']]
    # Var to be loaded must include that which quantile conditioning performed on
    if script_info['var_quant_condition'] not in script_info['var']:
        load_var = script_info['var'] + [script_info['var_quant_condition']]
    else:
        load_var = script_info['var']

    if script_info['av_method'] not in ['mean', 'median']:
        raise ValueError('av_method must be either "mean" or "median" but got {}'.format(script_info['av_method']))

    func_arg_names = inspect.signature(load_raw_data).parameters
    func_args = {k: v for k, v in script_info.items() if k in func_arg_names}
    func_args['var'] = load_var
    ds = load_raw_data(**func_args, logger=logger)
    logger.info(f"Finished lazy-loading datasets | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Fully load data if chose to do that
    if script_info['load_all_at_start']:
        ds.load()
        logger.info(f"Fully loaded data | Memory used {get_memory_usage() / 1000:.1f}GB")

    ds = ds.transpose(..., 'lev')    # ensure lev is the last dimension, required for saving to output_info
    # Initialize dict to save output data - n_surf (land or ocean) x n_lat x n_quant
    # Initialize as nan, so remains as Nan if not enough data to perform calculation
    n_lat = ds.lat.size
    n_lon = ds.lon.size
    n_lev = ds.lev.size
    # find quantile by looking for all values in quantile range between quant_use-quant_range to quant_use+quant_range
    n_quant = len_safe(script_info['quant'])
    output_info = {}
    output_dims = {'use_in_calc': ['quant', 'lat', 'lon', 'time']}            # Coordinate info to convert to xarray datasets
    output_long_name = {}
    output_units = {}
    for var in script_info['var']:
        if hasattr(ds[var], 'lev'):
            output_info[var] = np.full((n_quant, n_lat, n_lon, n_lev), np.nan, dtype=float)
            output_dims[var] = ['quant', 'lat', 'lon', 'lev']
        else:
            output_info[var] = np.full((n_quant, n_lat, n_lon), np.nan, dtype=float)
            output_dims[var] = ['quant', 'lat', 'lon']
        output_long_name[var] = ds[var].long_name if 'long_name' in ds[var].attrs else ''
        output_units[var] = ds[var].units if 'units' in ds[var].attrs else ''

    var_keys = [key for key in output_info.keys()]
    for var in var_keys:
        output_info[var + '_std'] = np.full_like(output_info[var], np.nan, dtype=float)
        output_dims[var + '_std'] = output_dims[var]
    output_info['use_in_calc'] = np.zeros((n_quant, n_lat, n_lon, ds.time.size), dtype=bool)

    coords = {'quant': np.array(script_info['quant'], ndmin=1),
              'lat': ds.lat, 'lon': ds.lon, 'time': ds.time, 'lev': ds.lev}


    logger.info(f"Starting iteration over {n_lat} latitudes, {n_lon} longitudes, and {n_quant} quantiles")

    # Loop through and get quantile info at each latitude and surface
    for i in range(n_lat):
        time_log = {'load': 0, 'calc': 0, 'start': time.time()}
        for k in range(n_lon):
            time_log['start'] = time.time()
            ds_latlon = ds.isel(lat=i, lon=k, drop=True)
            if not script_info['load_all_at_start']:
                ds_latlon.load()
            time_log['load'] += time.time() - time_log['start']
            time_log['start'] = time.time()
            for j in range(n_quant):
                # get indices corresponding to given near-surface temp quantile
                quant_mask = get_quant_ind(ds_latlon[script_info['var_quant_condition']], coords['quant'][j],
                                           script_info['quant_range_below'], script_info['quant_range_above'],
                                           return_mask=True, av_dim='time')
                if quant_mask.sum() == 0:
                    logger.info(
                        f"Latitude {i + 1}/{n_lat} | Longitude {k + 1}/{n_lon}: no data found for quant={coords['quant'][j]}")
                    continue
                ds_use = ds_latlon.where(quant_mask)
                output_info['use_in_calc'][j, i, k] = quant_mask
                for key in var_keys:
                    output_info[key][j, i, k] = getattr(ds_use[key], script_info['av_method'])(dim='time')
                    output_info[key + '_std'][j, i, k] = ds_use[key].std(dim='time')
            time_log['calc'] += time.time() - time_log['start']
            # if (i+1) == 1 or (i+1) == n_lat or (i+1) % 10 == 0:
            # # Log info on 1st, last and every 10th latitude
        logger.info(f"Latitude {i + 1}/{n_lat} | Loading took {time_log['load']:.1f}s |"
                    f" Calculation took {time_log['calc']:.1f}s | "
                    f"Memory used {get_memory_usage() / 1000:.1f}GB")

    # Convert output dict into xarray dataset
    # Convert individual arrays
    for var in output_info:
        output_info[var] = xr.DataArray(output_info[var], dims=output_dims[var],
                                        coords={key: coords[key] for key in output_dims[var]})
        if var == 'use_in_calc':
            continue
        output_info[var] = set_attrs(output_info[var], long_name=output_long_name[var.replace('_std','')],
                                     units=output_units[var.replace('_std','')])
    # Convert dict to dataset
    ds_out = xr.Dataset(output_info)
    ds_out = ds_out.astype('float32')           # save memory

    # Save output to nd2 file with compression - reduces size of file by factor of 10
    # Compression makes saving step slower
    ds_out.to_netcdf(os.path.join(script_info['out_dir'], script_info['out_name']), format="NETCDF4",
                     encoding={var: {"zlib": True, "complevel": script_info['complevel']} for var in ds_out.data_vars})
    logger.info("End")


if __name__ == "__main__":
    main(sys.argv[1])
