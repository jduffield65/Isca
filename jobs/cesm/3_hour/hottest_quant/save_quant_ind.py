# Script to save the time index of the hottest quant % of days
# One value for each `sample` in the hottest % of days. Only 1 time for each day corresponding to hottest time of day
from geocat.comp import interp_hybrid_to_pressure
from isca_tools import cesm
from isca_tools.utils.ds_slicing import lat_lon_range_slice
from isca_tools.utils import get_memory_usage
from isca_tools.utils.base import top_n_peaks_ind, parse_int_list
from isca_tools.thesis.lapse_theory import interp_var_at_pressure
import sys
import os
import numpy as np
import f90nml
import warnings
import logging
import inspect
from typing import Optional, Union, List
import xarray as xr
import gc  # garbage collector

# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


def load_raw_data(exp_name: str, archive_dir: str,
                  var: Union[str, List], var_lev = Optional[int],
                  year_files: Optional[Union[int, List, str]] = None,
                  month_files: Optional[Union[int, List, str]] = None,
                  lon_min: Optional[float] = None, lon_max: Optional[float] = None,
                  lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                  lat_ind: Optional[Union[int, List, str]] = None, lon_ind: Optional[Union[int, List, str]] = None,
                  chunks_time: Optional[int] = None, chunks_lat: Optional[int] = None, chunks_lon: Optional[int] = None,
                  load_parallel: bool = False, logger: Optional[logging.Logger] = None):
    if isinstance(var, str):
        var = [var]
    chunks = {"time": chunks_time, "lat": chunks_lat, "lon": chunks_lon}

    def preprocess_atm(ds):
        # Preprocessing so don't load in entire dataset
        # Probably want to use lat_ind or lat_max. But ind computed first as indices are relative to entire dataset
        if lat_ind is not None:
            ds = ds.isel(lat=parse_int_list(lat_ind, format_func=lambda x: int(x), all_values=np.arange(ds.lat.size)))
        if lon_ind is not None:
            ds = ds.isel(lon=parse_int_list(lon_ind, format_func=lambda x: int(x), all_values=np.arange(ds.lon.size)))
        ds = lat_lon_range_slice(ds, lat_min, lat_max, lon_min, lon_max)
        ds = ds[var]
        if var_lev is not None:
            ds = ds.isel(lev=var_lev)
        return ds

    ds = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=1, comp='atm',
                           year_files=year_files, month_files=month_files, chunks=chunks, parallel=load_parallel,
                           preprocess=preprocess_atm, logger=logger)
    logger.info('Loaded data')
    return ds


def process_ds(T_var: xr.DataArray, n_sample: int, ind_spacing: int,
               load_all_at_start: bool = True, ds_name: str = 'All Data', logger: Optional[logging.Logger] = None):
    logger.info(f"{ds_name} | Started | Memory used {get_memory_usage() / 1000:.1f}GB")

    if load_all_at_start:
        T_var = T_var.load()

    # Compute indices of top N peaks
    idx_quant = xr.apply_ufunc(
        top_n_peaks_ind,
        T_var,
        input_core_dims=[["time"]],
        output_core_dims=[["sample"]],
        vectorize=True,
        kwargs={"n": n_sample, "min_ind_spacing": ind_spacing},
        dask="parallelized",
        output_dtypes=[int],
    )
    idx_quant = idx_quant.assign_coords(sample=np.arange(1, n_sample + 1))
    logger.info(
        f"{ds_name} | Computed hottest {n_sample} indices | Memory used {get_memory_usage() / 1000:.1f}GB")

    logger.info(f"{ds_name} | Finished | Memory used {get_memory_usage() / 1000:.1f}GB")
    return idx_quant


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

    func_arg_names = inspect.signature(load_raw_data).parameters
    func_args = {k: v for k, v in script_info.items() if k in func_arg_names}
    func_args['var_lev'] = script_info['refht_level_index']
    var_name = 'TREFHT' if script_info['refht_level_index'] is None else 'T'
    func_args['var'] = [var_name]
    ds = load_raw_data(**func_args, logger=logger)
    logger.info(f"Finished lazy-loading datasets | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Compute temporal spacing and sampling parameters
    time_dt = (ds.time[1] - ds.time[0]) / np.timedelta64(1, 'h')
    ind_spacing = int(np.ceil(script_info['hour_spacing'] / time_dt))

    n_sample = int(np.ceil((1 - script_info['quant'] / 100)
                           * np.unique(ds.time.dt.floor('D')).size))

    n_lat = ds.lat.size
    # Prepare list to store results for each latitude
    if script_info['loop_over_lat']:
        lat_results = []
        for i, lat_val in enumerate(ds.lat.values):
            ds_out = process_ds(ds[var_name].isel(lat=i), n_sample, ind_spacing, script_info['load_all_at_start'],
                                f'Lat {i}/{n_lat}', logger)
            ds_out = ds_out.expand_dims(lat=[lat_val])  # Add latitude coordinate (needed after recombining)
            lat_results.append(ds_out)
        # Concatenate along latitude
        idx_quant = xr.concat(lat_results, dim='lat')
        logger.info(f"Recombined all latitudes | Final memory used {get_memory_usage() / 1000:.1f}GB")
    else:
        idx_quant = process_ds(ds[var_name], n_sample, ind_spacing, script_info['load_all_at_start'],
                            f'Lat {ds.lat[0]:.1f} to {ds.lat[-1]:.1f}', logger)

    # Only keep TREFHT corresponding to time_ind, but save initial time coordinate as well
    ds['time_ind'] = idx_quant
    time_old = ds.time
    ds = ds.isel(time=idx_quant)
    ds = ds.drop_vars("time")
    ds = ds.assign_coords(time=time_old)
    logger.info(f"Created ds_out | Memory used {get_memory_usage() / 1000:.1f}GB")
    ds.to_netcdf(os.path.join(script_info['out_dir'], script_info['out_name']), format="NETCDF4",
                 encoding={var: {"zlib": True, "complevel": script_info['complevel']} for var in ds.data_vars})
    logger.info('End')


if __name__ == "__main__":
    main(sys.argv[1])
