## Script to save variables conditioned on times of hottest % of days
# One value for each `sample` in the hottest % of days. Only 1 time for each day corresponding to hottest time of day
# Loop over raw input files, so only load one at a time to save time
# One output file for each raw input - each file is sam size but mainly nans, only contains `sample` for which
# that raw file contains, but when concatenate, get full dataset
from geocat.comp import interp_hybrid_to_pressure
from isca_tools import cesm
from isca_tools.utils.ds_slicing import lat_lon_range_slice, get_time_sample_indices
from isca_tools.utils import get_memory_usage
from isca_tools.utils.base import top_n_peaks_ind, parse_int_list
import sys
import os
import numpy as np
import f90nml
import warnings
import logging
import inspect
from typing import Optional, Union, List, Callable
import xarray as xr
import gc  # garbage collector

# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


def load_raw_data(exp_name: str, archive_dir: str,
                  var: Union[str, List],
                  ind_files: Optional[Union[int, List, str]] = None,
                  lon_min: Optional[float] = None, lon_max: Optional[float] = None,
                  lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                  lat_ind: Optional[Union[int, List, str]] = None, lon_ind: Optional[Union[int, List, str]] = None,
                  chunks_time: Optional[int] = None, chunks_lat: Optional[int] = None, chunks_lon: Optional[int] = None,
                  hist_file: int = 0,
                  load_parallel: bool = False, logger: Optional[logging.Logger] = None):
    if isinstance(var, str):
        var = [var]
    chunks = {"time": chunks_time, "lat": chunks_lat, "lon": chunks_lon}

    def preprocess_atm(ds):
        # Preprocessing so don't load in entire dataset
        # Probably want to use lat_ind or lat_max. But ind computed first as indices are relative to entire dataset
        if lat_ind is not None:
            ds = ds.isel(lat=parse_int_list(lat_ind, format_func=lambda x: int(x), all_values=np.arange(ds.lat.size).tolist()))
        if lon_ind is not None:
            ds = ds.isel(lon=parse_int_list(lon_ind, format_func=lambda x: int(x), all_values=np.arange(ds.lon.size).tolist()))
        ds = lat_lon_range_slice(ds, lat_min, lat_max, lon_min, lon_max)
        return ds[var]

    ds = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=hist_file, comp='atm',
                           ind_files=ind_files, chunks=chunks, parallel=load_parallel,
                           preprocess=preprocess_atm, logger=logger)
    logger.info('Loaded data')
    return ds


def process_ds(ds: xr.Dataset, idx_quant: xr.DataArray, hyam: xr.DataArray, hybm: xr.DataArray, p0: float,
               load_all_at_start: bool = True, temp_ft_plev: Optional[List] = None,
               ds_name: str = 'All Data', logger: Optional[logging.Logger] = None):
    logger.info(f"{ds_name} | Started | Memory used {get_memory_usage() / 1000:.1f}GB")

    if load_all_at_start:
        ds = ds.load()
        logger.info(f"{ds_name} | Fully loaded | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Optional: compute temp_ft_zonal_av for this latitude
    if temp_ft_plev is not None:
        # TODO: at the moment, this does not work. No variable named T_zonal_av added to ds_out
        temp_ft_lat = interp_hybrid_to_pressure(ds.T, ds.PS, hyam, hybm, p0,
                                                np.atleast_1d(temp_ft_plev),
                                                lev_dim='lev')

        temp_ft_lat = temp_ft_lat.load()

        temp_ft_zonal_av = temp_ft_lat.mean(dim='lon')
        # del temp_ft_lat  # release memory early
        logger.info(f"{ds_name} | Computed T_FT | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Select those times and store in temporary dataset
    idx_quant = idx_quant.fillna(-1).astype(int)
    ds_out = ds.isel(time=idx_quant)
    ds_out = ds_out.where(idx_quant >= 0)
    ds_out['time_ind'] = idx_quant

    if temp_ft_plev is not None:
        ds_out['T_zonal_av'] = temp_ft_zonal_av

    # Explicitly delete large intermediates
    # del ds_lat, ds_out_lat, idx_quant, T_var

    logger.info(f"{ds_name} | Finished | Memory used {get_memory_usage() / 1000:.1f}GB")
    return ds_out


def main_one_file(script_info: dict, ind_file: int, out_name_func: Callable, logger: Optional[logging.Logger] = None):
    # out_name_func is a function that takes in the default out_name and ind_file; returning new out_name
    # ind_file refers to raw input data - only load one at a time
    # Use combine_file_output.ipynb to combine to single output file
    script_info['ind_files'] = ind_file
    script_info['out_name'] = out_name_func(script_info['out_name'], ind_file)      # include file index in output name
    out_file = os.path.join(script_info['out_dir'], script_info['out_name'])
    if os.path.exists(out_file):
        if script_info['overwrite']:
            warnings.warn(f'Output file already exists at:\n{out_file}\nWill be overwriten')
        else:
            # Raise error if output data already exists
            raise ValueError('Output file already exists at:\n{}'.format(out_file))

    if isinstance(script_info['var'], str):
        script_info['var'] = [script_info['var']]
    load_var = script_info['var'] + ['gw', 'hyam', 'hybm', 'P0']

    func_arg_names = inspect.signature(load_raw_data).parameters
    func_args = {k: v for k, v in script_info.items() if k in func_arg_names}
    func_args['var'] = load_var
    ds = load_raw_data(**func_args, logger=logger)
    logger.info(f"Finished lazy-loading datasets | Memory used {get_memory_usage() / 1000:.1f}GB")

    gw = ds.gw.isel(time=0).load()
    hyam = ds.hyam.isel(time=0).load()
    hybm = ds.hybm.isel(time=0).load()
    p0 = float(ds.P0.isel(time=0))

    quant_times = xr.open_dataset(os.path.join(script_info['quant_time_dir'], script_info['quant_time_name'])
                                 )
    quant_times = quant_times.time.isel(time=quant_times.time_ind)
    quant_times = quant_times.load()
    quant_times = quant_times.reindex_like(ds)    # so only lat and lon in ds kept
    idx_quant = get_time_sample_indices(quant_times, ds.time)

    n_lat = ds.lat.size
    print(script_info['temp_ft_plev'])
    # Prepare list to store results for each latitude
    if script_info['loop_over_lat']:
        lat_results = []
        for i, lat_val in enumerate(ds.lat.values):
            ds_out = process_ds(ds.isel(lat=i), idx_quant.isel(lat=i), hyam, hybm, p0, script_info['load_all_at_start'],
                                script_info['temp_ft_plev'],f'Lat {i}/{n_lat}', logger)
            ds_out = ds_out.expand_dims(lat=[lat_val])  # Add latitude coordinate (needed after recombining)
            lat_results.append(ds_out)
        # Concatenate along latitude
        ds_out = xr.concat(lat_results, dim='lat')
        logger.info(f"Recombined all latitudes | Final memory used {get_memory_usage() / 1000:.1f}GB")
    else:
        ds_out = process_ds(ds, idx_quant, hyam, hybm, p0, script_info['load_all_at_start'],
                            script_info['temp_ft_plev'],
                            f'Lat {ds.lat[0]:.1f} to {ds.lat[-1]:.1f}', logger)
    print(ds_out.T_zonal_av)
    ds_out['gw'] = gw
    ds_out['hyam'] = hyam
    ds_out['hybm'] = hybm
    ds_out['P0'] = p0
    print(-1, ds_out.T_zonal_av)
    ds_out['time'] = quant_times            # replace time with times of the samples
    print(0, ds_out.T_zonal_av)
    ds_out = ds_out.drop_dims("time")       # drop the dimension of time
    print(1, ds_out.T_zonal_av)
    ds_out = ds_out.reset_coords("time")    # convert time from coordinate to variable
    print(2, ds_out.T_zonal_av)
    logger.info(f"Created ds_out | Memory used {get_memory_usage() / 1000:.1f}GB")
    ds_out.to_netcdf(out_file, format="NETCDF4",
                     encoding={var: {"zlib": True, "complevel": script_info['complevel']} for var in ds_out.data_vars})

def main(input_file_path: str):
    input_info = f90nml.read(input_file_path)
    script_info = input_info['script_info']
    if script_info['out_dir'] is None:
        script_info['out_dir'] = os.path.dirname(input_file_path)
    if script_info['quant_time_dir'] is None:
        script_info['quant_time_dir'] = script_info['out_dir']

    # Determine which files to get data for
    file_dates = cesm.get_exp_file_dates(script_info['exp_name'], 'atm', script_info['archive_dir'],
                                         1)
    all_file_ind = np.arange(len(file_dates)).tolist()
    if script_info['ind_files'] is None:
        files_ind_consider = all_file_ind
    else:
        files_ind_consider = parse_int_list(script_info['ind_files'], lambda x: int(x), all_values=all_file_ind)

    out_name_start = script_info['out_name']
    # Get function to modify out_name for each file
    n_digits = len(str(len(all_file_ind) - 1))
    out_name_func = lambda x, y: x.replace('.nc', f'{y:0{n_digits}d}.nc')
    for i, ind in enumerate(files_ind_consider):
        logger = logging.getLogger()  # for printing to console time info
        logger.info(f"Start | File {i + 1}/{len(files_ind_consider)} | {str(file_dates.values[ind])}")
        main_one_file(script_info, ind, out_name_func, logger)
        logger.info(f"End | File {i + 1}/{len(files_ind_consider)} | {str(file_dates.values[ind])}")
        script_info['out_name'] = out_name_start        # so don't repeatedly append number to the name

if __name__ == "__main__":
    main(sys.argv[1])