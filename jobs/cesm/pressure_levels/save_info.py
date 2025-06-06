from typing import Optional, Union, List
import numpy as np
import logging
import os
import xarray as xr
from isca_tools import cesm
from isca_tools.utils.base import parse_int_list, get_memory_usage, split_list_max_n, print_log, run_func_loop
from isca_tools.utils import set_attrs
from geocat.comp.interpolation import interp_hybrid_to_pressure
import sys
import f90nml
import inspect
# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


def ds_lat_lon_slice(ds, lat_min, lat_max, lon_min, lon_max):
    if (lon_min is None) and lon_max is None:
        lon_range = None
    else:
        if lon_max is None:
            raise ValueError('lon_max is required')
        if lon_min is None:
            raise ValueError('lon_min is required')
        lon_range = slice(lon_min, lon_max)

    if (lat_min is None) and (lat_max is None):
        lat_range = None
    else:
        if lat_max is None:
            raise ValueError('lat_max is required')
        if lat_min is None:
            raise ValueError('lat_min is required')
        lat_range = slice(lat_min, lat_max)

    if lat_range is not None:
        ds = ds.sel(lat=lat_range)
    if lon_range is not None:
        ds = ds.sel(lon=lon_range)
    return ds


def process_year(exp_name, archive_dir, out_dir: str, var: Union[str, List],
                 pressure_levels: Union[float, List, np.ndarray], year: int, hyam: xr.DataArray, hybm: xr.DataArray,
                 p0: float, hist_file: int = 0, lon_min: Optional[float] = None, lon_max: Optional[float] = None,
                 lat_min: Optional[float] = None,
                 lat_max: Optional[float] = None,
                 load_all_at_start: bool = True, overwrite: Optional[bool] = None, wait_interval: int = 20,
                 max_wait_time: int = 360, complevel: int = 4, logger: Optional[logging.Logger] = None) -> None:
    out_file = os.path.join(out_dir, f"{year:04d}.nc")
    if os.path.exists(out_file):
        if overwrite is None:
            print_log(f'Year {year} - Output file already exists, skipping to next year.', logger)
            # If None, and file exist skip
            return None
        elif overwrite:
            print_log(f'Year {year} - Output file already exists, will be overwritten', logger)
        else:
            raise ValueError(f'Output file {out_file} already exists')
    print_log(f'Year {year} - Start', logger)

    # Make sure var and pressure levels in correct form
    if isinstance(var, str):
        var = [var]
    if 'PS' not in var:
        var.append('PS')        # need surface pressure as well
    pressure_levels = np.array(pressure_levels, ndmin=1) * 100  # convert to Pa from hPa

    # Load dataset for given year
    ds = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=hist_file, comp='atm',
                           year_files=year, logger=logger)[var]
    ds = ds_lat_lon_slice(ds, lat_min, lat_max, lon_min, lon_max)

    print_log(f'Year {year} - lazy loaded | Memory used {get_memory_usage() / 1000:.1f}GB', logger)
    if load_all_at_start:
        ds.load()
        print_log(f'Year {year} - fully loaded | Memory used {get_memory_usage() / 1000:.1f}GB', logger)

    # Loop over all variables, do interpolation onto required pressure levels
    out_dict = {}
    var.remove('PS')
    n_var = len(var)
    for i, key in enumerate(var):
        out_dict[key] = interp_hybrid_to_pressure(ds[key], ds['PS'], hyam, hybm, p0, pressure_levels)
        set_attrs(out_dict[key].plev, long_name='pressure', units='Pa')
        print_log(f'Year {year}| Variable {i+1}/{n_var} complete: {key} | '
                    f'Memory used {get_memory_usage() / 1000:.1f}GB', logger)

    # Save data
    ds_out = xr.Dataset(out_dict)
    encoding = {var: {'zlib': True, 'complevel': complevel} for var in ds_out.data_vars}

    func_save = lambda: ds_out.to_netcdf(out_file, encoding=encoding)
    run_func_loop(func_save, max_wait_time=max_wait_time, wait_interval=wait_interval, logger=logger)
    return None


def main(input_file_path: str):
    # Run processing for all required years
    input_info = f90nml.read(input_file_path)
    script_info = input_info['script_info']

    # Determine which years to get data for
    file_dates = cesm.get_exp_file_dates(script_info['exp_name'], 'atm', script_info['archive_dir'],
                                         script_info['hist_file'])
    year_files_all = np.unique(file_dates.dt.year).tolist()
    if script_info['year_files'] is None:
        years_consider = year_files_all
    else:
        years_consider = parse_int_list(script_info['year_files'], lambda x: int(x), all_values=year_files_all)

    if script_info['out_dir'] is None:
        # Set out directory to experiment folder with directory name `out_name` if don't provide full path
        script_info['out_dir'] = os.path.join(script_info['archive_dir'], script_info['exp_name'], script_info['out_name'])

    func_arg_names = inspect.signature(process_year).parameters
    func_args = {k: v for k, v in script_info.items() if k in func_arg_names}

    logger = logging.getLogger()  # for printing to console time info
    logger.info(f'Years {years_consider} | Start')
    logger.info(f'Loading reference P0, hyam and hybm from Year {years_consider[0]} | Start')
    ds_ref = cesm.load_dataset(script_info['exp_name'], 'atm', script_info['archive_dir'],
                               script_info['hist_file'], year_files=years_consider[0])[['P0', 'hyam', 'hybm']]
    ds_ref = ds_ref.isel(time=0).load()
    logger.info(f'Loading reference P0, hyam and hybm from Year {years_consider[0]} | End')

    if not os.path.exists(script_info['out_dir']):
        # While loop to stop mutliple processes making directory at same time
        run_func_loop(func=lambda: os.makedirs(script_info['out_dir']),
                      func_check=lambda: os.path.exists(script_info['out_dir']),
                      max_wait_time=script_info['max_wait_time'], wait_interval=script_info['wait_interval'])
    for year in years_consider:
        try:
            process_year(**func_args, year=year, logger=logger, hyam=ds_ref.hyam, hybm=ds_ref.hybm,
                         p0=float(ds_ref['P0']))
        except Exception as e:
            # If error just go to next year
            logger.info(f'Year {year} | FAILED | {e} ')
            continue
    logger.info(f'Years {years_consider} | End')
if __name__ == '__main__':
    main(sys.argv[1])
