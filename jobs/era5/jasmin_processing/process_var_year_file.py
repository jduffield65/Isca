# Script to process a single variable, and save the processed variable as one file for each year
import f90nml
from get_jasmin_era5 import Find_era5
from isca_tools.utils.base import split_list_max_n
from typing import Literal, List, Optional, Union
import xarray as xr
import os
import numpy as np
import logging
import inspect
import sys
import copy
from isca_tools.utils.base import parse_int_list, get_memory_usage, split_list_max_n
# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

def create_years_per_job_nml(input_file_path: str, years_per_job: int, exist_ok: Optional[bool] = None) -> List:
    """
    Splits up list of all years into separate lists of no more than `years_per_job` in each.
    A `nml` file is then created for each of these with name same as `input_file_path` but with first year
    in job as a suffix e.g. `input_nml` becomes `input1985.nml` with `input_info['script_info']['years']` set to
    years to run for that job.

    Args:
        input_file_path: Path to `nml` file for experiment.
        years_per_job: Numbr of years to run for each job.
        exist_ok: If `True`, do not raise exception if any file to be created already exists.
            If `False`, will overwrite it. If `None` leaves the existing file unchanged.

    Returns:
        List of paths to nml files created e.g. `['/Users/.../input1985.nml', '/Users/.../input1985.nml']`
    """
    input_info = f90nml.read(input_file_path)
    years_all = parse_int_list(input_info['script_info']['years'], lambda x: int(x))
    years_jobs = split_list_max_n(years_all, years_per_job)
    if not os.path.exists(input_info['script_info']['out_dir']):
        # Create directory here, so don't have issue where two slurm scripts create it simultaneously
        os.makedirs(input_info['script_info']['out_dir'])
        print(f"Directory '{input_info['script_info']['out_dir']}' created.")
    out_file_names = []
    for years in years_jobs:
        input_info['script_info']['years'] = years
        out_file_names.append(input_file_path.replace('.nml', f'{years[0]}.nml'))
        if os.path.exists(out_file_names[-1]):
            if exist_ok is None:
                print(f'{years}: Output nml file already exists. Leaving unchanged')
                continue
        input_info.write(out_file_names[-1], force=exist_ok)
        print(f'{years}: Output nml file created')
    return out_file_names


def process_year(out_file: str, var: str, year: int, stat: Literal['mean', 'min', 'max'], stat_freq: str = '1D',
                 months: Optional[Union[str,List]] = None, months_at_a_time: int = 1, level: Optional[int] = None,
                 lon_min: Optional[float] = None, lon_max: Optional[float] = None, lat_min: Optional[float] = None,
                 lat_max: Optional[float] = None, model: Literal["oper", "enda"] = "oper",
                 load_all_at_start: bool = False, exist_ok: bool = False, logger: Optional[logging.Logger] = None):
    if os.path.exists(out_file):
        if exist_ok:
            if logger is not None:
                logger.info(f'Year {year} - Output file already exists, will be overwritten')
        else:
            raise ValueError(f'Output file {out_file} already exists')
    if logger is not None:
        logger.info(f'Year {year} - Start')
    era5 = Find_era5()
    # era5.enda to use ensemble version
    if months is None:
        months = np.arange(1, 13)
    else:
        months = parse_int_list(months, lambda x: int(x))
    months_chunk = split_list_max_n(months, months_at_a_time)
    n_chunks = len(months_chunk)
    time_start_chunk = [f"{year}-{i[0]:0d}-01" for i in months_chunk]
    time_end_chunk = time_start_chunk[1:]           # chunk ends just before first day of next chunk
    time_end_chunk.append(f"{year + 1}-01-01")      # last chunk ends just before 1st day of next year

    if (level is not None) and (level < 0):
        level = 138 + level     # highest level is 137

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

    out_chunks = []
    for i in range(n_chunks):
        # era5 object takes the following arguments: era5[var, time_range, level, lon_range, lat_range, model]
        var_in = era5[var, time_start_chunk[i]:time_end_chunk[i], level, lon_range, lat_range, model]
        if logger is not None:
            logger.info(f'Year {year} - Chunk {i + 1}/{n_chunks} lazy loaded | '
                        f'Memory used {get_memory_usage() / 1000:.1f}GB')
        if load_all_at_start:
            var_in.load()
            if logger is not None:
                logger.info(f'Year {year} - Chunk {i + 1}/{n_chunks} fully loaded | '
                            f'Memory used {get_memory_usage() / 1000:.1f}GB')
        var_out = getattr(var_in.resample(time=stat_freq), stat)(dim='time')
        # var_out = var_in.resample(time='1D').mean(dim='time')
        out_chunks.append(var_out)
        if logger is not None:
            logger.info(f'Year {year} - Chunk {i+1}/{n_chunks} complete | Memory used {get_memory_usage()/1000:.1f}GB')

    # Concatenate all days for the year and save
    full_year = xr.concat(out_chunks, dim='time')
    full_year.to_netcdf(out_file)

    if logger is not None:
        logger.info(f'Year {year} - End')

def main(input_file_path: str):
    # Run processing for all required years
    input_info = f90nml.read(input_file_path)
    script_info = input_info['script_info']
    years_all = parse_int_list(script_info['years'], lambda x: int(x))
    func_arg_names = inspect.signature(process_year).parameters
    func_args = {k: v for k, v in script_info.items() if k in func_arg_names}
    logger = logging.getLogger()  # for printing to console time info
    if not os.path.exists(script_info['out_dir']):
        os.makedirs(script_info['out_dir'])
        logger.info(f"Directory '{script_info['out_dir']}' created.")
    for year in years_all:
        process_year(**func_args, out_file=os.path.join(script_info['out_dir'], f"{year}.nc"),
                     year=year, logger=logger)

if __name__ == '__main__':
    main(sys.argv[1])
