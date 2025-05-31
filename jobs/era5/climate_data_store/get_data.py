import cdsapi
import sys
import os
import logging
import f90nml
import numpy as np
import concurrent.futures
import copy
from typing import Optional, List
from isca_tools.utils.base import parse_int_list, split_list_max_n

# Get daily average variable at given pressure level for a given year (one file created per year)
# Use freq=1 hour, which is the frequency to sample the source data
# Variable examples: temperature, geopotential
# Dataset url: https://cds.climate.copernicus.eu/datasets/derived-era5-pressure-levels-daily-statistics

# Set up logging configuration to output to console, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s,%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)

def create_years_per_job_nml(input_file_path: str, exist_ok: Optional[bool] = None) -> List:
    """
    Splits up list of all years into separate lists of no more than `max_workers` in each.
    A `nml` file is then created for each of these with name same as `input_file_path` but with first year
    in job as a suffix e.g. `input_nml` becomes `input1985.nml` with `input_info['request']['year']` set to
    years to run for that job.

    Args:
        input_file_path: Path to `nml` file for experiment.
        exist_ok: If `True`, do not raise exception if any file to be created already exists.
            If `False`, will overwrite it. If `None` leaves the existing file unchanged.

    Returns:
        List of paths to nml files created e.g. `['/Users/.../input1985.nml', '/Users/.../input1985.nml']`
    """
    input_info = f90nml.read(input_file_path)
    script_info = copy.deepcopy(input_info['script_info'])
    request_dict = initialize_request_dict(copy.deepcopy(input_info['request']))
    if not script_info['one_year_per_file']:
        out_file_names = [input_file_path]      # if more than one year per file, run job just once
    else:
        years_all = [int(year) for year in request_dict['year']]
        years_jobs = split_list_max_n(years_all, script_info['max_workers'])
        out_file_names = []
        for years in years_jobs:
            input_info['request']['year'] = years
            out_file_names.append(input_file_path.replace('.nml', f'{years[0]}.nml'))
            if os.path.exists(out_file_names[-1]):
                if exist_ok is None:
                    print(f'{years}: Output nml file already exists. Leaving unchanged')
                    continue
            input_info.write(out_file_names[-1], force=exist_ok)
            print(f'{years}: Output nml file created')
    return out_file_names


def initialize_request_dict(request_dict: dict) -> dict:
    """
    Takes request_dict as it is in nml file, and converts each key into expected format for cdsapi client.

    Args:
        request_dict: Request dictionary from nml file, required to get chosen data.

    Returns:
        request_dict with numbers set to strings, and time info changed to correct format.
    """
    # If months, days or time left blank then set to include all values
    if request_dict['month'] is None:
        request_dict['month'] = np.arange(1, 13).tolist()
    if request_dict['day'] is None:
        request_dict['day'] = np.arange(1, 32).tolist()
    if 'time' in request_dict:
        if request_dict['time'] is None:
            request_dict['time'] = np.arange(24).tolist()

    # Re-reformat time lists so correct format
    format_time = {'month': lambda x: f"{x:02d}", 'day': lambda x: f"{x:02d}", 'year': lambda x: str(x),
                   'time': lambda x: f"{x:02d}:00"}
    for key in format_time:
        if key not in request_dict:
            continue
        request_dict[key] = parse_int_list(request_dict[key], format_time[key])

    # Convert any numbers into strings
    for key in request_dict:
        if isinstance(request_dict[key], (int, float)):
            request_dict[key] = str(request_dict[key])
    return request_dict


def download_data(out_path: str, dataset: str, request_dict: dict, exist_ok: bool,
                  logger: Optional[logging.Logger] = None) -> None:
    """
    Function to run climate data store application programming interface (cdsapi) to download the ERA5 data
    requested in the `request_dict` dictionary.

    Args:
        out_path: Where to save the downloaded data.
        dataset: ERA5 dataset name e.g. `'derived-era5-pressure-levels-daily-statistics'`
        request_dict: Dictionary containing request information e.g. `year`, `month`, `day`, `pressure_level`.
        exist_ok: If output already exists, overwrite if `True` otherwise give error.
        logger: If provide logger, will log the time when download starts

    Returns:

    """
    if not exist_ok:
        if os.path.exists(out_path):
            raise ValueError(f"File '{out_path}' already exists. Run with exist_ok=True to overwrite.")
    if logger is not None:
        logger.info(f"Start year: {request_dict['year']}")
    c = cdsapi.Client()
    c.retrieve(dataset, request_dict, out_path)

def main(input_file_path: str) -> None:
    input_info = f90nml.read(input_file_path)
    script_info = input_info['script_info']
    request_dict = initialize_request_dict(input_info['request'])
    logger = logging.getLogger()  # for printing to console time info
    if not os.path.exists(script_info['out_dir']):
        os.makedirs(script_info['out_dir'])
        logger.info(f"Directory '{script_info['out_dir']}' created.")
    if not script_info['one_year_per_file']:
        # Combine all years in a single file
        logger.info(f"Downloading all {len(request_dict['year'])} years in single file")
        out_file_name = 'all_years.nc'
        try:
            download_data(os.path.join(script_info['out_dir'], out_file_name), script_info['dataset'], request_dict,
                          script_info['exist_ok'])
        except Exception as e:
            logger.info(f"Error: {e}")
        logger.info("End")
    else:
        # Get data for each year in turn
        years_all = request_dict['year']
        def download_one_year(year):
            # ensure a separate dictionary for each year, with year set to that particular year
            request_dict_use = copy.deepcopy(request_dict)
            request_dict_use['year'] = year
            try:
                download_data(os.path.join(script_info['out_dir'], f'{year}.nc'), script_info['dataset'],
                              request_dict_use, script_info['exist_ok'], logger)
            except Exception as e:
                logger.info(f"Error for year {year}: {e}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=script_info['max_workers']) as executor:
            futures = {executor.submit(download_one_year, year): year for year in years_all}
            for future in concurrent.futures.as_completed(futures):
                year = futures[future]
                logger.info(f"End year: {year}")


if __name__ == '__main__':
    main(sys.argv[1])
