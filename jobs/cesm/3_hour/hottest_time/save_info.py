# Script to find time of maximum TREFHT, then save variables conditioned on this time
from isca_tools import cesm
from isca_tools.utils.ds_slicing import lat_lon_range_slice
from isca_tools.utils import get_memory_usage
from isca_tools.thesis.lapse_theory import interp_var_at_pressure
import sys
import os
import numpy as np
import f90nml
import warnings
import logging
import inspect
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

    # Fully load data if chose to do that
    if script_info['load_all_at_start']:
        ds[script_info['var']].load()       # remove gw, hyam, hybm, p0 from ds
        logger.info(f"Fully loaded data | Memory used {get_memory_usage() / 1000:.1f}GB")

    if script_info['temp_ft_plev'] is not None:
        temp_ft_zonal_daily_av = interp_var_at_pressure(ds.T.resample(time="1D").mean(dim="time").mean(dim='lon'),
                                                     np.atleast_1d(script_info['temp_ft_plev']),
                                                     ds.PS.resample(time="1D").mean(dim="time").mean(dim='lon'), hyam,
                                                     hybm, p0).T
        logger.info(f"Computed daily zonal average T500 | Memory used {get_memory_usage() / 1000:.1f}GB")
        temp_ft_zonal_daily_av = temp_ft_zonal_daily_av.load()
        logger.info(f"Loaded daily zonal average T500 | Memory used {get_memory_usage() / 1000:.1f}GB")

    idx_max = ds['TREFHT'].argmax(dim='time')
    ds_out = ds.isel(time=idx_max)
    ds_out['time_max'] = ds.time.isel(time=idx_max)
    if script_info['temp_ft_plev'] is not None:
        ds_out['T500_zonal_daily_av'] = temp_ft_zonal_daily_av
    ds_out['gw'] = gw
    ds_out['hyam'] = hyam
    ds_out['hybm'] = hybm
    ds_out['P0'] = p0
    logger.info(f"Created ds_out | Memory used {get_memory_usage() / 1000:.1f}GB")
    ds_out.to_netcdf(os.path.join(script_info['out_dir'], script_info['out_name']), format="NETCDF4",
                     encoding={var: {"zlib": True, "complevel": script_info['complevel']} for var in ds_out.data_vars})
    logger.info('End')


if __name__ == "__main__":
    main(sys.argv[1])
