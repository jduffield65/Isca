from typing import Optional, Union, List
import numpy as np
import logging
import os
import xarray as xr
from isca_tools import cesm
from isca_tools.utils.base import parse_int_list, get_memory_usage, print_log, run_func_loop, round_any
from isca_tools.utils import set_attrs
from isca_tools.convection.base import lcl_metpy
from isca_tools.utils.constants import lapse_dry
from geocat.comp.interpolation import interp_hybrid_to_pressure
import sys
import f90nml
import inspect
from isca_tools.utils.ds_slicing import lat_lon_range_slice

# Set up logging configuration to output to console and don't output milliseconds, and stdout so saved to out file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


def process_year(exp_name, archive_dir, out_dir: str,
                 surf_geopotential_file: str, year: int, hyam: xr.DataArray, hybm: xr.DataArray,
                 p0: float, plev_step: float, extrapolate: bool = True, hist_file: int = 0,
                 lon_min: Optional[float] = None, lon_max: Optional[float] = None,
                 lat_min: Optional[float] = None, lat_max: Optional[float] = None,
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

    # Load dataset for given year
    var = ['PS', 'TREFHT', 'QREFHT', 'T', 'Z3']
    ds = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=hist_file, comp='atm',
                           year_files=year, logger=logger)[var]
    z_2m = cesm.load.load_z2m(surf_geopotential_file, var_reindex_like=ds['PS'])

    ds = lat_lon_range_slice(ds, lat_min, lat_max, lon_min, lon_max)
    z_2m = lat_lon_range_slice(z_2m, lat_min, lat_max, lon_min, lon_max)

    print_log(f'Year {year} - lazy loaded | Memory used {get_memory_usage() / 1000:.1f}GB', logger)
    if load_all_at_start:
        ds.load()
        z_2m.load()
        print_log(f'Year {year} - fully loaded | Memory used {get_memory_usage() / 1000:.1f}GB', logger)

    # Compute LCL
    p_lcl, T_lcl = lcl_metpy(ds.TREFHT, ds.QREFHT, ds.PS)
    Z3_lcl = z_2m + (ds.TREFHT - T_lcl) / lapse_dry
    ds = ds.drop_vars(['TREFHT', 'QREFHT'])         # drop variables no longer need
    del z_2m
    p_lcl = set_attrs(p_lcl, long_name='Pressure of LCL', units='Pa')
    T_lcl = set_attrs(T_lcl, long_name='Temperature of LCL', units='K')
    Z3_lcl = set_attrs(Z3_lcl, long_name='Geopotential height of LCL', units='m')

    plevs = np.arange(round_any(float(p_lcl.min()), plev_step, 'floor'),
                      round_any(float(p_lcl.max()), plev_step, 'ceil')+plev_step/2, plev_step)
    plevs_expand = xr.DataArray(
        plevs,
        dims=["plev"],
        coords={"plev": np.arange(len(plevs))}
    ).expand_dims(time=ds.time, lat=ds.lat, lon=ds.lon).transpose("time", "plev", "lat", "lon")
    idx_lcl_closest = np.abs(plevs_expand - p_lcl).argmin(dim='plev')
    p_at_lcl = plevs_expand.isel(plev=idx_lcl_closest)          # approx pressure of LCL used

    # Note that with extrapolate, will obtain values lower than surface
    T_at_plevs = interp_hybrid_to_pressure(data=ds['T'], ps=ds['PS'], hyam=hyam, hybm=hybm, p0=p0, new_levels=plevs,
                                           extrapolate=extrapolate, variable='other' if extrapolate else None)
    Z3_at_plevs = interp_hybrid_to_pressure(data=ds['Z3'], ps=ds['PS'], hyam=hyam, hybm=hybm, p0=p0, new_levels=plevs,
                                            extrapolate=extrapolate, variable='other' if extrapolate else None)
    # TODO: Should set T and Z3 below surface to TREFHT and z2m respectively i.e. clipping Or extrapolate=False, so will set to nan
    del ds          # save memory by deleting dataset as no longer needed
    if load_all_at_start:
        T_at_plevs.load()
        print_log(f'Year {year} | T on pressure-grid fully loaded | Memory used {get_memory_usage() / 1000:.1f}GB',
                  logger)
        Z3_at_plevs.load()
        print_log(f'Year {year} | Z on pressure-grid fully loaded | Memory used {get_memory_usage() / 1000:.1f}GB',
                  logger)

    T_at_lcl = T_at_plevs.isel(plev=idx_lcl_closest)
    print_log(f'Year {year} | T at LCL computed | Memory used {get_memory_usage() / 1000:.1f}GB', logger)
    Z3_at_lcl = Z3_at_plevs.isel(plev=idx_lcl_closest)
    print_log(f'Year {year} | Z at LCL computed | Memory used {get_memory_usage() / 1000:.1f}GB', logger)

    p_at_lcl = set_attrs(p_at_lcl, long_name='Approx pressure of LCL', units='Pa',
                         description='This is the pressure used for T_at_lcl and Z3_at_lcl')
    T_at_lcl = set_attrs(T_at_lcl, long_name='Temperature at LCL pressure', units='K',
                         description='Actually temperature at approx LCL pressure given by p_at_lcl.')
    Z3_at_lcl = set_attrs(Z3_at_lcl, long_name='Geopotential height at LCL pressure', units='m',
                          description='Actually geopotential height at approx LCL pressure given by p_at_lcl.')

    ds_out = xr.Dataset({'p_lcl': p_lcl, 'T_lcl': T_lcl, 'Z3_lcl': Z3_lcl,
                         'p_at_lcl': p_at_lcl.drop_vars('plev'), 'T_at_lcl': T_at_lcl.drop_vars('plev'),
                         'Z3_at_lcl': Z3_at_lcl.drop_vars('plev')})
    ds_out = ds_out.astype('float32')
    encoding = {var: {'zlib': True, 'complevel': complevel} for var in ds_out.data_vars}

    func_save = lambda: ds_out.to_netcdf(out_file, encoding=encoding)
    run_func_loop(func_save, max_wait_time=max_wait_time, wait_interval=wait_interval, logger=logger)
    print_log(f'Year {year} - End', logger)
    return None


def main(input_file_path: str):
    logger = logging.getLogger()  # for printing to console time info
    logger.info(f'Accessing input directory')

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
