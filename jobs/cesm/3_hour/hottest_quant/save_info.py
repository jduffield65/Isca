# Script to save variables conditioned on times of hottest % of days
# One value for each `sample` in the hottest % of days. Only 1 time for each day corresponding to hottest time of day
from geocat.comp import interp_hybrid_to_pressure
from isca_tools import cesm
from isca_tools.utils.ds_slicing import lat_lon_range_slice
from isca_tools.utils import get_memory_usage
from isca_tools.utils.base import top_n_peaks_ind
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
                  var: Union[str, List],
                  year_files: Optional[Union[int, List, str]] = None,
                  month_files: Optional[Union[int, List, str]] = None,
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
                           year_files=year_files, month_files=month_files, chunks=chunks, parallel=load_parallel,
                           preprocess=preprocess_atm, logger=logger)
    logger.info('Loaded data')
    return ds


def process_ds(ds: xr.Dataset, n_sample: int, ind_spacing: int, hyam: xr.DataArray, hybm: xr.DataArray, p0: float,
               load_all_at_start: bool = True, temp_ft_plev: Optional[List] = None,
               refht_level_index: Optional[int] = None, ds_name: str = 'All Data', logger: Optional[logging.Logger] = None):
    logger.info(f"{ds_name} | Started | Memory used {get_memory_usage() / 1000:.1f}GB")

    if load_all_at_start:
        ds = ds.load()

    # Optional: compute temp_ft_zonal_av for this latitude
    if temp_ft_plev is not None:
        temp_ft_lat = interp_hybrid_to_pressure(ds.T, ds.PS, hyam, hybm, p0,
                                                np.atleast_1d(temp_ft_plev),
                                                lev_dim='lev')

        temp_ft_lat = temp_ft_lat.load()

        temp_ft_zonal_av = temp_ft_lat.mean(dim='lon')
        # del temp_ft_lat  # release memory early
        logger.info(f"{ds_name} | Computed T_FT | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Select temperature variable
    T_var = (ds.TREFHT if refht_level_index is None else
             ds.T.isel(lev=refht_level_index))

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

    # Select those times and store in temporary dataset
    ds_out = ds.isel(time=idx_quant)
    ds_out['time_ind'] = idx_quant

    if temp_ft_plev is not None:
        ds_out['T_zonal_av'] = temp_ft_zonal_av

    # Explicitly delete large intermediates
    # del ds_lat, ds_out_lat, idx_quant, T_var

    logger.info(f"{ds_name} | Finished | Memory used {get_memory_usage() / 1000:.1f}GB")
    return ds_out


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
            ds_out = process_ds(ds.isel(lat=i), n_sample, ind_spacing, hyam, hybm, p0, script_info['load_all_at_start'],
                                script_info['temp_ft_plev'], script_info['refht_level_index'], f'Lat {i}/{n_lat}',
                                logger)
            ds_out = ds_out.expand_dims(lat=[lat_val])  # Add latitude coordinate (needed after recombining)
            lat_results.append(ds_out)
        # Concatenate along latitude
        ds_out = xr.concat(lat_results, dim='lat')
        logger.info(f"Recombined all latitudes | Final memory used {get_memory_usage() / 1000:.1f}GB")
    else:
        ds_out = process_ds(ds, n_sample, ind_spacing, hyam, hybm, p0, script_info['load_all_at_start'],
                            script_info['temp_ft_plev'], script_info['refht_level_index'], f'{n_lat} Latitudes',
                            logger)

    # Loop over each latitude so only load in one lat at a time
    for i, lat_val in enumerate(ds.lat.values):
        logger.info(f"Lat {i+1}/{n_lat} | Started | Memory used {get_memory_usage() / 1000:.1f}GB")

        # Load only this latitude slice into memory
        ds_lat = ds.sel(lat=lat_val)

        if script_info['load_all_at_start']:
            ds_lat = ds_lat.load()  # remove gw, hyam, hybm, p0 from ds
            logger.info(f"Lat {i+1}/{n_lat} | Fully loaded data | Memory used {get_memory_usage() / 1000:.1f}GB")

        # Optional: compute temp_ft_zonal_av for this latitude
        if script_info['temp_ft_plev'] is not None:
            temp_ft_lat = interp_hybrid_to_pressure(ds_lat.T, ds_lat.PS, hyam, hybm, p0,
                                                    np.atleast_1d(script_info['temp_ft_plev']),
                                                    lev_dim='lev')

            temp_ft_lat = temp_ft_lat.load()

            temp_ft_zonal_av = temp_ft_lat.mean(dim='lon')
            del temp_ft_lat  # release memory early
            logger.info(f"Lat {i + 1}/{n_lat} | Computed T_FT | Memory used {get_memory_usage() / 1000:.1f}GB")

        # Select temperature variable
        T_var = (ds_lat.TREFHT if script_info['refht_level_index'] is None else
                 ds_lat.T.isel(lev=script_info['refht_level_index']))

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
        logger.info(f"Lat {i + 1}/{n_lat} | Computed hottest {n_sample} indices | Memory used {get_memory_usage() / 1000:.1f}GB")

        # Select those times and store in temporary dataset
        ds_out_lat = ds_lat.isel(time=idx_quant)
        ds_out_lat['time_ind'] = idx_quant

        if script_info['temp_ft_plev'] is not None:
            ds_out_lat['T_zonal_av'] = temp_ft_zonal_av

        # Add latitude coordinate (needed after recombining)
        ds_out_lat = ds_out_lat.expand_dims(lat=[lat_val])

        # Append to results list
        lat_results.append(ds_out_lat)

        # Explicitly delete large intermediates
        del ds_lat, ds_out_lat, idx_quant, T_var
        # gc.collect()  # force garbage collection

        logger.info(f"Lat {i + 1}/{n_lat} | Finished | Memory used {get_memory_usage() / 1000:.1f}GB")


    # Concatenate along latitude
    ds_out = xr.concat(lat_results, dim='lat')
    logger.info(f"Recombined all latitudes | Final memory used {get_memory_usage() / 1000:.1f}GB")

    # # Fully load data if chose to do that
    # if script_info['load_all_at_start']:
    #     ds[script_info['var']].load()  # remove gw, hyam, hybm, p0 from ds
    #     logger.info(f"Fully loaded data | Memory used {get_memory_usage() / 1000:.1f}GB")
    #
    # if script_info['temp_ft_plev'] is not None:
    #     temp_ft_zonal_av = interp_hybrid_to_pressure(ds.T, ds.PS, hyam, hybm, p0,
    #                                                  np.atleast_1d(script_info['temp_ft_plev']), lev_dim='lev')
    #     logger.info(f"Computed temp_ft | Memory used {get_memory_usage() / 1000:.1f}GB")
    #     temp_ft_zonal_av = temp_ft_zonal_av.load()
    #     logger.info(f"Loaded temp_ft | Memory used {get_memory_usage() / 1000:.1f}GB")
    #     temp_ft_zonal_av = temp_ft_zonal_av.mean(dim='lon')
    #     logger.info(f"Taken zonal average of temp_ft | Memory used {get_memory_usage() / 1000:.1f}GB")
    #
    # # Get indices of hottest quant at each location
    # time_dt = (ds.time[1] - ds.time[0]) / np.timedelta64(1, 'h')
    # ind_spacing = int(np.ceil(script_info['hour_spacing'] / time_dt))
    #
    # # Find number of values to get: quantile is relative to number of days not number of time values
    # n_sample = int(np.ceil((1 - script_info['quant'] / 100) * np.unique(ds.time.dt.floor('D')).size))
    #
    # idx_quant = xr.apply_ufunc(
    #     top_n_peaks_ind,
    #     ds.TREFHT if script_info['refht_level_index'] is None
    #     else ds.T.isel(lev=script_info['refht_level_index']),
    #     input_core_dims=[["time"]],
    #     output_core_dims=[["sample"]],
    #     vectorize=True,
    #     kwargs={"n": n_sample, "min_ind_spacing": ind_spacing},
    #     dask="parallelized",
    #     output_dtypes=[int],
    # )
    # idx_quant = idx_quant.assign_coords(sample=np.arange(1, n_sample + 1))  # make sample a coordinate
    #
    # logger.info(f"Computed hottest {n_sample} indices | Memory used {get_memory_usage() / 1000:.1f}GB")
    #
    # ds_out = ds.isel(time=idx_quant)
    # ds_out['time_ind'] = idx_quant
    # if script_info['temp_ft_plev'] is not None:
    #     ds_out['T_zonal_av'] = temp_ft_zonal_av
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
