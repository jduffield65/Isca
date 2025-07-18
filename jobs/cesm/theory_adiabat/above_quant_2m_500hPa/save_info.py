# Script to save REFHT and 500hPa level info for averaged over days corresponding to a given quantile of TREFHT.
# Idea is that this gives useful variables to decompose vertical coupling between REFHT and 500hPa on hot days.
import os
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


def load_raw_data(exp_name: str, archive_dir: str, plev_dir: Union[List, str], surf_geopotential_file: Optional[str] = None,
                  year_files: Optional[Union[int, List, str]] = None, refht_level_index: Optional[int] = None,
                  lon_min: Optional[float] = None, lon_max: Optional[float] = None,
                  lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                  chunks_time: Optional[int] = None, chunks_lat: Optional[int] = None, chunks_lon: Optional[int] = None,
                  load_parallel: bool = False, logger: Optional[logging.Logger] = None, load_soilliq: bool = True) -> xr.Dataset:
    """
    Returns dataset with daily average variables TREFHT, QREFHT, ZREFHT, PREFHT, PS, SOILLIQ, as well as all the
    daily average variables in plev_dir directories.
    If `refht_level_index` is not `None`, REFHT will be the model level specified, otherwise the CESM `REFHT` is used.

    Args:
        exp_name: Name of experiment within archive_dir
        archive_dir: Directory containing cesm experiments on JASMIN
        plev_dir: list of directories containing info on p levels
        surf_geopotential_file: Location of file containing PHIS which is the 2m geopotential.
            Only required if `refht_level=None`
        year_files: Only files with these years in their name will be loaded. Leave as None to load all years.
        refht_level_index: use model level as REFHT rather than 2m if give number
        lon_min:
        lon_max:
        lat_min:
        lat_max:
        chunks_time: Chunking info for less memory usage (leave empty to not chunk in given dimension)
        chunks_lat: Chunking info for less memory usage (1 chunk per latitude as we do lat by lat comp)
        chunks_lon: Chunking info for less memory usage
        load_parallel: Whether to use parallel processing when loading
        logger:
        load_soilliq: Whether to include `SOILLIQ` variable in output

    Returns:

    """
    # Load land data first as smallest dataset so should be quick
    def preprocess_land(ds):
        # Only 2 variables, and sum over all soil levels
        ds = lat_lon_range_slice(ds, lat_min, lat_max, lon_min, lon_max)
        soil_liq_sum = ds['SOILLIQ'].sum(dim='levsoi')  # Sum over 'levsoi'
        return xr.Dataset({'SOILLIQ': soil_liq_sum})

    chunks = {"time": chunks_time, "lat": chunks_lat, "lon": chunks_lon}
    if load_soilliq:
        ds_land = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=1, comp='lnd',
                                    year_files=year_files, chunks=chunks, parallel=load_parallel,
                                    preprocess=preprocess_land,
                                    logger=logger)
        print_log('Loaded land data', logger=logger)

    # Load REFHT atmospheric data
    def preprocess_atm(ds):
        # Preprocessing so don't load in entire dataset
        ds = lat_lon_range_slice(ds, lat_min, lat_max, lon_min, lon_max)
        if refht_level_index is None:
            ds = ds[['TREFHT', 'QREFHT', 'PS']]
        else:
            ds = ds[['PS', 'T', 'Z3', 'Q']].isel(lev=refht_level_index)
        return ds

    ds_surf = cesm.load_dataset(exp_name, archive_dir=archive_dir, hist_file=1, comp='atm',
                                year_files=year_files, chunks=chunks, parallel=load_parallel,
                                preprocess=preprocess_atm, logger=logger)
    print_log('Loaded near-surface data', logger=logger)
    if refht_level_index is None:
        # If use 2m, need to load in geopotential
        ds_surf['PREFHT'] = ds_surf['PS']           # refht is 2m so assume surface pressure at this pressure
        ds_surf['ZREFHT'] = cesm.load.load_z2m(surf_geopotential_file, var_reindex_like=ds_surf['PS'])
        print_log('Loaded surface geopotential', logger=logger)
    else:
        # If use model level, need to rename
        ds_surf = ds_surf.rename({"T": "TREFHT", "Z3": "ZREFHT", "Q": "QREFHT"})
        def get_lev_info():
            # Use test dataset to get hybrid model levels
            ds = cesm.load_dataset('test', archive_dir=archive_dir,
                                   hist_file=1, year_files='first1')
            return ds.hyam.isel(time=0), ds.hybm.isel(time=0), float(ds.P0.isel(time=0))

        # Get pressure at the chosen reference level, if not surface
        hyam, hybm, p0 = get_lev_info()
        ds_surf['PREFHT'] = cesm.get_pressure(ds_surf['PS'], p0, hyam.isel(lev=refht_level_index),
                                              hybm.isel(lev=refht_level_index))

    if load_soilliq:
        # re-index so lat align - otherwise get issues because lat slightly different
        ds_land = ds_land.reindex_like(ds_surf['PS'], method="nearest", tolerance=0.01)

    # Load additional data on pressure levels, which is saved in previously created directory
    if isinstance(plev_dir, str):
        plev_dir = [plev_dir]
    ds_plev = []
    for dir in plev_dir:
        plev_path = os.path.join(archive_dir, exp_name, dir)
        if year_files is None:
            ds_plev.append(xr.open_mfdataset(os.path.join(plev_path, '*.nc')))
        else:
            # Each file within directory is just the year (with 4 digits) followed by .nc
            year_files_all = [int(var.replace('.nc', '')) for var in os.listdir(plev_path) if '.nc' in var]
            # Get all years in files which are requested and in year_files_all
            year_req0 = parse_int_list(year_files, format_func=lambda x: int(x), all_values=year_files_all)
            year_req = [year_files_all[i] for i in range(len(year_files_all)) if year_files_all[i] in year_req0]
            # Load in this dataset
            ds_plev.append(xr.open_mfdataset([os.path.join(plev_path, f'{year:04d}.nc') for year in year_req]))
        # re-index so lat align - otherwise get issues because lat slightly different
        ds_plev[-1] = ds_plev[-1].reindex_like(ds_surf['PS'], method="nearest", tolerance=0.01)
    ds_plev = xr.merge(ds_plev)
    print_log('Loaded pressure-level data', logger=logger)

    set_attrs(ds_surf.ZREFHT, long_name=ds_plev.Z3.long_name, units=ds_plev.Z3.units)

    if load_soilliq:
        return xr.merge([ds_surf, ds_land, ds_plev])
    else:
        return xr.merge([ds_surf, ds_plev])


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

    # Load daily average data
    # Arguments required for `load_raw_data` are all in `script_info` with same name, hence inspect.signature stuff
    func_arg_names = inspect.signature(load_raw_data).parameters
    func_args = {k: v for k, v in script_info.items() if k in func_arg_names}
    ds = load_raw_data(**func_args, logger=logger)
    logger.info(f"Finished lazy-loading datasets | Memory used {get_memory_usage() / 1000:.1f}GB")

    # Breakdown T_ft into zonal average and anomaly from this
    ds['T_zonal_av'] = ds.T.mean(dim='lon')     # zonal average is computed over all surfaces
    ds['T_anom'] = ds['T'] - ds['T_zonal_av']

    # Fully load data if chose to do that
    if script_info['load_all_at_start']:
        ds.load()
        logger.info(f"Fully loaded data | Memory used {get_memory_usage() / 1000:.1f}GB")


    # Initialize dict to save output data - n_surf (land or ocean) x n_lat x n_quant
    # Initialize as nan, so remains as Nan if not enough data to perform calculation
    n_lat = ds.lat.size
    n_lon = ds.lon.size
    # find quantile by looking for all values in quantile range between quant_use-quant_range to quant_use+quant_range
    n_quant = len_safe(script_info['quant'])
    vars_out_same_in = ['SOILLIQ', 'PS', 'TREFHT', 'QREFHT', 'T', 'T_zonal_av', 'T_anom', 'Z3',
                        'p_lcl', 'T_lcl', 'Z3_lcl', 'p_at_lcl', 'T_at_lcl', 'Z3_at_lcl']
    if script_info['refht_level_index'] is not None:
        vars_out_same_in += ['PREFHT', 'ZREFHT']        # if using model level as REFHT, save pressure and Z here

    output_info = {var: np.full((n_quant, n_lat, n_lon), np.nan, dtype=float) for var in
                   vars_out_same_in + ['rh_refht', 'mse_refht', 'mse_sat_ft', 'mse_lapse',
                                       'lapse_below_lcl', 'lapse_above_lcl']}
    # Record attribute info for output
    output_long_name = {'rh_refht': 'Relative humidity at reference height (2m)',
                       'mse_refht': 'Moist static energy at reference height (2m)',
                       'mse_sat_ft': 'Saturated moist static energy at free troposphere level set by plev',
                       'mse_lapse': 'mse_refht - mse_sat_ft averaged over days considering',
                       'lapse_below_lcl': 'Lapse rate between REFHT (2m) and LCL',
                       'lapse_above_lcl': 'Lapse rate between LCL and free troposphere level set by plev'}
    output_units = {'rh_refht': 'Dimensionless', 'mse_refht': 'kJ/kg', 'mse_sat_ft': 'kJ/kg', 'mse_lapse': 'kJ/kg',
                    'lapse_below_lcl': 'K/km', 'lapse_above_lcl': 'K/km'}
    for var in vars_out_same_in:
        output_long_name[var] = ds[var].long_name if 'long_name' in ds[var].attrs else ''
        output_units[var] = ds[var].units if 'units' in ds[var].attrs else ''

    # Also save standard deviation for each variable, as a way to quantify error
    var_keys = [key for key in output_info.keys()]
    for var in var_keys:
        output_info[var + '_std'] = np.full_like(output_info[var], np.nan, dtype=float)
    output_info['use_in_calc'] = np.zeros((n_quant, n_lat, ds.lon.size, ds.time.size), dtype=bool)

    coords = {'quant': np.array(script_info['quant'], ndmin=1),
              'lat': ds.lat, 'lon': ds.lon, 'time': ds.time}

    # Coordinate info to convert to xarray datasets
    output_dims = {var: ['quant', 'lat', 'lon'] for var in output_info}
    output_dims['use_in_calc'] = ['quant', 'lat', 'lon', 'time']


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
                quant_mask = get_quant_ind(ds_latlon.TREFHT, coords['quant'][j],
                                           script_info['quant_range_below'], script_info['quant_range_above'],
                                           return_mask=True, av_dim='time')
                if quant_mask.sum() == 0:
                    logger.info(
                        f"Latitude {i + 1}/{n_lat} | Longitude {k + 1}/{n_lon}: no data found for quant={coords['quant'][j]}")
                    continue
                ds_use = ds_latlon.where(quant_mask)
                output_info['use_in_calc'][j, i, k] = quant_mask
                # Gather all variables, want to get quantile info for
                var_use = {}
                for var in vars_out_same_in:
                    var_use[var] = ds_use[var]
                var_use['rh_refht'] = ds_use.QREFHT / sphum_sat(ds_use.TREFHT, ds_use.PREFHT)
                var_use['mse_refht'] = moist_static_energy(ds_use.TREFHT, ds_use.QREFHT, ds_use.ZREFHT)
                var_use['mse_sat_ft'] = moist_static_energy(ds_use.T, sphum_sat(ds_use.T, ds_use.plev), ds_use.Z3)
                var_use['mse_lapse'] = var_use['mse_refht'] - var_use['mse_sat_ft']
                var_use['lapse_below_lcl'] = (ds_use.T_at_lcl - ds_use.TREFHT) / (ds_use.ZREFHT - ds_use.Z3_at_lcl) * 1000  # *1000 so K/km
                var_use['lapse_above_lcl'] = (ds_use.T - ds_use.T_at_lcl) / (ds_use.Z3_at_lcl - ds_use.Z3) * 1000
                for key in var_use:
                    # Average over all days in the quant_mask for each variable
                    output_info[key][j, i, k] = var_use[key].mean(dim='time')
                    output_info[key + '_std'][j, i, k] = var_use[key].std(dim='time')
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
    # Add pressure level info to variables at single level
    for var in ['mse_sat_ft', 'mse_lapse', 'T', 'Z3']:
        output_info[var] = output_info[var].expand_dims(plev=ds.coords['plev'])
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
