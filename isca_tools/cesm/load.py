import os
import xarray as xr
import cftime
from typing import Optional, List, Union, Literal, Callable
import fnmatch
import numpy as np
import warnings
import logging
import re
from datetime import datetime, timedelta
from ..utils.base import parse_int_list
from isca_tools.utils.constants import g
from isca_tools.utils.xarray import set_attrs
from xarray.coding.cftimeindex import CFTimeIndex
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

jasmin_archive_dir = '/gws/nopw/j04/global_ex/jamd1/cesm/CESM2.1.3/archive/'
local_archive_dir = '/Users/joshduffield/Documents/StAndrews/Isca/cesm/archive/'
jasmin_surf_geopotential_file = ('/gws/nopw/j04/global_ex/jamd1/cesm/CESM2.1.3/cesm_inputdata/atm/cam/topo/'
                                 'fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc')


def load_dataset(exp_name: str, comp: str = 'atm',
                 archive_dir: str = jasmin_archive_dir,
                 hist_file: int = 0,
                 chunks: Optional[Union[dict, Literal["auto"], int]] = None,
                 combine: Literal["by_coords", "nested"] = 'nested',
                 concat_dim: str = 'time',
                 decode_times: bool = True,
                 parallel: bool = False,
                 preprocess: Optional[Callable] = None,
                 year_files: Optional[Union[int, List, str]] = None,
                 month_files: Optional[Union[int, List, str]] = None,
                 apply_month_shift_fix: bool = True,
                 logger: Optional[logging.Logger] = None) -> xr.Dataset:
    """
    This loads a dataset of a given component produced by CESM.

    Args:
        exp_name: Name of folder in `archive_dir` where data for this experiment was saved.
        comp: Component of CESM to load data from.</br>
            Options are:

            * `atm`: atmosphere
            * `ice`: ice
            * `lnd`: land
            * `rof`: river
        archive_dir: Directory where CESM archive data saved.
        hist_file: Which history file to load, `0` is the default monthly averaged data set.
        chunks: Dictionary with keys given by dimension names and values given by chunk sizes
            e.g. `{"time": 365, "lat": 50, "lon": 100}`.</br>
            Has big impact on memory usage. If `None`, no chunking is performed.
        combine: Whether `xarray.combine_by_coords` or `xarray.combine_nested` is used to combine all the data.
        concat_dim: Dimensions to concatenate files along.
            You only need to provide this argument if combine='nested'.
        parallel: Whether parallel loading is performed.
        preprocess: Function to preprocess the data before loading.
        decode_times: If `True`, will convert time to actual date.
        year_files: Only files with these years in their name will be loaded. Leave as `None` to load all years.</br>
            As well as integer or list of integers, there are three string options:
            * `'1975:1979'` will load in all years between 1975 and 1979 inclusive.
            * `first5` will load in the first 5 years.
            * `last5` will load in the last 5 years.
        month_files: Only files with these months (1 is Jan) in their names will be loaded.
            Leave as `None` to load all months.</br>
            As well as integer or list of integers, there are three is a single string option:
            `'2:5'` will load in all months between 2 and 5 inclusive.
        apply_month_shift_fix: If `True`, will apply `ds_month_shift` before returning dataset.</br>
            Only used for monthly averaged data i.e. `hist_file=0`.
        logger: Optional logger.

    Returns:
        Dataset containing all diagnostics specified for the experiment.
    """
    exp_dir, comp_id = get_exp_dir(exp_name, comp, archive_dir)
    if year_files is None and month_files is None:
        # Load all data in folder
        # * indicates where date index info is, so we combine all datasets
        data_files_load = os.path.join(exp_dir, f'{exp_name}.{comp_id}.h{hist_file}.*.nc')
    else:
        if hist_file != 0 and month_files is not None:
            warnings.warn(f'If h{hist_file} files not saved monthly then will not have a file for each month so '
                          f'using months_keep={month_files} will miss out different days in different years.')
        file_dates = get_exp_file_dates(exp_name, comp, archive_dir, hist_file)
        year_files_all = np.unique(file_dates.dt.year).tolist()
        if year_files is None:
            year_files = year_files_all         # all possible years
        else:
            year_files = parse_int_list(year_files, format_func=lambda x: int(x), all_values=year_files_all)
            years_request_missing = [x for x in year_files if x not in year_files_all]
            if len(years_request_missing) > 0:
                warnings.warn(f'The requested years = {years_request_missing}\n'
                              f'are missing from the available years = {year_files_all}')
        month_files_all = np.unique(file_dates.dt.month).tolist()
        if month_files is None:
            month_files = month_files_all       # all possible months
        else:
            month_files = parse_int_list(month_files, format_func= lambda x: int(x))
            month_request_missing = [x for x in month_files if x not in month_files_all]
            if len(month_request_missing) > 0:
                warnings.warn(f'The requested months = {month_request_missing}\n'
                              f'are missing from the available months = {month_files_all}')

        file_ind_keep = [i for i in range(file_dates.size) if (file_dates.dt.year[i] in year_files)
                         and (file_dates.dt.month[i] in month_files)]
        if len(file_ind_keep) == 0:
            raise ValueError(f'No files with requested years and months in file name\n'
                             f'Available years: {np.unique(file_dates.dt.year).tolist()}\n'
                             f'Available months: {np.unique(file_dates.dt.month).tolist()}\n'
                             f'Requested years: {year_files}\n'
                             f'Requested months: {month_files}\n')

        # Only load in specific years and/or months
        data_files_all = os.listdir(exp_dir)
        # only keep files of correct format
        data_files_all = [file for file in data_files_all if
                          fnmatch.fnmatch(file, f'{exp_name}.{comp_id}.h{hist_file}.*.nc')]
        # only keep files with requested years and months in file name
        data_files_load = [os.path.join(exp_dir, file) for i, file in enumerate(data_files_all) if
                           i in file_ind_keep]
    if logger:
        if isinstance(data_files_load, str):
            logger.info(f'Loading data from all files: {data_files_load}')
        else:
            files_str = "\n".join(data_files_load)
            logger.info(f'Loading data from {len(data_files_load)} files:\n{files_str}')
    if apply_month_shift_fix and hist_file == 0:
        ds = xr.open_mfdataset(data_files_load, decode_times=False, concat_dim=concat_dim,
                               combine=combine, chunks=chunks, parallel=parallel, preprocess=preprocess)
        return ds_month_shift(ds, decode_times)
    else:
        return xr.open_mfdataset(data_files_load, decode_times=decode_times,
                                 concat_dim=concat_dim, combine=combine, chunks=chunks, parallel=parallel,
                                 preprocess=preprocess)


def get_exp_dir(exp_name: str, comp: str = 'atm', archive_dir: str = jasmin_archive_dir):
    """

    Args:
        exp_name: Name of folder in `archive_dir` where data for this experiment was saved.
        comp: Component of CESM to load data from.</br>
            Options are:

            * `atm`: atmosphere
            * `ice`: ice
            * `lnd`: land
            * `rof`: river
        archive_dir: Directory where CESM archive data saved.

    Returns:

    """
    # LHS of comp_id_dict is name of directory containing hist files for the component
    # RHS of comp_id_dict is the string indicating the component in the individual .nc files within this directory
    comp_id_dict = {'atm': 'cam',  # atmosphere
                    'ice': 'cice',  # ice
                    'lnd': 'clm2',  # land
                    'rof': 'mosart'}  # river
    if comp not in comp_id_dict:
        # Generate inverse dict to comp_id_dict
        comp_id_dict_reverse = {key: list(comp_id_dict.keys())[i] for i, key in enumerate(comp_id_dict.values())}
        if comp in comp_id_dict_reverse:
            # Deal with case where give comp_file not comp_dir i.e. 'cam' rather than 'atm'
            comp_dir = os.path.join(archive_dir, exp_name, comp_id_dict_reverse[comp])
            comp_id = comp
        else:
            raise ValueError(f'comp must be one of {list(comp_id_dict.keys())} but got {comp}')
    else:
        comp_dir = os.path.join(archive_dir, exp_name, comp)
        comp_id = comp_id_dict[comp]
    return os.path.join(comp_dir, 'hist'), comp_id

def parse_cesm_datetime(time_str) -> datetime:
    """
    Given a time string in the form either 'YYYY-MM' or 'YYYY-MM-DD-sssss' where `sssss` are the seconds since midnight,
    this will return the datetime object corresponding to that time.

    Args:
        time_str: String to convert to datetime object.

    Returns:
        Datetime object corresponding to `time_str`.
    """
    date_part = time_str[:10]               # 'YYYY-MM-DD'
    if len(date_part) == 7:
        return datetime.strptime(date_part, '%Y-%m')
    else:
        seconds_since_midnight = int(time_str[11:])  # e.g., 00000 → 0 seconds, 04320 → 01:12:00 (1 hour, 12 minutes)
        base_date = datetime.strptime(date_part, '%Y-%m-%d')
        return base_date + timedelta(seconds=seconds_since_midnight)


def get_exp_file_dates(exp_name: str, comp: str = 'atm', archive_dir: str = jasmin_archive_dir,
                       hist_file: int = 0) -> xr.DataArray:
    """
    Get dates indicated in file names of a particular experiment.

    Args:
        exp_name: Name of folder in `archive_dir` where data for this experiment was saved.
        comp: Component of CESM to load data from.</br>
            Options are:

            * `atm`: atmosphere
            * `ice`: ice
            * `lnd`: land
            * `rof`: river
        archive_dir: Directory where CESM archive data saved.
        hist_file: Which history file to load, `0` is the default monthly averaged data set.

    Returns:
        DataArray of dates indicated in file names of `exp_name`.
    """
    exp_dir, comp_id = get_exp_dir(exp_name, comp, archive_dir)
    # Only load in specific years and/or months
    data_files_all = os.listdir(exp_dir)
    # only keep files of correct format
    file_dates = []
    for file in data_files_all:
        date = re.search(rf'h{hist_file}\.(.*?)\.nc', file)
        if not date:
            continue
        file_dates.append(parse_cesm_datetime(date.group(1)))
    try:
        return xr.DataArray(np.array(file_dates, dtype='datetime64[D]'), dims="time", name="time")
    except OutOfBoundsDatetime as e:
        warnings.warn(f"Got out of bounds error, re-trying with NoLeap Calendar and cftime\n{e}")
        cftime_dates = [cftime.DatetimeNoLeap(dt.year, dt.month, dt.day) for dt in file_dates]
        return xr.DataArray(CFTimeIndex(cftime_dates, calendar="noleap"), dims="time", name="time")



def ds_month_shift(ds: xr.Dataset, decode_times: bool = True) -> xr.Dataset:
    """
    When loading CESM data, for some reason the first month is marked as February, so this function
    shifts the time variable to correct it to January.

    Args:
        ds: Dataset to apply the shift to.
            It should have been loaded with `decode_times=False`.
        decode_times: If `True`, will convert time to actual date.

    Returns:
        Dataset with first months shifted by -1 so now first month is January.
    """
    n_day_month = np.asarray(
        [cftime.DatetimeNoLeap(1, i + 1, 1).daysinmonth for i in range(12)])  # number of days in each month
    months_in_ds = np.arange(1, 13)                             # TODO may have to get months from ds e.g. ds.time.dt.month
    n_day_month = n_day_month[np.asarray(months_in_ds) - 1]
    n_months_in_ds = ds.time.size
    n_years_in_ds = int(np.floor(n_months_in_ds / 12))
    month_shift_array = np.concatenate((np.tile(n_day_month, n_years_in_ds),
                                        n_day_month[:n_months_in_ds % 12]))
    ds_new = ds.assign_coords({'time': ('time', ds.time.values - month_shift_array, ds.time.attrs)})
    if decode_times:
        ds_new = xr.decode_cf(ds_new)
    return ds_new

def select_months(ds: xr.Dataset, month_nh: Union[np.ndarray, List[int]],
                  month_sh: Optional[Union[np.ndarray, List[int]]] = None) -> xr.Dataset:
    """
    In dataset, keep only `month_nh` months in the northern hemisphere, and `month_sh` in the southern hemisphere.

    Args:
        ds: Dataset to select months from.
        month_nh: List of months to keep in northern hemisphere.
        month_sh: List of months to keep in southern hemisphere. If `None`, will be the same as `month_nh`.

    Returns:
        Dataset with months selected.
    """
    # Select months for NH
    if month_sh is None:
        mask = ds.time.dt.month.isin(month_nh)
    else:
        mask_nh = (ds.lat >= 0) & (ds.time.dt.month.isin(month_nh))
        mask_sh = (ds.lat < 0) & (ds.time.dt.month.isin(month_sh))
        mask = mask_nh | mask_sh
    return ds.where(mask)


def load_z2m(surf_geopotential_file: str = jasmin_surf_geopotential_file,
             var_reindex_like: Optional[xr.DataArray] = None) -> xr.DataArray:
    """
    Returns 2m geopotential height for CESM simulation.

    Args:
        surf_geopotential_file: File location of input data containing the geopotential at the surface: `PHIS`.
        var_reindex_like: Can provide a variable so `z2m` will have the same lat-lon as this variable.

    Returns:
        2m geopotential height in units of meters.
    """
    # PHIS is the geopotential at the surface, so to get Z at reference height, divide by g and add 2
    ds_z2m = xr.open_dataset(surf_geopotential_file)[['PHIS']]
    z_refht = 2   # reference height is at 2m
    ds_z2m['ZREFHT'] = ds_z2m['PHIS'] / g + z_refht               # PHIS is geopotential in m2/s2 so need to convert
    del ds_z2m['PHIS']
    if var_reindex_like is not None:
        ds_z2m = ds_z2m.reindex_like(var_reindex_like, method="nearest", tolerance=0.01)
    ds_z2m = set_attrs(ds_z2m.ZREFHT, long_name='Geopotential height at reference height (2m)', units='m')
    return ds_z2m
