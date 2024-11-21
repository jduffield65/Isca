import os
import xarray as xr
import cftime
from typing import Optional, List
import fnmatch
import numpy as np
import warnings

jasmin_archive_dir = '/gws/nopw/j04/global_ex/jamd1/cesm/CESM2.1.3/archive/'
local_archive_dir = '/Users/joshduffield/Documents/StAndrews/Isca/cesm/archive/'


def load_dataset(exp_name: str, comp: str = 'atm',
                 archive_dir: str = jasmin_archive_dir,
                 hist_file: int = 0,
                 decode_times: bool = True,
                 year_first: int = 1, year_last: int = -1,
                 months_keep: Optional[List] = None,
                 apply_month_shift_fix: bool = True) -> xr.Dataset:
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
        hist_file: Which history file to load, `0` is the default monthly averaged data set.
        archive_dir: Directory where CESM archive data saved.
        decode_times: If `True`, will convert time to actual date.
        year_first: First year of simulation to load.</br>
            Only used for monthly averaged data i.e. `hist_file=0`.
        year_last: Last year of simulation to load.</br>
            Only used for monthly averaged data i.e. `hist_file=0`.
        months_keep: List of months which you want to load for each year.</br>
            Only used for monthly averaged data i.e. `hist_file=0`.</br>
            `1` refers to January. If `None`, all 12 months will be loaded.</br>
            If directory only contains specific months, should still specify those months here.
        apply_month_shift_fix: If `True`, will apply `ds_month_shift` before returning dataset.</br>
            Only used for monthly averaged data i.e. `hist_file=0`.

    Returns:
        Dataset containing all diagnostics specified for the experiment.
    """
    # LHS of comp_dir_file is name of directory containing hist files for the component
    # RHS of comp_dir_file is the string indicating the component in the individual .nc files within this directory
    comp_dir_file = {'atm': 'cam',  # atmosphere
                     'ice': 'cice',  # ice
                     'lnd': 'clm2',  # land
                     'rof': 'mosart'}  # river

    if comp not in comp_dir_file:
        # Generate inverse dict to comp_dir_file
        comp_file_dir = {key: list(comp_dir_file.keys())[i] for i, key in enumerate(comp_dir_file.values())}
        if comp in comp_file_dir:
            # Deal with case where give comp_file not comp_dir i.e. 'cam' rather than 'atm'
            comp_dir = os.path.join(archive_dir, exp_name, comp_file_dir[comp])
            comp_file = comp
        else:
            raise ValueError(f'comp must be one of {list(comp_dir_file.keys())} but got {comp}')
    else:
        comp_dir = os.path.join(archive_dir, exp_name, comp)
        comp_file = comp_dir_file[comp]
    if year_first == 1 and year_last == -1 and months_keep is None:
        # Load all data in folder
        # * indicates where date index info is, so we combine all datasets
        data_files_load = os.path.join(comp_dir, 'hist', f'{exp_name}.{comp_file}.h{hist_file}.*.nc')
    else:
        if hist_file != 0 and months_keep is not None:
            warnings.warn(f'If h{hist_file} files not saved monthly then will not have a file for each month so '
                          f'using months_keep={months_keep} will miss out different days in different years.')
        # Only load in specific years and/or months
        data_files_all = os.listdir(os.path.join(comp_dir, 'hist'))
        # only keep files of correct format
        data_files_all = [file for file in data_files_all if
                          fnmatch.fnmatch(file, f'{exp_name}.{comp_file}.h{hist_file}.*.nc')]

        # Extract the year and month that each file points to
        # If not h0 files, then files have time indicated in the name, hence the different indices used for non h0 files
        file_year = np.asarray([int(file[-10-9*int(hist_file != 0):-6-9*int(hist_file != 0)])
                                for file in data_files_all])
        file_month = np.asarray([int(file[-5-9*int(hist_file != 0):-3-9*int(hist_file != 0)])
                                 for file in data_files_all])

        if year_last < 0:
            year_last_use = year_last + np.max(file_year) + 1  # i.e. -1 changes to max(file_year)
        else:
            year_last_use = year_last
        if year_first < 0:
            year_first_use = year_first + np.max(file_year) + 1  # i.e. -1 changes to max(file_year)
        else:
            year_first_use = year_first

        if year_first_use not in file_year:
            raise ValueError(f'year_first={year_first_use} is not in available years: {file_year}')
        if year_last_use not in file_year:
            raise ValueError(f'year_last={year_last_use} is not in available years: {file_year}')

        if months_keep is None:
            months_keep = np.arange(1, 13)  # keep all months
        else:
            months_error = []
            for month in months_keep:
                if month not in months_keep:
                    months_error.append(month)
            if len(months_error) > 0:
                raise ValueError(f'months={months_error} are not in available months: {file_month}')

        data_files_load = [os.path.join(comp_dir, 'hist', file) for i, file in enumerate(data_files_all) if
                           year_last_use >= file_year[i] >= year_first_use and file_month[i] in months_keep]
    if apply_month_shift_fix and hist_file == 0:
        ds = xr.open_mfdataset(data_files_load, decode_times=False, concat_dim='time', combine='nested')
        return ds_month_shift(ds, decode_times, months_keep)
    else:
        return xr.open_mfdataset(data_files_load, decode_times=decode_times, concat_dim='time', combine='nested')


def ds_month_shift(ds: xr.Dataset, decode_times: bool = True,
                   months_in_ds: Optional[List] = None):
    """
    When loading CESM data, for some reason the first month is marked as February, so this function
    shifts the time variable to correct it to January.

    Args:
        ds: Dataset to apply the shift to.
            It should have been loaded with `decode_times=False`.
        decode_times: If `True`, will convert time to actual date.
        months_in_ds: List of months which are in `ds` for each year.
            `1` refers to January.
            If `None`, will assume there are all 12 months.

    Returns:
        Dataset with first months shifted by -1 so now first month is January.
    """
    n_day_month = np.asarray(
        [cftime.DatetimeNoLeap(1, i + 1, 1).daysinmonth for i in range(12)])  # number of days in each month
    if months_in_ds is None:
        months_in_ds = np.arange(1, 13)
    n_day_month = n_day_month[np.asarray(months_in_ds) - 1]
    n_months_in_ds = ds.time.size
    n_years_in_ds = int(np.floor(n_months_in_ds / 12))
    month_shift_array = np.concatenate((np.tile(n_day_month, n_years_in_ds),
                                        n_day_month[:n_months_in_ds % 12]))
    ds_new = ds.assign_coords({'time': ('time', ds.time.values - month_shift_array, ds.time.attrs)})
    if decode_times:
        ds_new = xr.decode_cf(ds_new)
    return ds_new
