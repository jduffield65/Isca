import os
import xarray as xr
from typing import List, Optional
import f90nml
import warnings

jasmin_archive_dir = '/gws/nopw/j04/global_ex/jamd1/cesm/CESM2.1.3/archive/'


def load_dataset(exp_name: str, comp: str = 'atm',
                 archive_dir: str = jasmin_archive_dir,
                 decode_times: bool = True) -> xr.Dataset:
    """
    This loads a dataset of a given componen produced by CESM.

    Args:
        exp_name: Name of folder in `archive_dir` where data for this experiment was saved.
        comp: Component of CESM to load data from.</br>
            Options are:

            * `atm`: atmosphere
            * `ice`: ice
            * `lnd`: land
            * `rof`: river
        archive_dir: Directory where CESM archive data saved.
        decode_times: If `True`, will convert time to actual date.

    Returns:
        Dataset containing all diagnostics specified for the experiment.
    """
    # LHS of comp_dir_file is name of directory containing hist files for the component
    # RHS of comp_dir_file is the string indicating the component in the individual .nc files within this directory
    comp_dir_file = {'atm': 'cam',          # atmosphere
                     'ice': 'cice',         # ice
                     'lnd': 'clm2',         # land
                     'rof': 'mosart'}       # river

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
    # * indicates where date index info is, so we combine all datasets
    data_files = os.path.join(comp_dir, 'hist', f'{exp_name}.{comp_file}.h0.*.nc')
    d = xr.open_mfdataset(data_files, decode_times=decode_times, concat_dim='time', combine='nested')
    # TODO: give option to load in specific dates / range of dates
    return d
