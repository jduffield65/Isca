import os
import xarray as xr
from typing import List, Optional
import f90nml
import warnings


def get_file_suffix(dir: str, suffix: str) -> List[str]:
    """
    Returns a list of all files in `dir` which end in `suffix`.

    Args:
        dir: Directory of interest.
        suffix: Usually the file type of interest e.g. `.nml` or `.txt`.

    Returns:
        List of all files with the correct `suffix`.
    """
    file_name = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            file_name += [file]
    return file_name


def load_dataset(exp_name: str, run_no: Optional[int] = None,
                 data_dir: Optional[str] = None, decode_times: bool = False,
                 use_cftime: bool = True) -> xr.Dataset:
    """
    This loads a dataset produced by Isca containing all the diagnostics specified.

    Args:
        exp_name: Name of folder in `data_dir` where data for this experiment was saved.
        run_no: Data is saved at intervals in time specified in the `main_nml` namelist of the *namelist.nml* file.
            This is typically monthly and the `run_no` would then refer to the month to load data for.
            If `None`, data for all months is loaded and combined into a single Dataset.
        data_dir: Directory which contains the `exp_name` directory. If `None`, will assume this is
            the directory specified through the environmental variable `GFDL_DATA`.
        decode_times: If `True`, decode times into datetime objects
        use_cftime: If `True`, will decode times into `cftime.datetime` objects,
            rather than `np.datetime64` objects. Only relevant if `decode_times` is `True`.
            Useful if using 360 day calendar.

    Returns:
        Dataset containing all diagnostics specified for the experiment.

    """
    if data_dir is None:
        data_dir = os.environ['GFDL_DATA']
    exp_dir = os.path.join(data_dir, exp_name)

    # Get index of first run file i.e. which month saved first
    files_run = [filename for filename in os.listdir(exp_dir) if filename.startswith('run')]
    files_run.sort()
    first_month = int(files_run[0][-4:])
    if first_month != 1:
        warnings.warn(f'First month saved is {first_month} not 1.')

    # File name is the same for all runs and is the only file with the suffix '.nc' in the run folder
    file_name = get_file_suffix(os.path.join(exp_dir, 'run%04d' % first_month), '.nc')[0]

    if run_no is None:
        data_file = os.path.join(exp_dir, 'run*', file_name)
        d = xr.open_mfdataset(data_file, concat_dim='time', combine='nested',
                              decode_times=decode_times, use_cftime=use_cftime)
    else:
        data_file = os.path.join(exp_dir, 'run%04d' % run_no, file_name)
        d = xr.open_dataset(data_file, decode_times=decode_times, use_cftime=use_cftime)
    return d


def load_namelist(exp_name: Optional[str] = None, data_dir: Optional[str] = None,
                  namelist_file: Optional[str] = None) -> f90nml.Namelist:
    """
    Returns all the namelists options and their corresponding values specified in the namelist *.nml* file
    for the experiment indicated by `exp_name`.

    Args:
        exp_name: Name of folder in `data_dir` where data for this experiment was saved.
        data_dir: Directory which contains the `exp_name` directory. If `None`, will assume this is
            the directory specified through the environmental variable `GFDL_DATA`.
        namelist_file: Path to the namelist *.nml* file to load. Use this option if data for the experiment has not
            been created yet.
    Returns:
        Namelist values used for this experiment.
    """
    if namelist_file is not None:
        # Make sure file_path has the .nml suffix
        file_path = namelist_file.replace('.nml', '')
        file_path = file_path + '.nml'
    else:
        if data_dir is None:
            data_dir = os.environ['GFDL_DATA']
        exp_dir = os.path.join(data_dir, exp_name)

        # Get index of first run file i.e. which month saved first
        files_run = [filename for filename in os.listdir(exp_dir) if filename.startswith('run')]
        files_run.sort()
        first_month = int(files_run[0][-4:])

        # Namelist file_name is the same for all runs and is the only file with the suffix '.nml' in the run folder
        file_name = get_file_suffix(os.path.join(exp_dir, 'run%04d' % first_month), '.nml')[0]
        file_path = os.path.join(exp_dir, 'run%04d' % first_month, file_name)
    return f90nml.read(file_path)
