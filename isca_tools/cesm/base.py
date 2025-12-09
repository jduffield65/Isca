import numpy as np
import xarray as xr
from typing import Union, Optional, List
import f90nml

from ..utils.base import parse_int_list, split_list_max_n
from .load import get_exp_file_dates
import os


def get_pressure(ps: Union[np.ndarray, xr.DataArray], p0: float, hya: Union[np.ndarray, xr.DataArray],
                 hyb: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculates pressure at the hybrid levels, similar to
    [NCAR function](https://www.ncl.ucar.edu/Document/Functions/Built-in/pres_hybrid_ccm.shtml).

    Args:
        ps: `float [n_time x n_lat x n_lon]`</br>
            Array of surface pressures in units of *Pa*.</br>
            `PS` in CESM atmospheric output.
        p0: Surface reference pressure in *Pa*.</br>
            `P0` in CESM atmospheric output.
        hya: `float [n_levels]`</br>
            Hybrid A coefficients.</br>
            `hyam` in CESM atmospheric output.
        hyb: `float [n_levels]`</br>
            Hybrid B coefficients.</br>
            `hybm` in CESM atmospheric output.

    Returns:
        `float [n_time x n_levels x n_lat x n_lon]`
            Pressure at the hybrid levels in *Pa*.
    """
    return hya * p0 + hyb * ps

def create_per_job_nml(input_file_path: str, files_per_job: int = 1, years_per_job: Optional[int] = None,
                       exist_ok: Optional[bool] = None) -> List:
    """
    Splits up list of all input files (or years) into separate lists of no more than `files_per_job` (`years_per_job`) in each.
    A `nml` file is then created for each of these with name same as `input_file_path` but with first file (year)
    in job as a suffix.

    E.g. `input_nml` becomes `input0.nml` with `input_info['script_info']['ind_files']` set to file
    indices to run for that job.

    E.g. `input_nml` becomes `input1985.nml` with `input_info['script_info']['year_files']` set to
    years to run for that job.

    Args:
        input_file_path: Path to `nml` file for experiment.
        files_per_job: Number of files to consider for each job.
        years_per_job: Number of years to run for each job. Overrides `files_per_job` if provided.
        exist_ok: If `True`, do not raise exception if any file to be created already exists.
            If `False`, will overwrite it. If `None` leaves the existing file unchanged.

    Returns:
        List of paths to nml files created e.g. `['/Users/.../input1985.nml', '/Users/.../input1985.nml']`
    """
    input_info = f90nml.read(input_file_path)

    # Determine which years to get data for
    file_dates = get_exp_file_dates(input_info['script_info']['exp_name'], 'atm',
                                    input_info['script_info']['archive_dir'], input_info['script_info']['hist_file'])
    year_files_all = np.unique(file_dates.dt.year).tolist()
    file_ind_all = np.arange(len(file_dates)).tolist()
    if years_per_job is not None:
        if input_info['script_info']['year_files'] is None:
            file_consider = year_files_all
        else:
            file_consider = parse_int_list(input_info['script_info']['year_files'], lambda x: int(x),
                                            all_values=year_files_all)
    else:
        if input_info['script_info']['ind_files'] is None:
            file_consider = file_ind_all
        else:
            file_consider = parse_int_list(input_info['script_info']['ind_files'], lambda x: int(x),
                                           all_values=file_ind_all)
    n_digit = np.max([len(str(val)) for val in file_consider])          # how many digits to use for file name
    file_jobs = split_list_max_n(file_consider, years_per_job if years_per_job is not None else files_per_job)
    out_file_names = []
    for ind in file_jobs:
        if years_per_job is not None:
            input_info['script_info']['year_files'] = ind
        else:
            input_info['script_info']['ind_files'] = ind
        out_file_names.append(input_file_path.replace('.nml', f'{ind[0]:0{n_digit}d}.nml'))
        if os.path.exists(out_file_names[-1]):
            if exist_ok is None:
                print(f'{ind}: Output nml file already exists. Leaving unchanged')
                continue
        input_info.write(out_file_names[-1], force=exist_ok)
        print(f'{ind}: Output nml file created')
    return out_file_names
