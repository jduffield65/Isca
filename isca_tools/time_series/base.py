import os
from typing import Callable, Optional, Sequence, Union

import numpy as np

from isca_tools.run.create_files import create_grid_file as _create_grid_file
from isca_tools.run.create_files import create_time_arr as _create_time_arr
from isca_tools.run.create_files import write_var as _write_var
from isca_tools.utils.load import load_namelist


def create_grid_file(*args, **kwargs):
    """Backward-compatible wrapper for creating a grid file."""
    return _create_grid_file(*args, **kwargs)


def create_time_arr(*args, **kwargs):
    """Backward-compatible wrapper for creating time coordinates."""
    return _create_time_arr(*args, **kwargs)


def create_time_series_file(
    file_name: str,
    namelist_file: str,
    var_name: str,
    var_func: Callable,
    time_spacing: int,
    exp_dir: Optional[str] = None,
    grid_file: Optional[str] = None,
    pressure_var: Optional[np.ndarray] = None,
    **kwargs,
) -> str:
    """
    Create an Isca time-varying input file from a user-supplied function.

    This is a convenience wrapper around :mod:`isca_tools.run.create_files.write_var`.
    The callback should produce values for the supplied time, pressure, latitude and longitude arrays,
    with the time axis first. For example::

        def co2_func(days, pressure, lat, lon):
            arr = np.ones((len(days), len(pressure), len(lat), len(lon))) * 600
            arr[days < 360 * 5] = 300
            return arr

    Args:
        file_name: Output file name, usually ending in ``.nc``.
        namelist_file: Path to the experiment namelist.
        var_name: Name to store in the NetCDF file.
        var_func: Function returning the variable values on the chosen grid.
        time_spacing: Interval between values in days.
        exp_dir: Optional override for the experiment directory. Defaults to the namelist directory.
        grid_file: Optional explicit grid file path. If omitted one is created for the experiment resolution.
        pressure_var: Optional pressure coordinate override.
    """
    if exp_dir is None:
        exp_dir = os.path.dirname(os.path.abspath(namelist_file))

    namelist = load_namelist(namelist_file=os.path.join(exp_dir, os.path.basename(namelist_file)))
    res = int(namelist['experiment_details']['resolution'][1:])

    if grid_file is None:
        grid_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'run', 'grid_files', f't{res}_grid.nc')
        if not os.path.exists(grid_file):
            create_grid_file(res=res)

    if os.path.exists(grid_file):
        import xarray as xr
        grid = xr.open_dataset(grid_file)
        lat_var = np.asarray(grid.coords['lat'].values)
        lon_var = np.asarray(grid.coords['lon'].values)
        if pressure_var is None and 'pfull' in grid.coords:
            pressure_var = np.asarray(grid.coords['pfull'].values)
    else:
        raise FileNotFoundError(f"Grid file does not exist: {grid_file}")

    main_nml = namelist.get('main_nml', {})
    days_total = int(namelist['experiment_details']['n_months_total'] * main_nml.get('days', 30))
    time_var = np.arange(0, days_total + 1, time_spacing)

    if pressure_var is not None:
        sample = var_func(time_var, pressure_var, lat_var, lon_var)
        write_var_kwargs = {
            'time_var': time_var,
            'pressure_var': pressure_var,
            'lat_var': lat_var,
            'lon_var': lon_var,
            'var_array': sample,
            'var_name': var_name,
            'namelist_file': os.path.basename(namelist_file),
            'exp_dir': exp_dir,
            'grid_file': grid_file,
        }
    else:
        sample = var_func(time_var, lat_var, lon_var)
        write_var_kwargs = {
            'time_var': time_var,
            'lat_var': lat_var,
            'lon_var': lon_var,
            'var_array': sample,
            'var_name': var_name,
            'namelist_file': os.path.basename(namelist_file),
            'exp_dir': exp_dir,
            'grid_file': grid_file,
        }

    _write_var(file_name=file_name, **write_var_kwargs)
    return os.path.abspath(file_name)


def write_var(*args, **kwargs):
    """Backward-compatible wrapper for the general variable writer."""
    return _write_var(*args, **kwargs)
