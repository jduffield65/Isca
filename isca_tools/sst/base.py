import numpy as np
from netCDF4 import Dataset
import os
from ..utils.load import load_namelist
from typing import List, Optional, Union
import xarray as xr


def write_sst(file_name: str, namelist_file: str, sst_array: np.ndarray):
    """
    This function generates a *.nc* file containing the variable `sst`, indicating the sst values at each coordinate.

    Args:
        file_name: *.nc* file containing the land coordinates and corresponding topography will be saved with this name
            in the folder given by `input_dir` in the `experiment_details` namelist of the `namelist_file`.
        namelist_file: File path to namelist `nml` file for the experiment.
            This specifies the physical parameters e.g. resolution used for the simulation.
        sst_array: `float [n_lat x n_lon]`
            SST values at each coordinate in Kelvin.

    """
    namelist = load_namelist(namelist_file=namelist_file)
    # Load in grid file containing longitude/latitude info for the resolution used for this experiment
    res = int(namelist['experiment_details']['resolution'][1:])  # resolution for experiment read in
    grid_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'time_series', 'grid_files')
    grid_file = os.path.join(grid_dir, f"t{res}_grid.nc")

    resolution_file = Dataset(grid_file, 'r', format='NETCDF3_CLASSIC')
    lons = resolution_file.variables['lon'][:]
    lats = resolution_file.variables['lat'][:]
    nlon = lons.shape[0]
    nlat = lats.shape[0]


    # Write land and topography arrays to file
    file_name = file_name.replace('.nc', '')
    file_name = file_name + '.nc'
    file_name = os.path.join(namelist['experiment_details']['input_dir'], file_name)
    if os.path.exists(file_name):
        raise ValueError(f"The file {file_name} already exists. Delete or re-name this to continue.")
    sst_file = Dataset(file_name, 'w', format='NETCDF3_CLASSIC')
    sst_file.createDimension('lat', nlat)
    sst_file.createDimension('lon', nlon)
    latitudes = sst_file.createVariable('lat','f4',('lat',))
    longitudes = sst_file.createVariable('lon','f4',('lon',))
    sst_array_netcdf = sst_file.createVariable('sst','f4',('lat','lon',))
    latitudes[:] = lats
    longitudes[:] = lons
    sst_array_netcdf[:] = sst_array
    sst_file.close()
    print('Output written to: ' + file_name)
