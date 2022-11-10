import numpy as np
from typing import Optional
import os
from netCDF4 import Dataset
from ..utils import load_namelist
import numpy_indexed


def load_land_file(namelist_file: Optional[str] = None, land_file: Optional[str] = None) -> Dataset:
    """
    Loads in the land data file for a given experiment.

    Args:
        namelist_file: File path to namelist `nml` file for the experiment.
            The location of the land `nc` file will be obtained from `input_dir` and `land_file_name` in this.
            Not required if `land_file` given.
        land_file: File path to the land `nc` file used for the experiment.
            Not required if `namelist_file` given.

    Returns:
        land dataset containing the variables `land_mask`, `zsurf`, `lat` and `lon`.
    """
    if land_file is None:
        if namelist_file is None:
            raise ValueError(f"Atleast one of namelist_file or land_file must be specified but both are None")
        # Determine land file name from info in namelist
        namelist = load_namelist(namelist_file=namelist_file)
        land_file = os.path.join(namelist['experiment_details']['input_dir'],
                                 namelist['idealized_moist_phys_nml']['land_file_name'].replace('INPUT/', ''))
    land_data = Dataset(land_file, 'r', format='NETCDF3_CLASSIC')
    return land_data


def get_land_coords(namelist_file: Optional[str] = None, land_file: Optional[str] = None) -> [np.ndarray, np.ndarray]:
    """
    Returns the latitude and longitude coordinates that correspond to land for a particular experiment.

    Args:
        namelist_file: File path to namelist `nml` file for the experiment.
            The location of the land `nc` file will be obtained from `input_dir` and `land_file_name` in this.
            Not required if `land_file` given.
        land_file: File path to the land `nc` file used for the experiment.
            Not required if `namelist_file` given.

    Returns:
        `land_lat`: `float [n_land_coords]`</br>
            Land is present at the coordinate indicated by (`land_lat[i]`, `land_lon[i]`) for all `i`.</br>
            Units are degrees ($-180 \leq \phi \leq 180$).
        `land_lon`: `float [n_land_coords]`</br>
            Land is present at the coordinate indicated by (`land_lat[i]`, `land_lon[i]`) for all `i`.</br>
            Units are degrees ($0 \leq \lambda \leq 360$).
    """
    land_data = load_land_file(namelist_file, land_file)
    # Get 2 arrays, one for latitude and one for longitude indicating where the indices where land is
    land_ind = np.where(land_data.variables['land_mask'][:] > 0)
    # See which dimension corresponds to which array (either 0 or 1)
    lat_dim_ind = np.where(np.asarray(land_data.variables['land_mask'].dimensions) == 'lat')[0][0]
    lon_dim_ind = np.where(np.asarray(land_data.variables['land_mask'].dimensions) == 'lon')[0][0]

    land_lat = np.asarray(land_data.variables['lat'][land_ind[lat_dim_ind]])
    land_lon = np.asarray(land_data.variables['lon'][land_ind[lon_dim_ind]])
    return land_lat, land_lon


def get_ocean_coords(namelist_file: Optional[str] = None, land_file: Optional[str] = None) -> [np.ndarray, np.ndarray]:
    """
    Returns the latitude and longitude coordinates that correspond to ocean for a particular experiment.

    Args:
        namelist_file: File path to namelist `nml` file for the experiment.
            The location of the land `nc` file will be obtained from `input_dir` and `land_file_name` in this.
            Not required if `land_file` given.
        land_file: File path to the land `nc` file used for the experiment.
            Not required if `namelist_file` given.

    Returns:
        `ocean_lat`: `float [n_ocean_coords]`</br>
            Ocean is present at the coordinate indicated by (`ocean_lat[i]`, `ocean_lon[i]`) for all `i`.</br>
            Units are degrees ($-180 \leq \phi \leq 180$).
        `ocean_lon`: `float [n_ocean_coords]`</br>
            Ocean is present at the coordinate indicated by (`ocean_lat[i]`, `ocean_lon[i]`) for all `i`.</br>
            Units are degrees ($0 \leq \lambda \leq 360$).
    """
    # Load land info
    land_data = load_land_file(namelist_file, land_file)
    land_lat, land_lon = get_land_coords(namelist_file, land_file)
    lat_lon_land = np.concatenate((land_lat.reshape(-1, 1), land_lon.reshape(-1, 1)), axis=1)

    # Get grid of all possible coordinates - land or ocean
    lat_all, lon_all = np.meshgrid(np.asarray(land_data.variables['lat'][:]),
                                   np.asarray(land_data.variables['lon'][:]))
    lat_lon_all = np.concatenate((lat_all.reshape(-1, 1), lon_all.reshape(-1, 1)), axis=1)

    # Find index of land coords in lat_lon_all
    land_inds = numpy_indexed.indices(lat_lon_all, lat_lon_land)
    if len(land_inds) != len(land_lat):
        raise ValueError(f"There are {len(land_lat)} land coordinates but {len(land_inds)} found in full "
                         f"latitude/longitude coordinate grid.")

    # Ocean coords are those which are not land
    ocean_inds = np.setdiff1d(np.arange(lat_lon_all.shape[0]), land_inds)
    lat_lon_ocean = lat_lon_all[ocean_inds]
    return lat_lon_ocean[:, 0], lat_lon_ocean[:, 1]
