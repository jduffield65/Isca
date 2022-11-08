import numpy as np
from netCDF4 import Dataset
import os
from ..utils.load import load_namelist
from .mountains import gaussian_mountain, mountain_range_height
from typing import List, Optional, Union


def write_land(file_name: str, namelist_file: str, land_mode: Optional[str] = None,
               boundaries: Optional[List[float]] = None, continents: Union[List[str], str] = 'all',
               topography: Optional[str] = None,
               topography_gauss: Optional[List[float]] = None, waterworld: bool = False):
    """
    This function generates a *.nc* file containing the variable `land_mask`, indicating the coordinates where land is,
    and `zsurf`, indicating the topography at each coordinate.

    Extended from an
    [Isca script](https://github.com/ExeClim/Isca/blob/master/src/extra/python/isca/land_generator_fn.py).

    Args:
        file_name: *.nc* file containing the land coordinates and corresponding topography will be saved with this name
            in the folder given by `input_dir` in the `experiment_details` namelist of the `namelist_file`.
        namelist_file: File path to namelist `nml` file for the experiment.
            This specifies the physical parameters e.g. resolution used for the simulation.
        land_mode: Type of land to use for the experiment. There are three options:

            * `None`: No land.
            * `square`: Square block of land with boundaries specified by the `boundaries` variable.
            * `continents`: Use all or a subset of Earth's continents as set by the `continents` variable.
        boundaries: `float [4]`.
            The `[South, North, West, East]` boundaries of the land in degrees.
            Only required if `land_mode = square`.
        continents: There are 7 possible continents:

            * `NA`: North America
            * `SA`: South America
            * `EA`: Eurasia
            * `AF`: Africa
            * `OZ`: Australia
            * `IN`: India
            * `SEA`: South East Asia

            If `continents = all`, all of the above will be used.</br>
            If `continents = old`, `[NA, SA, EA, AF]` will be used.</br>
            Otherwise, you can give a list indicating a subset of continents e.g. `[NA, SA]`.
        topography: Type of topography to use for the experiment. There are five options:

            * `None`: No topography.
            * `gaussian`: A single mountain with location and height specified through the `topography_gauss` variable.
            * `rockys`: Just the Rocky mountain range.
            * `tibet`: Just Tibet.
            * `all`: Includes the Rockys and Tibet.
        topography_gauss: `float [5]`.</br>
            List containing:

            * Central latitude of mountain (degrees).
            * Central longitude of mountain (degrees).
            * Radius of mountain in degrees. Typical would be 20.
            * Standard deviation indicating how steep the mountain is. The smaller the value, the steeper the
            mountain. Units are degrees and typical value would be 10.
            * Height of mountain peak in meters.
        waterworld: If `False`, topography is not allowed where there is no land.
            Otherwise, *aquamountains* are possible.

    """

    namelist = load_namelist(namelist_file=namelist_file)
    # Load in grid file containing longitude/latitude info for the resolution used for this experiment
    res = int(namelist['experiment_details']['resolution'][1:])  # resolution for experiment read in
    grid_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'time_series', 'grid_files')
    grid_file = os.path.join(grid_dir, f"t{res}_grid.nc")

    resolution_file = Dataset(grid_file, 'r', format='NETCDF3_CLASSIC')
    lons = resolution_file.variables['lon'][:]
    lats = resolution_file.variables['lat'][:]
    # lonb = resolution_file.variables['lonb'][:]
    # latb = resolution_file.variables['latb'][:]
    nlon = lons.shape[0]
    nlat = lats.shape[0]

    # make 2d arrays of latitude and longitude
    lon_array, lat_array = np.meshgrid(lons, lats)
    # lonb_array, latb_array = np.meshgrid(lonb, latb)

    # Configure where land is
    land_array = np.zeros((nlat, nlon))
    if land_mode is None:
        pass
    elif land_mode.lower() == 'square':
        if boundaries is None:
            raise ValueError(f"boundaries is {None} but should be a list of 4 values indicating the "
                             f"[south, north, west, east] boundaries of the land.")
        idx = (boundaries[0] <= lat_array) & (lat_array < boundaries[1]) & (boundaries[2] < lon_array) & (
                boundaries[3] > lon_array)
        land_array[idx] = 1
    elif land_mode.lower() == 'continents':
        if isinstance(continents, str):
            if continents.lower() == 'all':
                # All continents
                continents = ['NA', 'SA', 'EA', 'AF', 'OZ', 'IN', 'SEA']
            elif continents.lower() == 'old':
                # Continents from the original continent set-up adapted from the Sauliere 2012 paper (Jan 16)
                continents = ['NA', 'SA', 'EA', 'AF']
            else:
                raise ValueError(f"continents given as {continents} but must be 'all', 'old' or a list of continents.")
        idx = continent_land_location(lon_array, lat_array, continents[0])
        for i in range(1, len(continents)):
            idx += continent_land_location(lon_array, lat_array, continents[i])
        land_array[idx] = 1
    else:
        raise ValueError(f"land_mode was given as {land_mode} but must be None, 'square' or 'continents'.")

    # Configure topography
    topo_array = np.zeros((nlat, nlon))   # height field
    if topography is None:
        pass
    elif topography.lower() == 'gaussian':
        topo_array += gaussian_mountain(lon_array, lat_array, topography_gauss[0], topography_gauss[1],
                                        topography_gauss[2], topography_gauss[3], topography_gauss[4])
    elif topography.lower() == 'all':
        # Add both rockys and tibet from Sauliere 2012
        topo_array += mountain_range_height(lon_array, lat_array, 'rockys')
        topo_array += mountain_range_height(lon_array, lat_array, 'tibet')
    elif topography.lower() == 'tibet':
        topo_array += mountain_range_height(lon_array, lat_array, 'tibet')
    elif topography.lower() == 'rockys':
        topo_array += mountain_range_height(lon_array, lat_array, 'rockys')
    else:
        raise ValueError(f"topography given as {topography} but must be None, 'all', 'rockys', 'tibet' or 'gaussian'.")
    if not waterworld:
        # Don't allow topography where there is no land
        topo_array[(land_array == 0) & (topo_array != 0)] = 0

    # Write land and topography arrays to file
    file_name = file_name.replace('.nc', '')
    file_name = file_name + '.nc'
    file_name = os.path.join(namelist['experiment_details']['input_dir'], file_name)
    if os.path.exists(file_name):
        raise ValueError(f"The file {file_name} already exists. Delete or re-name this to continue.")
    topo_file = Dataset(file_name, 'w', format='NETCDF3_CLASSIC')
    topo_file.createDimension('lat', nlat)
    topo_file.createDimension('lon', nlon)
    latitudes = topo_file.createVariable('lat','f4',('lat',))
    longitudes = topo_file.createVariable('lon','f4',('lon',))
    topo_array_netcdf = topo_file.createVariable('zsurf','f4',('lat','lon',))
    land_array_netcdf = topo_file.createVariable('land_mask','f4',('lat','lon',))
    latitudes[:] = lats
    longitudes[:] = lons
    topo_array_netcdf[:] = topo_array
    land_array_netcdf[:] = land_array
    topo_file.close()
    print('Output written to: ' + file_name)


def continent_land_location(lon_array: np.ndarray, lat_array: np.ndarray, continent: str = 'NA') -> np.ndarray:
    """
    Returns a boolean array indicating the latitude, longitude coordinates containing land for a particular `continent`.

    Args:
        lon_array: `float [n_lat x n_lon]`.</br>
            Array indicating the longitude at each (latitude, longitude) coordinate in the grid used for the experiment.
        lat_array: `float [n_lat x n_lon]`.</br>
            Array indicating the latitude at each (latitude, longitude) coordinate in the grid used for the experiment.
        continent: There are 7 options indicating different continents:

            * `NA`: North America
            * `SA`: South America
            * `EA`: Eurasia
            * `AF`: Africa
            * `OZ`: Australia
            * `IN`: India
            * `SEA`: South East Asia

    Returns:
        `land_loc`: `bool [n_lat x n_lon]`</br>
            `land_loc[lat, lon]` will  be `True` if the `continent` contains land at the coordinate (`lat`, `lon`).
    """
    if continent.upper() == 'NA':
        land_loc = (103. - 43. / 40. * (lon_array - 180) < lat_array) & (
                (lon_array - 180) * 43. / 50. - 51.8 < lat_array) & (lat_array < 60.)
    elif continent.upper() == 'SA':
        land_loc = (737. - 7.2 * (lon_array - 180) < lat_array) & (
                (lon_array - 180) * 10. / 7. + -212.1 < lat_array) & (lat_array < -22. / 45 * (lon_array - 180) + 65.9)
    elif continent.upper() == 'EA':
        eurasia_pos = (17. <= lat_array) & (lat_array < 60.) & (-5. < lon_array) & (
                43. / 40. * lon_array - 101.25 < lat_array)
        eurasia_neg = (17. <= lat_array) & (lat_array < 60.) & (355. < lon_array)
        land_loc = eurasia_pos + eurasia_neg
    elif continent.upper() == 'AF':
        africa_pos = (lat_array < 17.) & (-52. / 27. * lon_array + 7.37 < lat_array) & (
                    52. / 38. * lon_array - 65.1 < lat_array)
        africa_neg = (lat_array < 17.) & (-52. / 27. * (lon_array - 360) + 7.37 < lat_array)
        land_loc = africa_pos + africa_neg
    elif continent.upper() == 'OZ':
        land_loc = (lat_array > - 35.) & (lat_array < -17.) & (lon_array > 115.) & (lon_array < 150.)
    elif continent.upper() == 'IN':
        land_loc = (lat_array < 23.) & (-15. / 8. * lon_array + 152 < lat_array) & (
                    15. / 13. * lon_array - 81 < lat_array)
    elif continent.upper() == 'SEA':
        land_loc = (lat_array < 23.) & (43. / 40. * lon_array - 101.25 < lat_array) & (
                    -14. / 13. * lon_array + 120 < lat_array)
    else:
        raise ValueError(f"Continent given was {continent} but it must be either 'NA', 'SA', 'EA', 'AF', 'OZ', 'IN' or "
                         f"'SEA'")
    return land_loc
