import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from typing import Optional


def show_land(land_file: str, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None):
    """
    Quick function to show where land is and any topography

    Args:
        land_file: Location of *.nc* file created using the `write_land` function.
        fig: Figure to show where land is.
        ax: Axes in Figure to show where land is.
    """
    land_data = Dataset(land_file, 'r', format='NETCDF3_CLASSIC')
    lons = land_data.variables['lon'][:]
    lats = land_data.variables['lat'][:]
    lon_array, lat_array = np.meshgrid(lons, lats)
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.gca()
    land_array = np.asarray(land_data.variables['land_mask'])
    if land_array.max() > 0:
        ax.contour(lon_array, lat_array, land_array, levels=[0.9, 1.1], linewidths=3, colors='k')
    topo_array = np.asarray(land_data.variables['zsurf'])
    if topo_array.max() > 0:
        cs = ax.contourf(lon_array, lat_array, topo_array, cmap=plt.get_cmap('RdBu_r'))
        cb = plt.colorbar(cs, shrink=0.5, extend='both')
    else:
        # If no topography, show ocean as blue and land as sand coloured
        topo_array[land_array > 0] = 1
        cs = ax.pcolormesh(lon_array, lat_array, topo_array, cmap=plt.get_cmap('Paired'), vmin=-0.2, vmax=1.2)
    ax.set_xticks(np.linspace(0, 360, 13))
    ax.set_yticks(np.linspace(-90, 90, 7))
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
