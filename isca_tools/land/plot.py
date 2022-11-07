import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import numpy as np


def show_land(land_file: str):
    """
    Quick function to show where land is and any topography

    Args:
        land_file: Location of *.nc* file created using the `write_land` function.

    """
    land_data = Dataset(land_file, 'r', format='NETCDF3_CLASSIC')
    lons = land_data.variables['lon'][:]
    lats = land_data.variables['lat'][:]
    lon_array, lat_array = np.meshgrid(lons, lats)
    lon_0 = lons.mean()
    lat_0 = lats.mean()
    m = Basemap(lat_0=lat_0, lon_0=lon_0)
    xi, yi = m(lon_array, lat_array)
    plt.figure()
    land_array = np.asarray(land_data.variables['land_mask'])
    if land_array.max() > 0:
        m.contour(xi, yi, land_array)
    topo_array = np.asarray(land_data.variables['zsurf'])
    if topo_array.max() > 0:
        cs = m.contourf(xi, yi, topo_array, cmap=plt.get_cmap('RdBu_r'))
        cb = plt.colorbar(cs, shrink=0.5, extend='both')
    plt.xticks(np.linspace(0, 360, 13))
    plt.yticks(np.linspace(-90, 90, 7))
    plt.show()
